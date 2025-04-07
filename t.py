import sqlite3
from annoy import AnnoyIndex
from collections import Counter
import os

# === 1. Get database connection ===
def get_db_connection(db_path):
    try:
        conn = sqlite3.connect(db_path)
        print("Connected to database.")
        return conn
    except sqlite3.Error as e:
        print(f"❌ Error connecting to DB: {e}")
        return None

# === 2. Build Annoy index from embeddings in DB ===
def build_annoy_index_from_db(dimension, index_file, db_path):
    conn = get_db_connection(db_path)
    if conn is None:
        return [], []

    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE tbl_register_faces SET account_id = id")
        cursor.execute("SELECT image_vector_process, account_id FROM tbl_register_faces")
    except sqlite3.Error as e:
        print(f"SQL Error: {e}")
        conn.close()
        return [], []

    labels = []
    account_ids = []
    index = AnnoyIndex(dimension, 'angular')

    for i, row in enumerate(cursor.fetchall()):
        embedding_blob, account_id = row
        try:
            # Parse embedding string safely
            clean_str = embedding_blob.replace('[', '').replace(']', '').replace('\n', '').strip()
            embedding = list(map(float, clean_str.split(',')))
        except Exception as e:
            print(f"⚠️ Error parsing embedding on row {i}: {e}")
            continue

        if len(embedding) != dimension:
            print(f"⚠️ Skipping row {i} due to incorrect dimension: {len(embedding)}")
            continue

        labels.append(i)
        account_ids.append(account_id)
        index.add_item(i, embedding)
        print(f"✔️ Added account_id: {account_id}, vector length: {len(embedding)}")

    conn.close()

    if labels:
        index.build(50)
        index.save(index_file)
        print(f"✅ Annoy index successfully saved at: {index_file}")
    else:
        print("⚠️ No valid embeddings to build Annoy index.")

    return labels, account_ids

# === 3. Search for a query embedding in the index ===

def search_in_annoy_index(query_embedding, index_file, dimension, account_ids, threshold=0.4):

    index = AnnoyIndex(dimension, 'angular')
    index.load(index_file)
    print(f" Loaded Annoy index from: {index_file}")

    # Get the single nearest neighbor
    nearest_indices, distances = index.get_nns_by_vector(query_embedding, n=1, include_distances=True)

    if not nearest_indices or nearest_indices[0] >= len(account_ids):
        print("❌ Unable to find a matching account ID.")
        return {"account_id": None, "distance": None}

    nearest_index = nearest_indices[0]
    nearest_distance = distances[0]
    nearest_account_id = account_ids[nearest_index]

    print(f"🏷 Nearest account_id: {nearest_account_id}")
    print(f"📏 Distance: {nearest_distance:.4f}")

    if nearest_distance <= threshold:
        return {
            "account_id": nearest_account_id,
            "distance": nearest_distance,
        }
    else:

        return {"account_id": "unknown", "distance": "unknown"}

# === 4. Optional: Get a vector from DB by account_id (to test self-match)
def get_embedding_by_account_id(account_id, db_path, dimension):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT image_vector_process FROM tbl_register_faces WHERE account_id = ?", (account_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        vector_str = result[0].replace('[', '').replace(']', '').replace('\n', '')
        vector = list(map(float, vector_str.split(',')))
        if len(vector) == dimension:
            return vector
        else:
            print(f"⚠️ Embedding found but dimension mismatch: {len(vector)}")
    else:
        print("❌ No embedding found for this account_id.")
    return None

# === 5. Example usage ===
if __name__ == "__main__":
    dimension = 128
    db_path = "./face_recognition.db"
    index_file = os.path.abspath("face_index.ann")

    print("🔨 Building Annoy index...")
    labels, account_ids = build_annoy_index_from_db(dimension, index_file, db_path)

    if not labels:
        print("❌ No index built. Exiting.")
    else:
        # 💡 Replace with an existing account_id in your DB to test matching
        test_account_id = "1"  # <-- REPLACE with actual ID in your DB
        print(f"🔍 Searching for account_id: {test_account_id}...")

        query_embedding = get_embedding_by_account_id(test_account_id, db_path, dimension)

        if query_embedding:
            result = search_in_annoy_index(query_embedding, index_file, dimension, account_ids)
            print("🔎 Search Result:", result)
        else:
            print("❌ Could not retrieve query embedding.")
