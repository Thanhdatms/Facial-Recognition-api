import requests
from src.utils.image_processing import encode_image
from src.models.siamese_model import SiameseModel
from src.config.config import captured_folder, dimension, threshold, checkpoint_path
import base64

def fetch_embeddings_from_url(url):
    """Fetch image embeddings from the provided URL."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            embeddings = []
            for item in data:
                image_vector = item.get("image_vector")
                username = item.get("username")
                member_id = item.get("member_id")
                if image_vector and username and member_id:
                    embeddings.append({"username": username, "embedding": image_vector[1], "member_id": member_id})
            return embeddings
        else:
            print(f"Failed to fetch embeddings. Status code: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching embeddings: {e}")
        return []
    
def embed_image(file_path):
    """Encode image to a 128-dimension embedding."""
    try:
        model = SiameseModel(embedding_size=dimension)
        encodings = encode_image(file_path, model)
        if encodings:
            return encodings
        else:
            print(f"No face found in the image: {file_path}")
            return None
    except Exception as e:
        print(f"Error encoding {file_path}: {e}")
        return None


def image_to_base64(image_path):
    """Convert image to base64."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    
# def build_annoy_index_from_url(dimension, index_file, url):
#     """Build Annoy index using embeddings fetched from URL."""
#     embeddings = fetch_embeddings_from_url(url)
#     index = AnnoyIndex(dimension, 'angular')
#     global labels, usernames, member_ids
#     labels = []
#     usernames = []
#     member_ids = []  # Initialize member_ids list
#     for i, item in enumerate(embeddings):
#         embedding = item.get("embedding")
#         username = item.get("username")
#         member_id = item.get("member_id")
#         if embedding and username and member_id:
#             labels.append(i)
#             usernames.append(username)
#             member_ids.append(member_id)  # Store member_id
#             index.add_item(i, embedding)
#     if labels:
#         index.build(50)
#         index.save(index_file)
#         print(f"Built and saved Annoy index to {index_file}.")
#     else:
#         print("No valid embeddings to build Annoy index.")
#     return labels, usernames, member_ids  # Return member_ids


# def search_in_annoy_index(query_embedding, index_file, usernames, member_ids, n=10, threshold=0.36):
#     """Search for the closest match in the Annoy index."""
#     index = AnnoyIndex(dimension, 'angular')
#     index.load(index_file)
#     nearest_indices, distances = index.get_nns_by_vector(query_embedding, n=n, include_distances=True)

#     if not nearest_indices or len(usernames) <= max(nearest_indices):
#         print("Unable to find corresponding usernames in Annoy index.")
#         return {"username": None, "distance": None, "member_id": -1}

#     result_usernames = [usernames[i] for i in nearest_indices]
#     result_member_ids = [member_ids[i] for i in nearest_indices]
#     label_counts = Counter(result_usernames)
#     most_common_label, count = label_counts.most_common(1)[0]
#     avg_distance = sum(distances) / n

#     if avg_distance <= 0.36:
#         most_common_member_id = result_member_ids[result_usernames.index(most_common_label)]
#         return {"username": most_common_label, "distance": avg_distance, "member_id": most_common_member_id}
#     else:
#         print(f"Average distance ({avg_distance}) exceeds threshold ({threshold}). No suitable match.")
#         return {"username": None, "distance": avg_distance, "member_id": -1}
    

