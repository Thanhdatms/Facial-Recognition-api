import cv2
import os
import json
import sqlite3
import requests
from datetime import datetime
import time
import base64
from collections import Counter
from annoy import AnnoyIndex
import threading
from flask import Flask, Response, request, jsonify, Blueprint
import paho.mqtt.client as mqtt
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import uuid

app = Flask(__name__)

# Configuration
esp32_cam_stream_url = "http://172.20.10.10:81/stream"
captured_folder = "captured_folder"
index_file = "./hr_annoy_index.ann"
dimension = 128
threshold = 0.8
# checkpoint_path = '0.0012_checkpoint.pth'
last_saved_time = 0
labels = []
account_ids = []
is_processing = False
broker = "172.20.10.6"
port = 1883
topic = "esp32/data"
#db_path = '/home/pi/deployment/config/database.db'
db_path = './face_recognition.db'

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def get_db_connection(db_path):
    """Create and return a database connection"""
    conn = sqlite3.connect(db_path)
   
    return conn

# Initialize MQTT client with protocol version 5 (latest)
client = mqtt.Client(protocol=mqtt.MQTTv5)

# Create required directories
os.makedirs(captured_folder, exist_ok=True)

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def build_annoy_index_from_db(dimension, index_file, db_path):
    conn = get_db_connection(db_path)
    if conn is None:
        return [], []

    cursor = conn.cursor()
    try:
        cursor.execute("SELECT image_vector_process, account_id FROM tbl_register_faces")
        rows = cursor.fetchall()
        print(rows)
    except sqlite3.Error as e:
        print(f"SQL Error: {e}")
        conn.close()
        return [], []

    labels = []
    account_ids = []
    index = AnnoyIndex(dimension, 'angular')

    for i, row in enumerate(rows):
        embedding_blob, account_id = row
        try:
            clean_str = embedding_blob.replace('[', '').replace(']', '').replace('\n', '').strip()
            embedding = list(map(float, clean_str.split(',')))
        except Exception as e:
            print(f"Error parsing embedding on row {i}: {e}")
            continue

        labels.append(i)
        account_ids.append(account_id)
        index.add_item(i, embedding)
    conn.close()

    if labels:
        index.build(50)
        index.save(index_file)
        print("Create successfully")
    else:
        print("No valid embeddings to build Annoy index.")

    return labels, account_ids

def search_in_annoy_index(query_embedding, account_ids, threshold=0.4):

    labels, account_ids = build_annoy_index_from_db(dimension, index_file, db_path)
    index = AnnoyIndex(dimension, 'angular')
    index.load(index_file)
    nearest_indices, distances = index.get_nns_by_vector(query_embedding, n=1, include_distances=True)

    if not nearest_indices or nearest_indices[0] >= len(account_ids):
        return {"account_id": "unknown", "distance": "unknown"}

    nearest_index = nearest_indices[0]
    nearest_distance = distances[0]
    print(nearest_distance)
    nearest_account_id = account_ids[nearest_index]
    print(nearest_account_id)
    if nearest_distance <= threshold:
        return {
            "account_id": nearest_account_id,
            "distance": nearest_distance,
        }
    else:
        return {"account_id": "unknown", "distance": "unknown"}
    
# Define the custom EmbeddingModel
class EmbeddingModel(nn.Module):
    def __init__(self, embedding_size=128):
        super(EmbeddingModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=10, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4, padding=1)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, padding=1)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(256 * 9 * 9, embedding_size)
        self.normalize = nn.functional.normalize

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = torch.relu(self.conv3(x))
        x = self.maxpool3(x)
        x = torch.relu(self.conv4(x))
        x = self.flatten(x)
        x = self.dense(x)
        x = self.normalize(x, p=2, dim=1)
        return x

class SiameseModel(nn.Module):
    def __init__(self, embedding_size=128):
        super(SiameseModel, self).__init__()
        self.embedding = EmbeddingModel(embedding_size=embedding_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        return self.embedding(x)

# Define transform for inference
inference_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Initialize the model globally
checkpoint = torch.load(os.path.join('0.0012_checkpoint.pth'), map_location=torch.device('cpu')) 
model = SiameseModel(embedding_size=128).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def get_embedding(img_path, model, transform):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)  
    with torch.no_grad():
        embedding = model.embedding(img)
        embedding = embedding[0]
    return embedding

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def send_post_request(image_base64, account_id, timestamp):
    url = "http://172.20.10.6:8081/createhistories"
    data = {
        "base64Image": image_base64,
        "account_id": account_id,
        "timestamp": timestamp
    }
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"Successfully sent data to the server. Response: {response.text}")
        else:
            print(f"Failed to send data. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error sending POST request: {e}")

def generate_frames():
    global last_saved_time, is_processing
    cap = cv2.VideoCapture(esp32_cam_stream_url)
    if not cap.isOpened():
        print("Unable to connect to ESP32-CAM.")
        return

    face_detected_time = 0
    padding = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        print("Processing frame...")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        display_frame = frame.copy()

        if len(faces) > 0:
            print(f"Number of faces detected: {len(faces)}")
            (x, y, w, h) = faces[0]

            # Apply padding to height
            y_pad = max(0, y )
            h_pad = min(frame.shape[0] - y_pad, h)

            # Draw rectangle on detected face
            cv2.rectangle(display_frame, (x, y_pad), 
                        (x + w, y_pad + h_pad), (0, 255, 0), 2)

            # Crop face region
            face_frame = frame[y_pad:y_pad + h_pad, x:x + w]

            current_time = time.time()
            if face_detected_time == 0:
                face_detected_time = current_time

            if current_time - face_detected_time >= 2 and current_time - last_saved_time >= 4:
                print(f"Saving image after {current_time - face_detected_time}s and {current_time - last_saved_time}s.")
                face_detected_time = 0
                last_saved_time = current_time

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(captured_folder, f"captured_{timestamp}.jpg")
                cv2.imwrite(image_path, face_frame)
                print(f"Saved image at: {image_path}")

                if not is_processing:
                    is_processing = True

                    def process_image():
                        global is_processing
                        image_base64 = None
                        try:
                            print("Processing image...")
                            query_embedding = get_embedding(image_path, model, inference_transform)
                            print(query_embedding)
                            if query_embedding is not None:
                                print("Encoded image successfully.")
                                result = search_in_annoy_index(query_embedding, account_ids)
                                print(f"Search result: {result}")

                                image_base64 = image_to_base64(image_path)
                                if result["account_id"] != 'unknown':
                                    print(f"Sending post request for account_id {result['account_id']}.")
                                    client.publish(topic, result["account_id"])
                                    send_post_request(image_base64, result["account_id"], timestamp)
                            else:
                                print("Unable to generate embedding for the image.")
                                client.publish(topic, "unknown")
                                send_post_request(image_base64, 'unknow', timestamp)
                        except Exception as e:
                            print(f"Error processing image: {e}")
                            client.publish(topic, "unknown")
                            # send_post_request(image_base64, member_id, timestamp)
                        finally:
                            is_processing = False

                    threading.Thread(target=process_image).start()

        else:
            face_detected_time = 0

        _, buffer = cv2.imencode('.jpg', display_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/register_face", methods=["POST"])
def register_face():
    data = request.json
    account_id = data.get("account_id")
    base64_image = data.get("base64Image")

    if not account_id or not base64_image:
        return jsonify({"error": "Missing account_id or base64Image"}), 400

    try:
        # Decode base64 image to OpenCV format
        image_data = np.frombuffer(base64.b64decode(base64_image), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return jsonify({"error": "No face detected"}), 400

        # Create directory if it doesn't exist
        upload_dir = os.path.join("process", str(account_id))
        os.makedirs(upload_dir, exist_ok=True)

        # Process each detected face
        padding = 30
        saved_faces = []

        for i, (x, y, w, h) in enumerate(faces):
            # Calculate coordinates with padding
            x_start = max(x - padding, 0)
            y_start = max(y - padding, 0)
            x_end = min(x + w + padding, image.shape[1])
            y_end = min(y + h + padding, image.shape[0])

            # Crop the face
            face_crop = image[y_start:y_end, x_start:x_end]

            # Generate a unique filename for the cropped face
            id = uuid.uuid4()
            cropped_face_filename = f"face_{account_id}_{id}.jpg"
            cropped_path = os.path.join(upload_dir, cropped_face_filename)
            # Save the cropped face
            cv2.imwrite(cropped_path, face_crop)
            saved_faces.append(cropped_path)

            # Embed the cropped face
            embedding = get_embedding(cropped_path, model, inference_transform)
            embedding_json = json.dumps(embedding.tolist())
            # Insert into database
            # with sqlite3.connect(db_path) as conn:
            #     cursor = conn.cursor()
            #     face_image_process = f"http://localhost:8888/process/{account_id}/processed_{id}.jpg"
            #     cursor.execute(
            #         "INSERT INTO tbl_register_faces (face_image_process, account_id, image_vector_process) VALUES (?, ?, ?)",
            #         (face_image_process, account_id, embedding_json),
            #     )
            #     conn.commit()
        processPath = cropped_path
        return jsonify({
            "embedding": embedding_json,
            "processPath": processPath})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load model checkpoint
    # load_checkpoint(model, checkpoint_path)
    
    # Initialize Annoy index
    print("Initializing Annoy index...")
    labels, account_ids = build_annoy_index_from_db(dimension, index_file, db_path)
    print("Annoy index initialized.")
    # Connect MQTT and run Flask app
    client.connect(broker, port)
    app.run(host='0.0.0.0', port=5000)