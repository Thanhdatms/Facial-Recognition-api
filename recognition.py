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
from flask import Flask, Response
import paho.mqtt.client as mqtt
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

app = Flask(__name__)

# Configuration
url_embedding = "http://192.168.208.211:8081/getAllDataWithUsername"
esp32_cam_stream_url = "http://192.168.208.58:81/stream"
captured_folder = "image"
index_file = "hr_annoy_index.ann"
dimension = 128
threshold = 0.8
checkpoint_path = os.path.join('0.0000_checkpoint.pth')
last_saved_time = 0
labels = []
usernames = []
member_ids = []
is_processing = False
broker = "192.168.208.211"
port = 1883
topic = "esp32/data"
# db_path = 'face_recognition'

def get_db_connection(db_path='face_recognition.db'):
    """Create and return a database connection"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

# Initialize MQTT client with protocol version 5 (latest)
client = mqtt.Client(protocol=mqtt.MQTTv5)

# Create required directories
os.makedirs(captured_folder, exist_ok=True)

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
model = SiameseModel(embedding_size=dimension)

def load_checkpoint(model, checkpoint_path):
    """Load weights from checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=model.device)
        # Use 'model_state_dict' if it exists, otherwise use checkpoint directly
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # Remove unexpected keys like 'epoch', 'optimizer_state_dict', 'loss'
        state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

def fetch_embeddings_from_url(url):
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

# def build_annoy_index_from_url(dimension, index_file, url):
#     embeddings = fetch_embeddings_from_url(url)
#     index = AnnoyIndex(dimension, 'angular')
#     global labels, usernames, member_ids
#     labels = []
#     usernames = []
#     member_ids = []
#     for i, item in enumerate(embeddings):
#         embedding = item.get("embedding")
#         username = item.get("username")
#         member_id = item.get("member_id")
#         if embedding and username and member_id:
#             labels.append(i)
#             usernames.append(username)
#             member_ids.append(member_id)
#             index.add_item(i, embedding)
#     if labels:
#         index.build(50)
#         index.save(index_file)
#         print(f"Built and saved Annoy index to {index_file}.")
#     else:
#         print("No valid embeddings to build Annoy index.")
#     return labels, usernames, member_ids

# def build_annoy_index_from_db(dimension, index_file, db_path):
    
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("SELECT image_vector_process, account_id, member_id FROM tbl_register_face")
    
#     embeddings = []
#     labels = []
#     usernames = []
#     member_ids = []
#     index = AnnoyIndex(dimension, 'angular')
    
#     for i, row in enumerate(cursor.fetchall()):
#         embedding_blob, username, member_id = row
#         embedding = list(map(float, embedding_blob.split(','))) 
#         if embedding and username and member_id:
#             labels.append(i)
#             usernames.append(username)
#             member_ids.append(member_id)
#             index.add_item(i, embedding)
    
#     conn.close()
    
#     if labels:
#         index.build(50)
#         index.save(index_file)
#         print(f"Built and saved Annoy index to {index_file}.")
#     else:
#         print("No valid embeddings to build Annoy index.")
    
#     return labels, usernames, member_ids

def build_annoy_index_sample(dimension, index_file):
    sample_data = [
        {"embedding": [ 0.0650,  0.0825,  0.0149,  0.0949,  0.1470, -0.0238, -0.0624,  0.1270,
         -0.1354, -0.1120, -0.0754,  0.1426, -0.0095, -0.0597, -0.0445,  0.0180,
         -0.1284,  0.1015, -0.1072, -0.1799, -0.1064,  0.0546,  0.0280, -0.1509,
         -0.0395,  0.0357, -0.0261, -0.0868, -0.0881,  0.0526, -0.0091, -0.0896,
          0.0884, -0.0756, -0.0711,  0.0250,  0.0861,  0.0837,  0.0863, -0.1475,
          0.0401,  0.0221, -0.1014, -0.0897, -0.0524, -0.1373, -0.1743, -0.0436,
          0.0170,  0.0014, -0.0391,  0.0747,  0.2257, -0.0202,  0.0978, -0.0094,
         -0.0246,  0.0536,  0.1537, -0.0831,  0.1042,  0.0737,  0.0463,  0.0441,
          0.1680,  0.0965, -0.0134,  0.0048, -0.0515,  0.1577, -0.1668,  0.0821,
          0.0474,  0.0731, -0.1263, -0.0415, -0.0754,  0.0682, -0.0239,  0.0209,
         -0.0372, -0.1894,  0.0591, -0.1411, -0.1264,  0.0512,  0.0713, -0.1293,
          0.0067,  0.0366,  0.0646, -0.0385, -0.0780, -0.0292, -0.1287,  0.0189,
         -0.0183,  0.0917,  0.0784, -0.1084,  0.0142,  0.1496,  0.0748, -0.0422,
          0.0952,  0.1602, -0.0046,  0.0735,  0.0056,  0.0401, -0.0179, -0.0762,
          0.1354,  0.0009,  0.0308,  0.0649, -0.1170,  0.0441,  0.0597, -0.0828,
          0.0862,  0.0153, -0.1449, -0.0659,  0.0158,  0.0446, -0.1157, -0.0738], "username": "dat", "member_id": 1},
        {"embedding": [ 0.0635,  0.1409, -0.0246,  0.0246,  0.1526,  0.0519, -0.1107,  0.0972,
         -0.1791, -0.0296,  0.0006,  0.0843,  0.0444, -0.0702, -0.1145,  0.0781,
         -0.1140, -0.0329,  0.0345, -0.0925, -0.1268,  0.0942, -0.0754, -0.0848,
          0.0952, -0.1038, -0.1516, -0.0730,  0.0166,  0.0271, -0.0891, -0.1260,
         -0.0588,  0.0173, -0.0212,  0.1922,  0.0669,  0.0018,  0.0480, -0.0430,
          0.1092,  0.0453,  0.0269,  0.0141,  0.0906, -0.1036, -0.2199, -0.0537,
         -0.0755, -0.0573, -0.0185, -0.0174,  0.1926, -0.0377,  0.0935, -0.0083,
          0.0706, -0.0358,  0.0640, -0.0095,  0.0089,  0.1019,  0.0504,  0.0974,
          0.1507,  0.0048, -0.0930, -0.0382, -0.0719,  0.0865, -0.0734,  0.0126,
         -0.0507,  0.0409,  0.0041, -0.0322, -0.0440, -0.0140, -0.1414, -0.0287,
         -0.0460, -0.1151,  0.1428, -0.0960, -0.0507,  0.0780,  0.1886, -0.0963,
         -0.0219, -0.0123, -0.0053, -0.1439, -0.0559,  0.1153, -0.1123, -0.0392,
         -0.1712,  0.0182,  0.1318,  0.0094,  0.0940,  0.0912,  0.1871,  0.1001,
          0.0704,  0.1183, -0.0143,  0.0067, -0.0605, -0.0539,  0.0297, -0.0772,
          0.1209,  0.1284, -0.0582,  0.0470, -0.1592,  0.1305, -0.0528, -0.1033,
          0.0050,  0.0872, -0.1437, -0.0736, -0.0156, -0.0155, -0.0566, -0.0151], "username": "na", "member_id": 2},
    ]
    
    index = AnnoyIndex(dimension, 'angular')
    labels = []
    usernames = []
    member_ids = []
    
    for i, item in enumerate(sample_data):
        embedding = item["embedding"]
        username = item["username"]
        member_id = item["member_id"]
        
        labels.append(i)
        usernames.append(username)
        member_ids.append(member_id)
        index.add_item(i, embedding)
    
    if labels:
        index.build(50)
        index.save(index_file)
        print(f"Built and saved Annoy index to {index_file}.")
    else:
        print("No valid embeddings to build Annoy index.")
    
    return labels, usernames, member_ids


def check(embedding, threshold=0.5):
    embeddings = [
        {"embedding": [ 0.0650,  0.0825,  0.0149,  0.0949,  0.1470, -0.0238, -0.0624,  0.1270,
         -0.1354, -0.1120, -0.0754,  0.1426, -0.0095, -0.0597, -0.0445,  0.0180,
         -0.1284,  0.1015, -0.1072, -0.1799, -0.1064,  0.0546,  0.0280, -0.1509,
         -0.0395,  0.0357, -0.0261, -0.0868, -0.0881,  0.0526, -0.0091, -0.0896,
          0.0884, -0.0756, -0.0711,  0.0250,  0.0861,  0.0837,  0.0863, -0.1475,
          0.0401,  0.0221, -0.1014, -0.0897, -0.0524, -0.1373, -0.1743, -0.0436,
          0.0170,  0.0014, -0.0391,  0.0747,  0.2257, -0.0202,  0.0978, -0.0094,
         -0.0246,  0.0536,  0.1537, -0.0831,  0.1042,  0.0737,  0.0463,  0.0441,
          0.1680,  0.0965, -0.0134,  0.0048, -0.0515,  0.1577, -0.1668,  0.0821,
          0.0474,  0.0731, -0.1263, -0.0415, -0.0754,  0.0682, -0.0239,  0.0209,
         -0.0372, -0.1894,  0.0591, -0.1411, -0.1264,  0.0512,  0.0713, -0.1293,
          0.0067,  0.0366,  0.0646, -0.0385, -0.0780, -0.0292, -0.1287,  0.0189,
         -0.0183,  0.0917,  0.0784, -0.1084,  0.0142,  0.1496,  0.0748, -0.0422,
          0.0952,  0.1602, -0.0046,  0.0735,  0.0056,  0.0401, -0.0179, -0.0762,
          0.1354,  0.0009,  0.0308,  0.0649, -0.1170,  0.0441,  0.0597, -0.0828,
          0.0862,  0.0153, -0.1449, -0.0659,  0.0158,  0.0446, -0.1157, -0.0738], "username": "dat", "member_id": 1},
        {"embedding": [-0.0197,  0.1045,  0.0089,  0.0693,  0.0519, -0.0412,  0.0412,  0.1020,
         -0.0919, -0.0827, -0.0629,  0.1008,  0.0478, -0.1176, -0.0321, -0.0125,
         -0.0855,  0.1147,  0.0025, -0.0491, -0.0713,  0.1025, -0.1137, -0.1171,
          0.0359, -0.0683, -0.0749, -0.0499, -0.1016,  0.0447, -0.0956, -0.2016,
         -0.0592, -0.0230, -0.0249,  0.0295,  0.0918,  0.0171,  0.0255, -0.2139,
          0.0016,  0.0142, -0.1890, -0.1636,  0.0010, -0.1067, -0.1544,  0.0420,
         -0.0328, -0.1435, -0.0762,  0.0782,  0.1992, -0.1529,  0.1540, -0.0373,
          0.0283,  0.0315,  0.1483,  0.0027,  0.0139,  0.1302,  0.1272,  0.1216,
          0.1519,  0.0069, -0.1179, -0.0607, -0.0299,  0.0507, -0.0480,  0.0381,
         -0.0292,  0.0095,  0.0093, -0.1011, -0.0026,  0.0205, -0.1388,  0.0059,
         -0.1911, -0.0992,  0.0976, -0.1360,  0.0641,  0.0842,  0.1409, -0.0237,
         -0.0564,  0.0987,  0.0417, -0.1233, -0.0472,  0.0861,  0.0463,  0.0294,
         -0.0748, -0.0250,  0.0640, -0.0896,  0.1144,  0.1477,  0.1055,  0.0895,
         -0.0364,  0.0679,  0.1050, -0.0283,  0.0583,  0.0638, -0.0523,  0.0164,
          0.0601,  0.0551,  0.0240, -0.0429, -0.0627, -0.0033,  0.0597, -0.0257,
          0.0339,  0.1480, -0.1856, -0.0270, -0.0797, -0.0626, -0.0898, -0.0602], "username": "na", "member_id": 2},
    ]

    closest_match = None
    min_distance = float("inf")

    for e in embeddings:
        distance = np.linalg.norm(np.array(e["embedding"]) - np.array(embedding))

        if distance < threshold:
            if distance < min_distance:
                min_distance = distance
                closest_match = {
                    "username": e["username"],
                    "distance": distance,
                    "member_id": e["member_id"]
                }

    if closest_match:
        return closest_match
    else:
        return {"username": "unknown", "distance": "unknown", "member_id": "unknown"}



def search_in_annoy_index(query_embedding, index_file, usernames, member_ids, n=10, threshold=threshold):
    index = AnnoyIndex(dimension, 'angular')
    index.load(index_file)
    nearest_indices, distances = index.get_nns_by_vector(query_embedding, n=n, include_distances=True)

    if not nearest_indices or len(usernames) <= max(nearest_indices):
        print("Unable to find corresponding usernames in Annoy index.")
        return {"username": None, "distance": None, "member_id": -1}

    result_usernames = [usernames[i] for i in nearest_indices]
    result_member_ids = [member_ids[i] for i in nearest_indices]
    label_counts = Counter(result_usernames)
    most_common_label, count = label_counts.most_common(1)[0]
    avg_distance = sum(distances) / n
    print(avg_distance)
    if avg_distance <= threshold:
        most_common_member_id = result_member_ids[result_usernames.index(most_common_label)]
        return {"username": most_common_label, "distance": avg_distance, "member_id": most_common_member_id}
    else:
        print(f"Average distance ({avg_distance}) exceeds threshold ({threshold}). No suitable match.")
        return {"username": None, "distance": avg_distance, "member_id": -1}


def encode_image(file_path, model):
    try:
        img = Image.open(file_path).convert("RGB")
        img_tensor = inference_transform(img).unsqueeze(0).to(model.device)
        with torch.no_grad():
            embedding = model.embedding(img_tensor).cpu().numpy().flatten()
        return embedding
    except Exception as e:
        print(f"Error encoding {file_path}: {e}")
        return None

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def send_post_request(image_base64, member_id, timestamp):
    url = "http://192.168.208.211:8081/createhistories"
    data = {
        "base64Image": image_base64,
        "member_id": member_id,
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
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            print(f"Number of faces detected: {len(faces)}")
            for (x, y, w, h) in faces:
                x_start = max(x - padding, 0)
                y_start = max(y - padding, 0)
                x_end = min(x + w + padding, frame.shape[1])
                y_end = min(y + h + padding, frame.shape[0])
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

            current_time = time.time()
            if face_detected_time == 0:
                face_detected_time = current_time

            if current_time - face_detected_time >= 3 and current_time - last_saved_time >= 8:
                print(f"Saving image after {current_time - face_detected_time}s and {current_time - last_saved_time}s.")
                face_detected_time = 0
                last_saved_time = current_time

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(captured_folder, f"captured_{timestamp}.jpg")
                cv2.imwrite(image_path, frame)
                print(f"Saved image at: {image_path}")

                if not is_processing:
                    is_processing = True

                    def process_image():
                        global is_processing
                        image_base64 = None
                        try:
                            print("Processing image...")
                            query_embedding = encode_image(image_path, model)
                            if query_embedding is not None:
                                print("Encoded image successfully.")
                                result = check(query_embedding)
                                print(f"Search result: {result}")

                                image_base64 = image_to_base64(image_path)
                                if result["username"] != 'unknown':
                                    print(f"Sending post request for username {result['username']}.")
                                    member_id = result["member_id"]
                                    client.publish(topic, result["username"])
                                    # send_post_request(image_base64, member_id, timestamp)
                                else:
                                    print("No suitable match found within the threshold.")
                                    client.publish(topic, "unknown")
                                    member_id = -1
                                    # send_post_request(image_base64, member_id, timestamp)
                            else:
                                print("Unable to generate embedding for the image.")
                                client.publish(topic, "unknown")
                                member_id = -1
                                # send_post_request(image_base64, member_id, timestamp)
                        except Exception as e:
                            print(f"Error processing image: {e}")
                            client.publish(topic, "unknown")
                            member_id = -1
                            # send_post_request(image_base64, member_id, timestamp)
                        finally:
                            is_processing = False

                    threading.Thread(target=process_image).start()

        else:
            face_detected_time = 0

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Load model checkpoint
    load_checkpoint(model, checkpoint_path)
    
    # Initialize Annoy index
    print("Initializing Annoy index...")
    build_annoy_index_sample(dimension, index_file)
    print("Annoy index initialized.")
    
    # Connect MQTT and run Flask app
    client.connect(broker, port)
    app.run(host='0.0.0.0', port=5000)