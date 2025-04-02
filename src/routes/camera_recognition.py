import cv2
import os
import time
from flask import Flask, Response, Blueprint, jsonify, request
import threading
import torch
import numpy as np
import paho.mqtt.client as mqtt
from src.models.siamese_model import SiameseModel
from src.utils.image_processing import encode_image, inference_transform
from src.utils.comparison import compare_embedding
from src.config.config import captured_folder, dimension, threshold, checkpoint_path, face_crop_path, na_crop_face

# Flask Blueprint
face_recognition_bp = Blueprint('face_recognition', __name__)
esp32_cam_stream_url = "http://172.16.1.21:81/stream"
# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Configuration
last_saved_time = 0
is_processing = False

# Specific embedding for comparison
specific_embedding = np.array([-0.0847, -0.1159,  0.1373,  0.0595,  0.0765,  0.0843, -0.0138, -0.0733,
         -0.1065,  0.0915,  0.1379, -0.0970,  0.0155, -0.0930, -0.0849,  0.1385,
          0.1293,  0.0445,  0.0074, -0.1225,  0.0110, -0.0097,  0.0372, -0.0788,
         -0.0240, -0.0912,  0.1708, -0.0880,  0.0156,  0.0755,  0.1000, -0.0429,
         -0.1107, -0.0996,  0.0634,  0.0810, -0.0262,  0.0645,  0.0645, -0.0809,
          0.1489,  0.0846,  0.0358, -0.0443,  0.0289, -0.0817,  0.0377,  0.0154,
         -0.0307,  0.1686, -0.0694, -0.1233, -0.1603,  0.1476,  0.0201, -0.0540,
         -0.0309,  0.0375, -0.0859,  0.0932, -0.0881, -0.0901,  0.0534, -0.0357,
          0.1024,  0.0858, -0.1080,  0.0727, -0.0740, -0.0188, -0.1231, -0.0664,
          0.1133,  0.1192,  0.0705,  0.0287, -0.0979, -0.1021,  0.1150, -0.0445,
          0.0128,  0.0200, -0.1031, -0.0998, -0.0456, -0.0635, -0.1173,  0.0441,
          0.0775,  0.0414,  0.0773, -0.1147,  0.0122,  0.1039, -0.0757,  0.0244,
         -0.0888,  0.0644,  0.1632,  0.1187,  0.1445,  0.0476,  0.0736, -0.1757,
         -0.0945, -0.0841,  0.0900,  0.0066, -0.0066, -0.0353, -0.0987,  0.1559,
          0.0963, -0.0086,  0.0634, -0.0631, -0.0850,  0.1175, -0.0126,  0.0747,
          0.1497,  0.1593, -0.1054,  0.1016, -0.0598,  0.0150, -0.0923,  0.0928], 
    dtype=np.float32
)

# Load model
model = SiameseModel(embedding_size=dimension)
checkpoint = torch.load(checkpoint_path, map_location=model.device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def crop_and_embed_faces(image_path, model, output_folder=None, padding=30):

    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image from {image_path}")
            return []

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )

        results = []
        
        if len(faces) == 0:
            print("No faces detected in the image")
            return results

        print(f"Detected {len(faces)} faces in the image")

        # Create output folder if specified and doesn't exist
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Process each detected face
        for i, (x, y, w, h) in enumerate(faces):
            # Calculate coordinates with padding
            x_start = max(x - padding, 0)
            y_start = max(y - padding, 0)
            x_end = min(x + w + padding, image.shape[1])
            y_end = min(y + h + padding, image.shape[0])

            # Crop the face
            face_crop = image[y_start:y_end, x_start:x_end]

            # Generate timestamp for unique filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save cropped face if output_folder is specified
            cropped_path = None
            if output_folder:
                cropped_path = os.path.join(
                    output_folder, 
                    f"face_{timestamp}_{i}.jpg"
                )
                cv2.imwrite(cropped_path, face_crop)
                print(f"Saved cropped face at: {cropped_path}")

            # Generate embedding
            try:
                embedding = encode_image(cropped_path if cropped_path else face_crop, model)
                if embedding is not None:
                    results.append((cropped_path, embedding))
                    print(f"Generated embedding for face {i}")
                else:
                    print(f"Failed to generate embedding for face {i}")
            except Exception as e:
                print(f"Error generating embedding for face {i}: {e}")

        return results

    except Exception as e:
        print(f"Error in crop_and_embed_faces: {e}")
        return []

def process_image(image_path):
    global is_processing
    try:
        print("Processing image...")
        face_results = crop_and_embed_faces(
            image_path=image_path,
            model=model,
            output_folder=na_crop_face
        )
        
        for cropped_path, query_embedding in face_results:
            result = compare_embedding(query_embedding, specific_embedding, threshold)
            print(f"Comparison result for {cropped_path}: {result}")
            
    except Exception as e:
        print(f"Error processing image: {e}")
    finally:
        is_processing = False

def generate_frames():
    global last_saved_time
    global is_processing

    cap = cv2.VideoCapture(esp32_cam_stream_url)
    if not cap.isOpened():
        print("Unable to connect to camera.")
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

            if current_time - face_detected_time >= 0 and current_time - last_saved_time >= 0:
                print(f"Saving image after {current_time - face_detected_time}s and {current_time - last_saved_time}s.")
                face_detected_time = 0
                last_saved_time = current_time

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(captured_folder, f"captured_{timestamp}.jpg")
                cv2.imwrite(image_path, frame)
                print(f"Saved image at: {image_path}")

                if not is_processing:
                    is_processing = True
                    threading.Thread(target=process_image, args=(image_path,)).start()

        else:
            face_detected_time = 0

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@face_recognition_bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
