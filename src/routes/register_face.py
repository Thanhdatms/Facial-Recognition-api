import os
import sqlite3
import base64
import uuid
import cv2
import torch
from flask import Flask, request, jsonify, Blueprint
import numpy as np
from src.models.siamese_model import SiameseModel
from src.config.config import captured_folder, dimension, threshold, checkpoint_path, face_crop_path, na_crop_face
from src.utils.image_processing import encode_image
import json
import time

register_bp = Blueprint("register", __name__)
DB_PATH = "face_recognition.db"

# Load OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


@register_bp.route("/register_face", methods=["POST"])
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
        upload_dir = os.path.join("uploads", str(account_id))
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
            cropped_face_filename = f"face_{account_id}_{uuid.uuid4()}.jpg"
            cropped_path = os.path.join(upload_dir, cropped_face_filename)
            
            # Save the cropped face
            cv2.imwrite(cropped_path, face_crop)
            saved_faces.append(cropped_path)

            # Embed the cropped face
            model = SiameseModel(embedding_size=dimension)
            embedding = encode_image(cropped_path, model)
            embedding_json = json.dumps(embedding.tolist())

            # Insert into database
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO tbl_register_faces (face_image_process, account_id, image_vector_process) VALUES (?, ?, ?)",
                    (cropped_path, account_id, embedding_json),
                )
                conn.commit()

        return jsonify({"message": "Face registered successfully", "saved_faces": saved_faces})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
