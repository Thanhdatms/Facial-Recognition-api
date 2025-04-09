import cv2
import os
import uuid
import time

# Initialize webcam stream from ESP32
esp32_cam_stream_url = "http://172.20.10.10:81/stream"
DAT_PATH = "ma"
os.makedirs(DAT_PATH, exist_ok=True)

cap = cv2.VideoCapture(esp32_cam_stream_url)

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Padding for face crop (optional)
height_padding = 0  # Increase box height (top and bottom)

# Timer to delay saves
last_save_time = 0

while cap.isOpened(): 
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Copy frame for display
    display_frame = frame.copy()

    if len(faces) > 0:
        print("Face detected")
        (x, y, w, h) = faces[0]

        # Apply padding to height
        y_pad = max(0, y - height_padding)
        h_pad = min(frame.shape[0] - y_pad, h + 2 * height_padding)

        # Draw rectangle on detected face
        cv2.rectangle(display_frame, (x, y_pad), 
                      (x + w, y_pad + h_pad), (0, 255, 0), 2)

        # Crop face region
        face_frame = frame[y_pad:y_pad + h_pad, x:x + w]

        # Delay-controlled save
        current_time = time.time()

        imgname = os.path.join(DAT_PATH, '{}.jpg'.format(uuid.uuid1()))
        if cv2.imwrite(imgname, face_frame):
            print(f"Saved image to {imgname}")
            last_save_time = current_time
        else:
            print("Failed to save image")
    else:
        print("No face detected")
        face_frame = frame
        cv2.putText(display_frame, "No face detected", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show live feed
    cv2.imshow('Image Collection', display_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and cleanup
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)