# import cv2

# # URL of the ESP32-CAM stream
# esp32_cam_stream_url = "http://192.168.242.58:81/stream"

# # Open video stream
# cap = cv2.VideoCapture(esp32_cam_stream_url)

# if not cap.isOpened():
#     print("Error: Could not open video stream.")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame.")
#         break
    
#     # Display the frame
#     cv2.imshow("ESP32-CAM Stream", frame)
    
#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# import torch
# import torch.nn as nn
# import torch.onnx
# import os

# # Định nghĩa lại mô hình
# class EmbeddingModel(nn.Module):
#     def __init__(self, embedding_size=128):
#         super(EmbeddingModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=10, padding=1)
#         self.maxpool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=7, padding=1)
#         self.maxpool2 = nn.MaxPool2d(2, 2)
#         self.conv3 = nn.Conv2d(128, 128, kernel_size=4, padding=1)
#         self.maxpool3 = nn.MaxPool2d(2, 2)
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=4, padding=1)
#         self.flatten = nn.Flatten()
#         self.dense = nn.Linear(256 * 9 * 9, embedding_size)
#         self.normalize = nn.functional.normalize

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = self.maxpool1(x)
#         x = torch.relu(self.conv2(x))
#         x = self.maxpool2(x)
#         x = torch.relu(self.conv3(x))
#         x = self.maxpool3(x)
#         x = torch.relu(self.conv4(x))
#         x = self.flatten(x)
#         x = self.dense(x)
#         x = self.normalize(x, p=2, dim=1)
#         return x

# class SiameseModel(nn.Module):
#     def __init__(self, embedding_size=128):
#         super(SiameseModel, self).__init__()
#         self.embedding = EmbeddingModel(embedding_size=embedding_size)

#     def forward(self, x):
#         return self.embedding(x)

# # Load mô hình đã train
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# checkpoint_dir = "./src/training_checkpoints"
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = torch.load(os.path.join(checkpoint_prefix, "0.0000_checkpoint.pth"), map_location=device)

# model = SiameseModel(embedding_size=128).to(device)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# # Chuyển đổi sang ONNX
# onnx_path = "siamese_model.onnx"
# dummy_input = torch.randn(1, 3, 100, 100).to(device)  # Input mẫu

# torch.onnx.export(model, dummy_input, onnx_path,
#                   input_names=["input"], output_names=["output"],
#                   dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

# print(f"✅ Model đã được chuyển đổi thành ONNX: {onnx_path}")


# import onnxruntime as ort
# import numpy as np
# from PIL import Image
# import torchvision.transforms as transforms
# import time

# # Load ONNX model
# onnx_model = "siamese_model.onnx"
# session = ort.InferenceSession(onnx_model)

# # Define image preprocessing
# transform = transforms.Compose([
#     transforms.Resize((100, 100)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# def preprocess_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     image = transform(image).unsqueeze(0).numpy()
#     return image

# # Test with an image
# image_path = "face_20250331_153851_1.jpg"
# start = time.time()
# input_image = preprocess_image(image_path)

# # Run inference
# input_name = session.get_inputs()[0].name
# output_name = session.get_outputs()[0].name
# output = session.run([output_name], {input_name: input_image})[0]
# print(time.time()-start)
# print("ONNX Model Output:", output)

import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Example usage
image_path = './src/captured_folder/captured_20250401_162501.jpg'
base64_string = image_to_base64(image_path)
print('-----',base64_string, '---')
