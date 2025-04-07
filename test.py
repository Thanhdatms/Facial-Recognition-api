import cv2
import torch
import os
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

inference_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Example inference (đã sửa)
anchor_path = './uploads/8/face_8_a8ee84c4-b31f-4871-9e00-cc52b0852072.jpg'

positive_path = './uploads/8/face_8_a8ee84c4-b31f-4871-9e00-cc52b0852072.jpg'

# Define Embedding Model (đã sửa)
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

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define Siamese Model
class SiameseModel(nn.Module):
    def __init__(self, embedding_size=128):
        super(SiameseModel, self).__init__()
        self.embedding = EmbeddingModel(embedding_size=embedding_size)

    def forward(self, anchor, positive, negative):
        anchor_emb = self.embedding(anchor)
        positive_emb = self.embedding(positive)
        negative_emb = self.embedding(negative)
        return anchor_emb, positive_emb, negative_emb
    
def get_embedding(img_path, model, transform):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.embedding(img)
    return embedding

# Load trained model
checkpoint = torch.load(os.path.join('0.0012_checkpoint.pth'), map_location=torch.device('cpu')) 
model = SiameseModel(embedding_size=128).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get embeddings
anchor_embedding = get_embedding(anchor_path, model, inference_transform) 
positive_embedding = get_embedding(positive_path, model, inference_transform)

print(anchor_embedding)
print(positive_embedding)
# Calculate distance
distance = torch.norm(anchor_embedding - positive_embedding, p=2)
print(distance)
print(f"Distance between anchor and positive: {distance.item()}")

threshold = 0.5
print(f"Prediction: {'Similar' if distance.item() < threshold else 'Dissimilar'}")