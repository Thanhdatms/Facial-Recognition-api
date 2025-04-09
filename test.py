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

# Example inference (đã sửa)
anchor_path = './dat.jpeg'

positive_path = './uploads/8/face_8_a8ee84c4-b31f-4871-9e00-cc52b0852072.jpg'

# Get embeddings
anchor_embedding = get_embedding(anchor_path, model, inference_transform) 
positive_embedding = [[ 0.0008, -0.1386,  0.0895,  0.0419, -0.0096, -0.0967, -0.0259,  0.1035,
        -0.1159, -0.1529,  0.1020,  0.1082,  0.0265,  0.0361,  0.1379,  0.1134,
         0.1104,  0.1074, -0.1356,  0.1449, -0.0006,  0.1072,  0.0285, -0.1147,
         0.1183, -0.0910, -0.1152, -0.0355,  0.1338,  0.0311,  0.1009,  0.0215,
        -0.0273,  0.0804, -0.0049,  0.0683, -0.1218,  0.0317, -0.0354, -0.0569,
         0.1285,  0.0739, -0.0327,  0.0777, -0.1245, -0.1002,  0.0614,  0.1006,
        -0.0437,  0.0879,  0.0900, -0.1064, -0.0686,  0.1301, -0.1218,  0.0306,
         0.0760,  0.1470,  0.0943, -0.0513,  0.1275, -0.1137,  0.0096,  0.0610,
        -0.0876,  0.0907, -0.0161, -0.0557, -0.1341, -0.1399, -0.0833, -0.1086,
         0.0540, -0.0267, -0.1224, -0.0799, -0.0381, -0.0843,  0.1020,  0.0763,
         0.0988, -0.0752, -0.1324,  0.0005,  0.0034, -0.0770,  0.0889,  0.1381,
        -0.0903, -0.1004, -0.0124, -0.1092, -0.0824,  0.0739, -0.0849,  0.0986,
        -0.0799,  0.0615,  0.0638,  0.0493, -0.0212, -0.0513,  0.0131, -0.0992,
         0.0963,  0.0820, -0.0025, -0.1167, -0.0335, -0.0610,  0.1421,  0.0636,
         0.1118,  0.0640,  0.1309,  0.0071,  0.0722, -0.1026,  0.0840,  0.0501,
         0.0914,  0.1146, -0.1061, -0.0006,  0.0536, -0.0198, -0.1220, -0.1198]]

positive_embedding_tensor = torch.tensor(positive_embedding, dtype=torch.float32)

# Now both are tensors, so you can subtract them
distance = torch.norm(anchor_embedding - positive_embedding_tensor, p=2)
print(f"Distance between anchor and positive: {distance.item()}")

threshold = 0.5
print(f"Prediction: {'Similar' if distance.item() < threshold else 'Dissimilar'}")