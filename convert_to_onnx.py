import torch
import torch.nn as nn
import torch.onnx
import os

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

    def forward(self, x):
        return self.embedding(x)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "./src/training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = torch.load(os.path.join(checkpoint_prefix, "0.0000_checkpoint.pth"), map_location=device)

model = SiameseModel(embedding_size=128).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Convert to ONNX
onnx_path = "siamese_model.onnx"
dummy_input = torch.randn(1, 3, 100, 100).to(device)

torch.onnx.export(model, dummy_input, onnx_path,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

print(f"Model has been convert to ONNX: {onnx_path}")
