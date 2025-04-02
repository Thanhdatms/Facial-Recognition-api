import torch
import torch.nn as nn

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