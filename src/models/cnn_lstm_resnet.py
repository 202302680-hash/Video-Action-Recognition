
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CNNLSTMResNet(nn.Module):
    def __init__(self, num_classes=10, hidden_size=128, freeze_backbone=True):
        super().__init__()

        # Pretrained ResNet18 for spatial feature extraction
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.cnn = nn.Sequential(*list(backbone.children())[:-1])

        # Freeze pretrained CNN initially to reduce training cost
        if freeze_backbone:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # LSTM for temporal feature learning
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Input: (Batch, Frames, Channels, Height, Width)
        batch_size, num_frames, channels, height, width = x.shape

        # Extract features from all frames
        x = x.reshape(
            batch_size * num_frames,
            channels,
            height,
            width
        )

        features = self.cnn(x)
        features = features.reshape(batch_size, num_frames, 512)

        # Learn temporal relationships
        lstm_output, (hidden, cell) = self.lstm(features)

        last_hidden = hidden[-1]
        output = self.fc(self.dropout(last_hidden))

        return output
