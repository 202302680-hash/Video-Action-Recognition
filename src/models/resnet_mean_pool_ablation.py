
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNetMeanPoolAblation(nn.Module):

    def __init__(
        self,
        num_classes=7,
        freeze_backbone=True,
        dropout=0.4
    ):
        super().__init__()

        # Pretrained ResNet18 feature extractor
        backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        self.cnn = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        if freeze_backbone:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # Temporal mean pooling replaces the Transformer
        self.norm = nn.LayerNorm(512)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):

        # Input: (Batch, Frames, Channels, Height, Width)
        batch_size, num_frames, channels, height, width = x.shape

        # Extract spatial features from each frame
        x = x.reshape(
            batch_size * num_frames,
            channels,
            height,
            width
        )

        features = self.cnn(x)

        features = features.reshape(
            batch_size,
            num_frames,
            512
        )

        # Aggregate frame features without temporal attention
        features = features.mean(dim=1)

        features = self.norm(features)

        output = self.fc(
            self.dropout(features)
        )

        return output
