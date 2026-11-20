import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CNNTransformerResNetImproved(nn.Module):

    def __init__(
        self,
        num_classes=7,
        num_frames=16,
        d_model=512,
        num_heads=8,
        num_layers=1,
        dropout=0.4,
        freeze_backbone=True
    ):
        super().__init__()

        backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        self.cnn = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        if freeze_backbone:
            for param in self.cnn.parameters():
                param.requires_grad = False

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_frames, d_model)
        )

        nn.init.trunc_normal_(
            self.pos_embedding,
            std=0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            d_model,
            num_classes
        )

    def forward(self, x):

        batch_size, num_frames, channels, height, width = x.shape

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

        features = features + self.pos_embedding[:, :num_frames, :]

        features = self.transformer(features)

        features = features.mean(dim=1)

        features = self.norm(features)

        output = self.fc(
            self.dropout(features)
        )

        return output
