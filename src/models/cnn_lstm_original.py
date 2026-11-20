
import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=10, hidden_size=128):
        super(CNNLSTM, self).__init__()

        # CNN: Extract spatial features from each frame
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # LSTM: Learn temporal information between frames
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Classifier: Predict the action
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Input shape: (Batch, Frames, Channels, Height, Width)
        batch_size, num_frames, channels, height, width = x.size()

        # Process all frames using CNN
        x = x.view(batch_size * num_frames, channels, height, width)
        features = self.cnn(x)

        # Reshape features for LSTM
        features = features.view(batch_size, num_frames, 128)

        # Process temporal sequence
        lstm_output, (hidden, cell) = self.lstm(features)

        # Use the last hidden state
        last_hidden = hidden[-1]

        # Predict class
        output = self.fc(self.dropout(last_hidden))

        return output
