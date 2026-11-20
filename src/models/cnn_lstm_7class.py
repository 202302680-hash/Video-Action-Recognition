import torch
import torch.nn as nn


class CNNLSTM7(nn.Module):

    def __init__(self, num_classes=7, hidden_size=128):
        super(CNNLSTM7, self).__init__()

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

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        batch_size, num_frames, channels, height, width = x.size()

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
            128
        )

        lstm_output, (hidden, cell) = self.lstm(features)

        last_hidden = hidden[-1]

        output = self.fc(self.dropout(last_hidden))

        return output
