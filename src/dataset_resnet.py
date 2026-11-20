
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class UCF101ResNetDataset(Dataset):

    def __init__(self, split, root_dir="."):
        self.root_dir = Path(root_dir)
        self.split = split

        csv_path = (
            self.root_dir
            / "data"
            / "splits"
            / f"{split}.csv"
        )

        with open(csv_path, newline="", encoding="utf-8") as f:
            self.samples = list(csv.DictReader(f))

        self.frames_dir = (
            self.root_dir
            / "data"
            / "processed"
            / "frames"
            / split
        )

        self.mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1)

        self.std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)

        print(
            f"{split.upper()} ResNet dataset loaded: "
            f"{len(self.samples)} videos"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        video_name = Path(sample["filepath"]).stem
        class_name = sample["class_name"]

        video_path = (
            self.frames_dir
            / class_name
            / f"{video_name}.npy"
        )

        video = np.load(video_path)

        # Convert to [0, 1]
        video = video.astype(np.float32) / 255.0

        # (T, H, W, C) -> (T, C, H, W)
        video = np.transpose(video, (0, 3, 1, 2)).copy()

        video_tensor = torch.from_numpy(video)

        # Data augmentation: training only
        if self.split == "train":

            # Horizontal flip: same for all frames
            if torch.rand(1).item() < 0.5:
                video_tensor = torch.flip(
                    video_tensor,
                    dims=[-1]
                )

            # Brightness augmentation
            brightness = torch.empty(1).uniform_(0.9, 1.1).item()
            video_tensor = video_tensor * brightness

            # Contrast augmentation
            contrast = torch.empty(1).uniform_(0.9, 1.1).item()

            video_mean = video_tensor.mean(
                dim=(0, 2, 3),
                keepdim=True
            )

            video_tensor = (
                (video_tensor - video_mean) * contrast
                + video_mean
            )

            video_tensor = torch.clamp(
                video_tensor,
                0.0,
                1.0
            )

        # ImageNet normalization
        video_tensor = (
            video_tensor - self.mean
        ) / self.std

        label = int(sample["label"])
        label_tensor = torch.tensor(
            label,
            dtype=torch.long
        )

        return video_tensor, label_tensor


def create_resnet_dataloaders(batch_size=4):

    train_dataset = UCF101ResNetDataset("train")
    val_dataset = UCF101ResNetDataset("val")
    test_dataset = UCF101ResNetDataset("test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader
