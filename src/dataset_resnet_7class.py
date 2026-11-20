import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class UCF101ResNet7Dataset(Dataset):

    def __init__(self, split, root_dir="."):
        self.root_dir = Path(root_dir)

        csv_path = (
            self.root_dir
            / "data"
            / "splits_7class"
            / f"{split}.csv"
        )

        self.frames_dir = (
            self.root_dir
            / "data"
            / "processed"
            / "frames"
            / split
        )

        self.split = split

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            self.samples = list(csv.DictReader(f))

        self.mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1)

        self.std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)

        print(
            f"{split.upper()} 7-class dataset loaded: "
            f"{len(self.samples)} videos"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        class_name = sample["class_name"]
        video_name = Path(sample["filepath"]).stem

        video_path = (
            self.frames_dir
            / class_name
            / f"{video_name}.npy"
        )

        video = np.load(video_path)

        video = torch.from_numpy(video.copy()).float() / 255.0

        # (T, H, W, C) -> (T, C, H, W)
        video = video.permute(0, 3, 1, 2)

        # Train-only augmentation
        if self.split == "train":

            if random.random() < 0.5:
                video = torch.flip(video, dims=[3])

            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)

            video = video * brightness

            mean = video.mean(
                dim=(0, 2, 3),
                keepdim=True
            )

            video = (video - mean) * contrast + mean
            video = torch.clamp(video, 0.0, 1.0)

        # ImageNet normalization
        video = (video - self.mean) / self.std

        label = int(sample["label"])
        label = torch.tensor(label, dtype=torch.long)

        return video, label


def create_resnet7_dataloaders(batch_size=4):

    train_dataset = UCF101ResNet7Dataset("train")
    val_dataset = UCF101ResNet7Dataset("val")
    test_dataset = UCF101ResNet7Dataset("test")

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
