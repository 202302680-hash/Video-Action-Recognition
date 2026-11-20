
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class UCF101Dataset(Dataset):

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

        print(
            f"{split.upper()} dataset loaded: "
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

        # Normalize pixel values to [0, 1]
        video = video.astype(np.float32) / 255.0

        # Convert (T, H, W, C) to (T, C, H, W)
        video = np.transpose(video, (0, 3, 1, 2))

        video_tensor = torch.from_numpy(video)

        label = int(sample["label"])
        label_tensor = torch.tensor(
            label,
            dtype=torch.long
        )

        return video_tensor, label_tensor


def create_dataloaders(batch_size=8):

    train_dataset = UCF101Dataset("train")
    val_dataset = UCF101Dataset("val")
    test_dataset = UCF101Dataset("test")

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
