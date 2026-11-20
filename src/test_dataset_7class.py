import torch
from src.dataset_resnet_7class import create_resnet7_dataloaders

train_loader, val_loader, test_loader = create_resnet7_dataloaders(
    batch_size=2
)

videos, labels = next(iter(train_loader))

print("\n===== 7-CLASS DATASET VERIFICATION =====")
print("Videos shape:", videos.shape)
print("Labels shape:", labels.shape)
print("Video dtype:", videos.dtype)
print("Video min:", videos.min().item())
print("Video max:", videos.max().item())
print("Labels:", labels.tolist())
print("Number of classes:", len(set(
    int(row["label"])
    for row in train_loader.dataset.samples
)))
