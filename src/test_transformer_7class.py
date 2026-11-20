import torch

from src.models.cnn_transformer_resnet import CNNTransformerResNet

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CNNTransformerResNet(
    num_classes=7,
    num_frames=16,
    freeze_backbone=True
).to(device)

dummy_video = torch.randn(
    2, 16, 3, 112, 112
).to(device)

model.eval()

with torch.no_grad():
    output = model(dummy_video)

print("===== TRANSFORMER MODEL VERIFICATION =====")
print("Input shape:", dummy_video.shape)
print("Output shape:", output.shape)
print("Device:", device)
print("Number of classes:", output.shape[1])
print("Total parameters:", sum(
    p.numel() for p in model.parameters()
))
print("Trainable parameters:", sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
))

assert output.shape == (2, 7)

print("MODEL VERIFICATION PASSED")
