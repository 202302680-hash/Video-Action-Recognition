import torch
from src.models.cnn_lstm_7class import CNNLSTM7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNLSTM7(num_classes=7).to(device)

x = torch.randn(2, 16, 3, 112, 112).to(device)

with torch.no_grad():
    output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Device:", device)
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("MODEL VERIFICATION PASSED")
