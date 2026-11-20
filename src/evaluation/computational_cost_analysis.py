
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset_resnet_7class import UCF101ResNet7Dataset
from src.models.cnn_lstm_7class import CNNLSTM7
from src.models.cnn_lstm_resnet import CNNLSTMResNet
from src.models.cnn_transformer_resnet_improved import CNNTransformerResNetImproved
from src.models.resnet_mean_pool_ablation import ResNetMeanPoolAblation


# ==========================================
# CONFIGURATION
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 2
WARMUP_BATCHES = 5
REPEATS = 3

CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("COMPUTATIONAL COST ANALYSIS")
print("=" * 60)
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# TEST DATA
# ==========================================

test_dataset = UCF101ResNet7Dataset(
    "test",
    root_dir="experiments"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

print("Test videos:", len(test_dataset))
print("Batch size:", BATCH_SIZE)


# ==========================================
# MODEL CONFIGURATIONS
# ==========================================

models_config = [
    {
        "name": "CNN + LSTM",
        "class": CNNLSTM7,
        "kwargs": {"num_classes": 7, "hidden_size": 128},
        "checkpoint": CHECKPOINT_DIR / "cnn_lstm_7class_best.pth",
    },
    {
        "name": "ResNet18 + LSTM",
        "class": CNNLSTMResNet,
        "kwargs": {
            "num_classes": 7,
            "hidden_size": 128,
            "freeze_backbone": True
        },
        "checkpoint": CHECKPOINT_DIR / "resnet_lstm_7class_best.pth",
    },
    {
        "name": "ResNet18 + Transformer",
        "class": CNNTransformerResNetImproved,
        "kwargs": {"num_classes": 7},
        "checkpoint": CHECKPOINT_DIR / "transformer_resnet_7class_improved_best.pth",
    },
    {
        "name": "ResNet18 + Mean Pooling",
        "class": ResNetMeanPoolAblation,
        "kwargs": {
            "num_classes": 7,
            "freeze_backbone": True,
            "dropout": 0.4
        },
        "checkpoint": CHECKPOINT_DIR / "resnet_mean_pool_ablation_best.pth",
    },
]


# ==========================================
# CHECKPOINT LOADING
# ==========================================

def load_checkpoint(model, checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    return model


# ==========================================
# INFERENCE BENCHMARK
# ==========================================

def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def benchmark_model(model):

    model.eval()

    # Warm-up
    with torch.inference_mode():
        for batch_index, (videos, labels) in enumerate(test_loader):

            if batch_index >= WARMUP_BATCHES:
                break

            videos = videos.to(DEVICE, non_blocking=True)

            with torch.no_grad():
                _ = model(videos)

            synchronize()

    # Timed inference passes
    total_forward_time = 0.0
    total_videos = 0

    with torch.inference_mode():

        for repeat in range(REPEATS):

            for videos, labels in test_loader:

                videos = videos.to(DEVICE, non_blocking=True)

                # Synchronize before timing
                synchronize()
                start_time = time.perf_counter()

                outputs = model(videos)

                # Synchronize after forward pass
                synchronize()
                elapsed = time.perf_counter() - start_time

                total_forward_time += elapsed
                total_videos += videos.size(0)

    average_ms_per_video = (
        total_forward_time / total_videos
    ) * 1000

    throughput = total_videos / total_forward_time

    return average_ms_per_video, throughput


# ==========================================
# RUN ANALYSIS
# ==========================================

results = []

for config in models_config:

    model_name = config["name"]
    checkpoint_path = config["checkpoint"]

    print("\n" + "=" * 60)
    print("Evaluating:", model_name)
    print("=" * 60)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = config["class"](**config["kwargs"])
    model = load_checkpoint(model, checkpoint_path)
    model = model.to(DEVICE)

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    checkpoint_size_mb = (
        checkpoint_path.stat().st_size / (1024 ** 2)
    )

    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)
    print(f"Checkpoint size: {checkpoint_size_mb:.2f} MiB")

    avg_ms, throughput = benchmark_model(model)

    print(f"Average inference time: {avg_ms:.4f} ms/video")
    print(f"Throughput: {throughput:.2f} videos/second")

    results.append({
        "model": model_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "checkpoint_size_mib": round(checkpoint_size_mb, 4),
        "inference_ms_per_video": round(avg_ms, 4),
        "throughput_videos_per_second": round(throughput, 4),
        "test_videos": len(test_dataset),
        "batch_size": BATCH_SIZE,
        "benchmark_repeats": REPEATS,
        "warmup_batches": WARMUP_BATCHES,
        "device": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"
    })

    del model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


# ==========================================
# SAVE RESULTS
# ==========================================

output_file = OUTPUT_DIR / "computational_cost_comparison.csv"

fieldnames = list(results[0].keys())

with open(output_file, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


# ==========================================
# DISPLAY SUMMARY
# ==========================================

print("\n" + "=" * 70)
print("FINAL COMPUTATIONAL COST RESULTS")
print("=" * 70)

for result in results:

    print("\nModel:", result["model"])
    print("Parameters:", f'{result["total_parameters"]:,}')
    print("Trainable:", f'{result["trainable_parameters"]:,}')
    print("Checkpoint:", result["checkpoint_size_mib"], "MiB")
    print("Inference:", result["inference_ms_per_video"], "ms/video")
    print("Throughput:", result["throughput_videos_per_second"], "videos/sec")

print("\nResults saved to:", output_file)
print("\nComputational cost analysis completed.")
