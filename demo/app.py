
import sys
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

# =========================
# PROJECT PATH
# =========================

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.models.cnn_lstm_resnet import CNNLSTMResNet


# =========================
# CONFIGURATION
# =========================

MODEL_PATH = ROOT_DIR / "checkpoints" / "resnet_lstm_7class_best.pth"

CLASSES = [
    "ApplyEyeMakeup",
    "Biking",
    "CliffDiving",
    "Diving",
    "PlayingGuitar",
    "PlayingViolin",
    "SoccerJuggling"
]

NUM_FRAMES = 16
IMAGE_SIZE = 112

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1)

STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1)


# =========================
# STREAMLIT PAGE
# =========================

st.set_page_config(
    page_title="Video Action Recognition",
    page_icon="🎬",
    layout="centered"
)


# =========================
# LOAD TRAINED MODEL
# =========================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}"
        )

    model = CNNLSTMResNet(
        num_classes=7,
        hidden_size=128,
        freeze_backbone=True
    )

    # Unfreeze Layer4, matching training setup
    for param in model.cnn[7].parameters():
        param.requires_grad = True

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


# =========================
# EXTRACT VIDEO FRAMES
# =========================

def extract_frames(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not open this video.")

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        raise ValueError("Could not determine video length.")

    indices = np.linspace(
        0,
        total_frames - 1,
        NUM_FRAMES,
        dtype=int
    )

    selected = {}
    target_indices = set(indices.tolist())

    frame_index = 0
    max_index = int(indices[-1])

    while frame_index <= max_index:

        success, frame = cap.read()

        if not success:
            break

        if frame_index in target_indices:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame = cv2.resize(
                frame,
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            selected[frame_index] = frame

        frame_index += 1

    cap.release()

    frames = [
        selected[index]
        for index in indices
        if index in selected
    ]

    if len(frames) == 0:
        raise ValueError("No readable frames found.")

    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1].copy())

    return np.stack(frames[:NUM_FRAMES])


# =========================
# PREPROCESS VIDEO
# =========================

def preprocess_video(video_path):

    frames = extract_frames(video_path)

    # (T, H, W, C) -> (T, C, H, W)
    video = torch.from_numpy(
        frames.copy()
    ).float() / 255.0

    video = video.permute(0, 3, 1, 2)

    # ImageNet normalization
    video = (video - MEAN) / STD

    # Add batch dimension
    # Shape: (1, 16, 3, 112, 112)
    video = video.unsqueeze(0)

    return video


# =========================
# PREDICTION
# =========================

def predict(video_path, model):

    video = preprocess_video(video_path)
    video = video.to(DEVICE)

    with torch.no_grad():

        outputs = model(video)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    probabilities = probabilities[0].cpu().numpy()

    predicted_index = int(
        np.argmax(probabilities)
    )

    return predicted_index, probabilities


# =========================
# USER INTERFACE
# =========================

st.title("🎬 Video Action Recognition")

st.write(
    "Upload a video to recognize the human action "
    "using our ResNet18 + LSTM deep learning model."
)

st.caption(
    "The model recognizes 7 action classes from UCF101."
)

st.divider()

st.subheader("Upload Your Video")

uploaded_file = st.file_uploader(
    "Choose a video",
    type=["avi", "mp4", "mov", "mkv"]
)

if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button(
        "Analyze Video",
        type="primary",
        use_container_width=True
    ):

        temp_path = None

        try:

            with st.spinner("Loading trained model..."):
                model = load_model()

            suffix = Path(
                uploaded_file.name
            ).suffix

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getvalue()
                )

                temp_path = temp_file.name

            with st.spinner(
                "Extracting frames and analyzing action..."
            ):

                predicted_index, probabilities = predict(
                    temp_path,
                    model
                )

            predicted_class = CLASSES[predicted_index]

            confidence = float(
                probabilities[predicted_index] * 100
            )

            st.success("Analysis completed!")

            st.subheader("Prediction Result")

            st.metric(
                "Predicted Action",
                predicted_class
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.subheader("Top 3 Predictions")

            top_indices = np.argsort(
                probabilities
            )[::-1][:3]

            for index in top_indices:

                class_name = CLASSES[index]

                score = float(
                    probabilities[index] * 100
                )

                st.write(
                    f"**{class_name}** — {score:.2f}%"
                )

                st.progress(
                    min(score / 100, 1.0)
                )

            st.caption(
                "Confidence is the model's predicted probability, "
                "not a guarantee that the prediction is correct."
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )

        finally:

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


st.divider()

st.caption(
    "Deep Learning Project | ResNet18 + LSTM | UCF101"
)
