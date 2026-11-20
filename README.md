# Video Action Recognition Using Deep Learning

## Project Overview
This project recognizes human actions from video clips using deep learning models. It uses the UCF101 dataset and compares three video classification approaches.

## Dataset
The project initially explored 10 selected UCF101 action classes. The final model comparison used 7 classes:

- ApplyEyeMakeup
- Biking
- CliffDiving
- Diving
- PlayingGuitar
- PlayingViolin
- SoccerJuggling

The final 7-class split contains:
- Training: 556 videos
- Validation: 142 videos
- Testing: 276 videos

Each video is represented by 16 uniformly sampled RGB frames resized to 112x112 pixels.

The original UCF101 videos are not included in this repository and must be obtained separately.

## Models
Three architectures were implemented and evaluated:

1. CNN + LSTM
2. ResNet18 + LSTM
3. ResNet18 + Transformer

Each final model was evaluated using three random seeds.

## Final Results

| Model | Mean Accuracy | Accuracy Std | Mean Macro F1 | Macro F1 Std |
|---|---:|---:|---:|---:|
| CNN + LSTM | 59.66% | 2.41 pp | 57.90% | 3.68 pp |
| ResNet18 + LSTM | 93.72% | 0.75 pp | 93.19% | 0.71 pp |
| ResNet18 + Transformer | 91.43% | 1.23 pp | 91.27% | 1.36 pp |

The ResNet18 + LSTM model achieved the best overall performance.

Note: The final 7-class subset was selected after exploratory experiments with the initial 10 classes. Therefore, the final test results should be interpreted with this selection process in mind.

## Project Structure

- `src/`: Dataset utilities, preprocessing, model training, and evaluation scripts.
- `src/models/`: Model architecture definitions.
- `checkpoints/`: Saved model weights.
- `experiments/`: Training logs, evaluation results, and comparison outputs.
- `configs/`: Project configuration files.
- `contributions/`: Project contribution materials.
- `demo/`: Streamlit demonstration application.
- `deliverables/`: Project requirements and preprocessing documentation.
- `report/`: Project reports.
- `Phase1_Shared/`: Shared initial dataset preparation files and documentation.

## Installation

Create and activate a Python virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the Demo

From the project root directory, run:

```bash
python -m streamlit run demo/app.py
```

Upload a video to view the predicted action class and confidence scores.

## Requirements
- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Scikit-learn
- Streamlit

## Notes
- The demo uses the trained 7-class ResNet18 + LSTM checkpoint.
- Checkpoint files must remain in the expected `checkpoints/` directory.
- The dataset is not distributed with this project.

## Model Checkpoints

The trained model checkpoint is available on Google Drive:

- **ResNet Mean Pool Ablation Checkpoint (42.73 MB):** [Download Checkpoint](https://drive.google.com/file/d/1VH6DJbCmpO0ygM1iApkxh5WX_H_ClVg4/view?usp=drive_link)

The checkpoint is shared with the project TAs.
