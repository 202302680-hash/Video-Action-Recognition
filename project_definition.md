# Video Action Recognition Using Deep Learning

## 1. Problem Definition

The goal is to develop a deep learning system that recognizes human actions from video clips.

- Task: Multi-class video classification.
- Input: A video clip represented by a sequence of RGB frames.
- Output: One predicted action class and confidence scores.

## 2. Dataset

- Dataset: UCF101.
- Original dataset: 13,320 videos across 101 action classes.
- Initial exploration: 10 selected action classes.
- Final evaluation: 7 selected action classes.

### Final 7 Action Classes

1. ApplyEyeMakeup
2. Biking
3. CliffDiving
4. Diving
5. PlayingGuitar
6. PlayingViolin
7. SoccerJuggling

### Final Dataset Split

- Training: 556 videos.
- Validation: 142 videos.
- Testing: 276 videos.

Each video is represented by 16 uniformly sampled RGB frames resized to 112x112 pixels.

The final 7-class subset was selected after exploratory experiments with the initial 10 classes. This selection process should be considered when interpreting the final test results.

## 3. Project Objectives

- Prepare and preprocess video data.
- Extract uniformly sampled frames from video clips.
- Develop and compare deep learning architectures.
- Investigate pretrained and scratch-based approaches.
- Perform hyperparameter experiments and ablation studies.
- Evaluate model performance using multiple metrics.
- Analyze classification errors and computational cost.
- Develop and test a Streamlit demonstration application.

## 4. Research Hypotheses

- H1: Models that learn temporal information can improve video action recognition.
- H2: Pretrained image feature extractors can improve performance compared with training a CNN from scratch.
- H3: Different temporal modeling strategies can produce different accuracy and computational-cost trade-offs.
- H4: Some visually similar actions may cause classification errors.

## 5. Implemented Models

1. CNN + LSTM (trained from scratch).
2. ResNet18 + LSTM (pretrained backbone).
3. ResNet18 + Transformer (pretrained backbone).
4. ResNet18 + Mean Pooling (temporal ablation model).

## 6. Evaluation Metrics

- Accuracy.
- Macro Precision.
- Macro Recall.
- Macro F1-score.
- Confusion Matrix.
- Per-class performance.
- Mean and standard deviation across random seeds.
- Parameter count, checkpoint size, inference time, and throughput.

## 7. Experimental Work

- Consistent final data splits and preprocessing were used for model comparison.
- Three main architectures were evaluated using three random seeds.
- Hyperparameter experiments were conducted on the Transformer architecture.
- Scratch-based and pretrained approaches were compared.
- A temporal mean-pooling ablation was evaluated.
- Classification errors and misclassified video examples were analyzed.
- Computational cost was measured using the NVIDIA GeForce RTX 3060 Laptop GPU.

The mean-pooling ablation was evaluated using one seed, so its comparison with the three-seed Transformer results should be interpreted as preliminary.

## 8. Main Results

The ResNet18 + LSTM model achieved the best overall performance:

- Mean Accuracy: 93.72%.
- Accuracy Standard Deviation: 0.75 percentage points.
- Mean Macro F1-score: 93.19%.
- Macro F1 Standard Deviation: 0.71 percentage points.

The final comparison included 276 test videos.

## 9. Demonstration Application

A Streamlit application was implemented and tested successfully.

The application allows users to upload a video and displays:
- The predicted action class.
- Confidence scores for the supported classes.

The demo uses the trained 7-class ResNet18 + LSTM checkpoint.

## 10. Limitations

- The final evaluation covers 7 selected classes, not all 101 UCF101 classes.
- The final 7-class subset was selected after initial exploratory experiments on 10 classes.
- The mean-pooling ablation was evaluated using one seed only.
- The original UCF101 dataset is not distributed with the project.

## 11. Expected Outcome

A working deep learning system for recognizing seven selected human actions from video clips, supported by experimental comparisons, error analysis, computational-cost measurements, and a tested Streamlit demonstration.
