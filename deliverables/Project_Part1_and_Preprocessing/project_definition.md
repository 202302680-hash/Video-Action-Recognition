
# Video Action Recognition Using Deep Learning

## 1. Problem Definition

The goal of this project is to develop a deep learning system that recognizes human actions from video clips.

- Task: Multi-class video classification.
- Input: A video containing a sequence of RGB frames.
- Output: One predicted human action class.

## 2. Dataset

- Dataset: UCF101
- Source: https://www.crcv.ucf.edu/data/UCF101.php
- Original dataset: 13,320 videos and 101 action classes.
- Selected classes: 10 classes.

### Selected Action Classes
1. ApplyEyeMakeup
2. Basketball
3. Biking
4. CliffDiving
5. CricketBowling
6. Diving
7. PlayingGuitar
8. PlayingViolin
9. SoccerJuggling
10. TennisSwing

## 3. Project Objectives

- Load and preprocess video clips.
- Extract frames from videos.
- Build deep learning models for action recognition.
- Compare different model architectures.
- Evaluate model performance.
- Analyze classification errors.
- Build a simple video prediction demo.

## 4. Research Hypotheses

- H1: Models that learn temporal information will perform better than a frame-based CNN.
- H2: Pretrained models can achieve better performance than models trained from scratch.
- H3: Increasing the number of sampled frames may improve accuracy but also increases computational cost.
- H4: Visually similar actions may cause more classification errors.

## 5. Proposed Models

1. Baseline: CNN.
2. Classic model: CNN + LSTM.
3. Modern model: Video Transformer.
4. Advanced model: 3D CNN / 3D ResNet.

## 6. Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Per-class performance
- Mean and standard deviation across random seeds.

## 7. Experimental Plan

- Use consistent data splits and preprocessing.
- Compare the proposed architectures.
- Perform hyperparameter experiments.
- Study different data strategies.
- Compare pretrained and scratch training where applicable.
- Perform ablation studies.
- Analyze errors and computational cost.

## 8. Expected Outcome

A working deep learning system that predicts one of ten human actions from a video clip, with experimental comparisons and a live demonstration.
