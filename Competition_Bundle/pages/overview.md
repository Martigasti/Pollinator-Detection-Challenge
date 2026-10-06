# Pollinator Detection

## Introduction
Monitoring insect pollinators is key to tracking biodiversity, but reviewing hours of flower video by hand does not scale. This challenge asks participants to automatically detect whether a pollinator is visiting a flower in a given video frame.

## Task
Binary image classification: each sample is a 224×224 RGB frame labeled **1 (visitor)** or **0 (no visitor)**. The visitor class is rare (about 2.7% of the ~30k training frames), so a model that always predicts "no visitor" is useless despite reaching ~97% accuracy.

Frames come from video sequences. Train/validation splits must keep every frame of a sequence on the same side to avoid leakage; the starting kit provides a sequence-aware, class-balanced split.

## Evaluation
- **Balanced accuracy** (main metric): the mean of per-class recall. Predicting a single class scores 0.5.
- **F1-score** on the visitor class: balances false alarms (precision) against missed visitors (recall).

## How to join this competition?
- Log in or create an account on [Codabench](https://www.codabench.org/)
- Download the starting kit and follow the notebook
- Register for the competition and submit from the `My Submissions` tab
