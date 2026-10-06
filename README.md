# Pollinator Detection Challenge

Design of a complete **CodaBench machine-learning competition** together with its baseline solution: detect insect pollinators visiting flowers in video frames, where only **~2.7%** of frames contain a visitor.

M1 Data Science "AI Challenge" course, Université Paris-Saclay (January 2026).
Team (Group 3): Baptiste Pras, Vladimir Herrera Nativi, Martin Leiva.

## What's in here

| Part | Content |
|---|---|
| [`Starting_Kit/starting_kit.ipynb`](Starting_Kit/starting_kit.ipynb) | Participant notebook: data loading, EDA, the HOG + XGBoost baseline, hyperparameter search, scoring, and submission |
| [`Competition_Bundle/`](Competition_Bundle/) | Everything CodaBench needs to host the challenge: ingestion program, scoring program, sample submission, pages, `competition.yaml` |

## The problem

- ~30k RGB frames (224×224) labeled *visitor* / *no visitor*, extracted from flower video sequences.
- **Extreme imbalance** (about 1 positive for 36 negatives), so plain accuracy is meaningless. The metrics are **balanced accuracy** and **F1** on the visitor class.
- **Leakage-safe split**: frames from the same sequence never appear in both train and test. A plain `GroupSplit` skews the class ratio, so the kit searches for a sequence-respecting split that keeps the same positive rate on both sides.

## Baseline: HOG features + XGBoost

The baseline is deliberately lightweight and runs on a laptop CPU (about 30 s per model on 10% of the data).

1. **EDA**: class distribution, sample grids, and PCA projections of raw pixels vs. HOG, grayscale HOG and multiscale HOG, to choose a feature representation.
2. **Features**: Histogram of Oriented Gradients (9 orientations, 16×16 cells), which captures insect shape (wings, legs, body) and is robust to color and lighting.
3. **Model**: XGBoost with `scale_pos_weight` and `max_delta_step` to push recall on the rare class without exploding false positives, plus L1/L2 regularization.
4. **Tuning**: a grid over HOG parameters, then a 4-phase randomized search over XGBoost hyperparameters (tree structure → learning rate → regularization → subsampling), then decision-threshold tuning.
5. **Rebalancing**: random oversampling to 10% positives works best. SMOTE consistently hurts, because the classes aren't separable in feature space, so interpolated "insects" are unrealistic.

### Results (held-out split)

| Training data | Balanced accuracy | F1 (visitor) | CV balanced acc. / F1 |
|---|---|---|---|
| 10% | 0.678 | 0.513 | 0.93 / 0.73 |
| 100% | 0.668 | 0.331 | 0.84 / 0.49 |

The gap between cross-validation and held-out scores shows significant overfitting. This is the main lever participants are invited to work on (regularization, better features, CNNs).

## Running locally

```bash
pip install numpy h5py scikit-learn scikit-image xgboost imbalanced-learn matplotlib seaborn tqdm gdown
jupyter notebook Starting_Kit/starting_kit.ipynb   # data (~4.5 GB) downloads automatically
```

To test the competition pipeline end to end, put the data in `Competition_Bundle/input_data/` and `reference_data/`, then run:

```bash
cd Competition_Bundle
python3 ingestion_program/run_ingestion.py   # trains sample_code_submission/model.py, writes predictions
python3 scoring_program/run_scoring.py       # computes balanced accuracy and F1
python3 utilities/compile_bundle.py          # builds the zip to upload to CodaBench
```

The bundle structure follows the course template by Ihsan Ullah.
