# DL and GenAI NPPE 1 - Chest X-ray Pathology Classification

**DL and GenAI NPPE 1, IIT Madras, T1 2026** \
**Rank: 54 / 778 (Top 7%)**

- [Kaggle Competition](https://www.kaggle.com/competitions/26-t-1-dl-gen-ainppe-1)
- [Leaderboard (23f1002653)](https://www.kaggle.com/competitions/26-t-1-dl-gen-ainppe-1/leaderboard?search=23f1002653)
- [My Notebook](https://www.kaggle.com/code/rahulmaurya026/23f1002653-notebook-26t1)

---

## Problem Statement

Given a dataset of 51,043 labeled chest X-ray images, classify each image into one of **20 thoracic pathology classes** (e.g., Atelectasis, Cardiomegaly, Pneumonia, Effusion, No Finding, etc.).

**What makes it challenging:**

- **Severe class imbalance** - "No Finding" makes up ~67% of all samples. At the extreme end, Pneumomediastinum has only 5 training examples.
- **Asymmetric scoring** - A False Negative (missing a disease) is penalized -5, a False Positive is -1, and a True Positive earns +1. Simply predicting "No Finding" everywhere destroys your score.
- **Macro-averaged evaluation** - Every class, including the rarest ones, contributes equally to the final score. You cannot ignore minority classes.
- **Subtle visual differences** - Pathologies are often indistinguishable even by eye, demanding strong learned features.

---

## My Approach

- **Started with a custom CNN baseline** - built a 5-block CNN from scratch to validate the pipeline, but it heavily overfit toward the majority class.
- **Switched to transfer learning with DenseNet121** - loaded ImageNet-pretrained weights. DenseNet121 is architecturally validated for chest X-ray data (used in the original NIH ChestX-ray14 paper), and its dense connectivity aids gradient flow and feature reuse.
- **Replaced the classifier head** with a custom two-layer head: `Linear(1024 → 512) → ReLU → Dropout(0.3) → Linear(512 → 20)`.
- **Designed X-ray-appropriate augmentation** - Resize → RandomCrop, horizontal flip, ±10° rotation, brightness/contrast jitter (no hue/saturation for grayscale images), RandomErasing. ImageNet normalization for the pretrained backbone.
- **Computed class weights** using six strategies (inverse frequency, sqrt scaling, log scaling, neg/pos ratio, effective number of samples, uniform). Selected sqrt scaling for stable training.
- **Implemented Focal Loss** from scratch with a `(1 - pt)^gamma` modulating factor to focus training on hard, misclassified examples rather than the abundant easy negatives.
- **Used Adam optimizer** at a constant `lr=1e-4`, and wrapped the model in `nn.DataParallel` to utilize both Kaggle T4 GPUs.
- **Built a full checkpoint system** - saves model weights, optimizer state, scheduler state, epoch index, and training history, enabling exact resumption if training is interrupted.
- **Implemented early stopping** with patience=5 on validation loss to prevent overfitting.
- **Generated submission** in the required one-hot encoded format, with an assertion check to verify exactly one label per row.

---

## Folder Structure

```
dl-genai-nppe1/
│
├── main.ipynb                # Full training notebook (run on Kaggle)
├── problem-statement.pdf     # Officially released competition problem statement
│
├── data/
│   ├── train.csv             # Training labels in one-hot encoded format (51,043 rows)
│   ├── test.csv              # Test image IDs for inference (17,015 rows)
│   └── sample_submission.csv # Submission format reference
│
├── checkpoints/
│   └── best_model.pth        # Best saved model checkpoint (~87 MB)
│
├── reports/
│   ├── 23f1002653_report.md  # Detailed project report
│   ├── 23f1002653_code.md    # Full notebook code (markdown export)
│   └── x-ray-rahul-notebook.pdf # Notebook exported as PDF
│
└── archive/
    ├── main-vidu.ipynb       # Vidu's full training notebook
    └── vidu-reports/
        ├── 23f2000575_report.md     # Vidu's detailed project report
        ├── 23f2000575_code.md       # Vidu's full notebook code (markdown export)
        └── x-ray-vidu-notebook.pdf  # Vidu's notebook exported as PDF
```

> **Note:** The `data/images/` directory (68k+ X-ray PNG files) is not tracked in this repository due to size. It is available via the Kaggle competition dataset.
