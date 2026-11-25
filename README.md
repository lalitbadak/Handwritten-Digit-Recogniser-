# Handwritten Digit Recognition using a Neural Network

A feed-forward neural network, built with TensorFlow/Keras, that classifies 28×28 grayscale images of handwritten digits (0–9). It reaches **97.15% validation accuracy** on 8,400 held-out images.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-ff6f00.svg)](https://www.tensorflow.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Workflow](#workflow)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Limitations & Next Steps](#limitations--next-steps)
- [Tech Stack](#tech-stack)

---

## Overview

Handwritten digit recognition is a classic computer vision problem and a good first project for neural networks. This notebook walks through the full pipeline:

1. Load pixel data from CSV
2. Normalize and reshape it into images
3. One-hot encode the labels
4. Train a fully connected neural network
5. Evaluate on a validation split and plot learning curves
6. Predict digits on an unlabeled test set and visualize the results

---

## Dataset

The data is in the MNIST-style CSV format used by the Kaggle [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) competition.

| Property | Value |
|---|---|
| Training file | `Train.csv` (42,000 rows × 785 columns) |
| Columns | `label` (the digit) + `pixel0` … `pixel783` |
| Image size | 28 × 28 pixels, grayscale (flattened to 784 values) |
| Pixel range | 0–255 |
| Classes | 10 (digits 0–9), roughly balanced at about 9–11% each |
| Test file | `test.csv` (same pixel columns, **no labels**) |

The dataset is not included in this repository because of its size (the training CSV is about 77 MB). Download it from Kaggle and place both files in `data/`.

---

## Model Architecture

A simple multi-layer perceptron (MLP):

| Layer | Output shape | Activation | Parameters |
|---|---|---|---|
| Input | 28 × 28 × 1 | | 0 |
| Flatten | 784 | | 0 |
| Dense | 128 | ReLU | 100,480 |
| Dense | 64 | ReLU | 8,256 |
| Dense | 10 | Softmax | 650 |
| **Total** | | | **109,386** |

**Training setup**
- Optimizer: Adam
- Loss: categorical cross-entropy
- Epochs: 10
- Batch size: 32
- Split: 80% train (33,600) / 20% validation (8,400), `random_state=42`

---

## Workflow

**Preprocessing**
- Separate the `label` column from the pixel columns
- Convert pixels to numeric, fill any missing values with 0
- Scale pixel values to the 0–1 range (divide by 255)
- Reshape each row into a `28 × 28 × 1` image
- One-hot encode labels into 10 classes

**Training and evaluation**
- Train for 10 epochs, tracking training and validation accuracy
- Report final validation accuracy and plot both accuracy curves
- Run `model.predict` on the test set and display the first five images with their predicted digits

---

## Results

| Metric | Value |
|---|---|
| Final training accuracy | 99.49% |
| Final validation accuracy | **97.15%** |
| Final validation loss | 0.129 |

Validation accuracy per epoch:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Val accuracy | 95.18% | 96.40% | 96.42% | 96.87% | 96.86% | 96.79% | 96.87% | **97.19%** | 96.70% | 97.15% |

**What the curves show:** the model learns quickly, passing 95% validation accuracy after a single epoch. After roughly epoch 4, training accuracy keeps climbing toward 99.5% while validation accuracy flattens near 97% and validation loss starts to rise. That gap is a sign of mild overfitting, which is typical for an MLP on image data.

The test set has no labels, so test predictions are checked visually only; no test accuracy is reported.

---

## Project Structure

```
.
├── Handwritten_Digit_Recognition_using_Neural_Network.ipynb   # Full pipeline
├── data/
│   ├── Train.csv      # Labeled training data (download separately)
│   └── test.csv       # Unlabeled test data (download separately)
├── requirements.txt
└── README.md
```

---

## Getting Started

### Option A: Google Colab

1. Open the notebook in Colab
2. Upload `Train.csv` and `test.csv` to the Colab session (the notebook reads them from `/content/`)
3. Run all cells

### Option B: Run locally

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Then open the notebook and change the two file paths from `/content/Train.csv` and `/content/test.csv` to `data/Train.csv` and `data/test.csv`.

### `requirements.txt`

```
numpy
pandas
matplotlib
scikit-learn
tensorflow
jupyter
```

---

## Limitations & Next Steps

**Current limitations**
- The network flattens each image, so it ignores spatial structure; a convolutional network usually does better on images.
- Overfitting appears after the first few epochs; there is no regularization or early stopping.
- Evaluation is a single validation split, and the notebook reports overall accuracy only (no confusion matrix or per-digit metrics).
- The trained model is not saved, so predictions require re-running training.
- No data augmentation, so the model may struggle with digits that are shifted, rotated, or drawn in unusual styles.

**Ideas for improvement**
- Replace the MLP with a **CNN** (Conv2D + MaxPooling layers), which typically reaches 99%+ on this dataset
- Add `Dropout` and `EarlyStopping` to reduce overfitting
- Use data augmentation (small rotations, shifts, zoom)
- Add a confusion matrix to see which digits get mixed up (for example 4 vs 9, or 3 vs 5)
- Save the model (`model.save(...)`) and build a Streamlit or Gradio app where users draw a digit and get a prediction
- Export predictions to a CSV for Kaggle submission

---

## Tech Stack

- **Python:** NumPy, pandas, matplotlib, scikit-learn
- **Deep learning:** TensorFlow / Keras
- **Environment:** Jupyter Notebook / Google Colab

---

## Acknowledgements

Dataset format from the Kaggle [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) competition, derived from the MNIST database of handwritten digits.
