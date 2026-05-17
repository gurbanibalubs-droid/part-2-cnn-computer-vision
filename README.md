# Part 2: Computer Vision Problem Formulation and CNN Prototype

## Overview

This repository contains the solution for **Part 2** of the assignment. The goal is to formulate a computer vision problem and build a CNN-based prototype to classify images from a real-world dataset.

---

## Dataset

- **Source:** [Part 2 Dataset – Google Drive](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing)
- **Type:** Multi-class image dataset (folder-per-class structure)

> ⚠️ Dataset is not uploaded to this repository. Download it from the link above and place it in your Google Drive before running the notebook.

---

## Problem Type

**Image Classification** — Each image is assigned exactly one class label. The CNN learns to map raw pixel values to class probabilities using convolutional feature extraction followed by dense classification layers.

---

## Approach & Steps

### Task 1 – Problem Identification
Identified as Image Classification. Each image belongs to a single class; no bounding boxes or segmentation masks are required.

### Task 2 – Dataset Exploration
- Counted images per class and checked for imbalance
- Visualized sample images from each class
- Checked image dimensions and color modes

### Task 3 – Image Preprocessing
- Resized all images to **128×128**
- Normalized pixel values to **[0, 1]**
- Applied **data augmentation** (rotation, flip, shift, zoom) to training set
- **80/20 train-validation split** using `ImageDataGenerator`

### Task 4 – CNN Model Architecture

```
Input (128×128×3)
  → Conv2D(32) + BatchNorm + MaxPool
  → Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25)
  → Conv2D(128) + BatchNorm + MaxPool + Dropout(0.25)
  → Flatten
  → Dense(256, ReLU) + Dropout(0.5)
  → Dense(num_classes, Softmax)
```

- **Loss:** Categorical Cross-Entropy
- **Optimizer:** Adam (lr=0.001)
- **Callbacks:** EarlyStopping + ReduceLROnPlateau

### Task 5 – Training & Evaluation
- Plotted accuracy/loss curves for train and validation
- Generated confusion matrix
- Displayed 12 sample predictions with confidence scores (green = correct, red = wrong)

---

## Task 6: CNN Concept Explanation

**What is Convolution?**
A filter (small matrix) slides across the image computing dot products, producing feature maps that highlight edges, textures, and patterns at each location.

**Why is Pooling Used?**
MaxPooling reduces spatial dimensions, cuts computation, and makes the model translation-invariant — recognizing features regardless of their position in the image.

**Why is ReLU Used?**
ReLU (`max(0, x)`) introduces non-linearity without the vanishing gradient problem. It is fast, sparse, and allows CNNs to learn complex visual patterns efficiently.

**Why CNNs over Feed-Forward Networks for Images?**
CNNs use parameter sharing (one filter across the whole image) and local connectivity, reducing parameters drastically. A 128×128 RGB image has 49,152 values — fully connected layers would be impractical. CNNs handle this with a few thousand shared weights per filter.

---

## Task 7: Business Use Case – Healthcare

A CNN classifier like this can classify **chest X-rays** into categories (Normal, Pneumonia, COVID-19, Tuberculosis).

**Benefits:**
- Classifies images in milliseconds vs minutes of manual review
- Eliminates radiologist fatigue and inconsistency
- Scales to thousands of scans per hour
- Detects subtle patterns for earlier diagnosis

---

## Repository Structure

```
part-2-cnn-computer-vision/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
├── sample_predictions/
│   └── prediction_outputs.png
└── results/
    ├── class_distribution.png
    ├── sample_images.png
    ├── accuracy_loss_curves.png
    └── confusion_matrix.png
```

---

## How to Run

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `notebook.ipynb`
3. Mount Google Drive and update `DATASET_PATH` in the notebook
4. Run all cells — outputs save automatically to `results/` and `sample_predictions/`

---

## Requirements

See `requirements.txt`. Key libraries: `tensorflow`, `scikit-learn`, `matplotlib`, `seaborn`, `Pillow`
