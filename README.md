# Devanagari OCR

A CNN-based optical character recognizer for handwritten Devanagari script, served through a Gradio web interface.

## Problem / Motivation

Devanagari is used by over 600 million people across Hindi, Sanskrit, Marathi, and Nepali, yet robust handwriting recognition tools for it remain scarce compared to Latin-script alternatives. This project trains a convolutional neural network from scratch on the standard Devanagari Handwritten Character Dataset (46 classes: 36 consonants + 10 digits) and wraps it in a browser-based UI for single-character inference.

## Key Features

- Classifies 46 Devanagari character classes (consonants and digits) from handwritten input images
- Three-block CNN with batch normalization, dropout, and learning-rate scheduling -- achieves ~97% validation accuracy
- Gradio web UI for drag-and-drop image upload with real-time top-5 predictions and confidence scores
- Preprocessing pipeline handles RGBA/RGB/grayscale input, automatic background inversion, Otsu thresholding, contour-based cropping, and aspect-ratio-preserving padding
- Training script includes early stopping, learning-rate reduction on plateau, and automatic train/val/test evaluation with saved accuracy/loss plots

## Tech Stack

- **Language:** Python 3
- **Deep Learning:** TensorFlow 2.17 / Keras 3.4
- **Computer Vision:** OpenCV 4.8
- **ML Utilities:** scikit-learn 1.4, NumPy, Pandas
- **Web UI:** Gradio 4.44
- **Visualization:** Matplotlib

## Architecture

```
Input Image
    |
[Preprocessing]  -- grayscale conversion, background inversion,
    |                Otsu threshold, contour crop, pad to square, resize to 32x32
    v
[CNN Model]      -- Conv2D(32) -> BN -> MaxPool -> Dropout(0.25)
    |               Conv2D(64) -> BN -> MaxPool -> Dropout(0.25)
    |               Conv2D(128) -> BN -> MaxPool -> Dropout(0.25)
    |               Flatten -> Dense(256) -> Dropout(0.5) -> Softmax(46)
    v
[Gradio UI]      -- displays predicted character + top-5 ranked predictions
```

The model is a sequential CNN with three convolutional blocks, each followed by batch normalization, max-pooling, and dropout. A fully connected layer (256 units) feeds into a 46-class softmax output. Total model size is approximately 7.3 MB.

## Setup / Installation

**Prerequisites:** Python 3.10+, pip

```bash
# Clone the repository
git clone https://github.com/ShreyashV38/Devnagri-OCR.git
cd Devnagri-OCR

# Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Dataset setup:** Download the [Devanagari Handwritten Character Dataset](https://www.kaggle.com/datasets/shreenidhi26/devanagari-handwritten-character-dataset) from Kaggle and extract it so the directory structure is:

```
dataset/
  DevanagariHandwrittenCharacterDataset/
    Train/
    Test/
```

**Train the model** (skip if `backend/devanagari_master.h5` already exists):

```bash
python train_model.py
```

Training runs for up to 50 epochs with early stopping (typically converges in ~30 epochs on CPU).

## Usage

```bash
python app.py
```

This launches a Gradio interface at `http://127.0.0.1:7860`. Upload or drag-and-drop an image of a single handwritten Devanagari character. The app returns the predicted character with a confidence score and a ranked list of the top 5 predictions.

## Notable Technical Challenges

- **Robust input normalization for variable image quality.** Handwritten character images arrive in different formats (RGBA, RGB, grayscale), with either light or dark backgrounds, and with noise artifacts. The preprocessing pipeline chains alpha-channel extraction, adaptive background detection via border-pixel median, Otsu binarization, morphological closing, and contour-based cropping to produce clean 32x32 inputs -- handling edge cases that a naive resize would misclassify.

- **Dealing with an incomplete training split.** The local Train directory contained only 3 of 46 class folders, while Test had all 46. Rather than discard the majority of labeled data, the training script merges both splits and performs a stratified 80/20 re-split, ensuring every class is represented proportionally in both training and validation sets.

- **Achieving ~97% accuracy with a lightweight model.** The CNN is only 7.3 MB and runs inference on CPU, yet reaches ~97% validation accuracy across 46 classes. This was achieved by combining batch normalization (to stabilize training across classes with different stroke distributions), aggressive dropout (0.25 per conv block, 0.5 before softmax), and ReduceLROnPlateau scheduling to fine-tune convergence without overfitting.
