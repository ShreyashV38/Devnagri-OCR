"""
train_model.py - Devanagari Character Classifier Training Pipeline

Usage: python train_model.py
"""

import sys
import os
import shutil

# Fix Windows encoding before ANY print statement
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'dataset', 'DevanagariHandwrittenCharacterDataset')
TRAIN_DIR = os.path.join(DATASET_ROOT, 'Train')
TEST_DIR = os.path.join(DATASET_ROOT, 'Test')
BACKEND_DIR = os.path.join(PROJECT_ROOT, 'backend')
MODEL_PATH = os.path.join(BACKEND_DIR, 'devanagari_master.h5')
LABELS_PATH = os.path.join(BACKEND_DIR, 'label_classes.npy')
PLOT_PATH = os.path.join(BACKEND_DIR, 'training_history.png')

IMG_SIZE = 32
EPOCHS = 50
BATCH_SIZE = 128

# ---------------------------------------------------------------------------
# STEP 0: CLEANUP OLD FILES
# ---------------------------------------------------------------------------
CLEANUP_TARGETS = [
    'core_engine',
    'backend/api.py',
    'backend/__init__.py',
    'backend/__pycache__',
    'crash_log.txt',
    'dataset_structure.txt',
    'train_output.txt',
    'test_import.py',
    'output.txt',
    '__pycache__',
]

print('\n--- Cleaning up old files ---')
for target in CLEANUP_TARGETS:
    full_path = os.path.join(PROJECT_ROOT, target)
    if os.path.isdir(full_path):
        shutil.rmtree(full_path)
        print(f'  [DELETED] folder: {target}')
    elif os.path.isfile(full_path):
        os.remove(full_path)
        print(f'  [DELETED] file: {target}')
print('  Cleanup done.\n')

# ---------------------------------------------------------------------------
# IMPORTS (after cleanup, so old core_engine doesn't interfere)
# ---------------------------------------------------------------------------
import glob
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten,
    Dropout, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import cv2


# ---------------------------------------------------------------------------
# LABEL EXTRACTION
# ---------------------------------------------------------------------------
def extract_label(folder_name):
    """
    Extract label from folder name by splitting on '_' and taking last part.
    'character_1_ka' -> 'ka'
    'digit_0' -> 'digit_0' (split gives ['digit', '0'], but we want 'digit_0')
    """
    if folder_name.startswith('digit_'):
        return folder_name  # keep as-is: digit_0, digit_1, etc.
    parts = folder_name.split('_')
    return parts[-1]


# ---------------------------------------------------------------------------
# IMAGE LOADING
# ---------------------------------------------------------------------------
def load_dataset(root_dir, dataset_name):
    """
    Load all images from root_dir/class_folder/*.png structure.
    Returns (images_array, labels_array).
    """
    print(f'\n{"="*60}')
    print(f'  Loading: {dataset_name}')
    print(f'  Path:    {root_dir}')
    print(f'{"="*60}')

    assert os.path.isdir(root_dir), f"Directory not found: {root_dir}"

    class_folders = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    assert len(class_folders) > 0, f"No class folders found in {root_dir}"

    all_images = []
    all_labels = []

    for folder_name in class_folders:
        label = extract_label(folder_name)
        folder_path = os.path.join(root_dir, folder_name)

        count = 0
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
            for img_path in glob.glob(os.path.join(folder_path, ext)):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype(np.float32) / 255.0
                img = img.reshape(IMG_SIZE, IMG_SIZE, 1)
                all_images.append(img)
                all_labels.append(label)
                count += 1

        print(f'  Loading {label}: {count} images')

    images = np.array(all_images)
    labels = np.array(all_labels)
    n_classes = len(np.unique(labels))
    print(f'\n  [TOTAL] {len(images)} samples, {n_classes} classes')
    return images, labels


# ---------------------------------------------------------------------------
# CNN MODEL
# ---------------------------------------------------------------------------
def build_model(num_classes):
    """Build the CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ---------------------------------------------------------------------------
# PLOT TRAINING HISTORY
# ---------------------------------------------------------------------------
def save_plot(history, path):
    """Save accuracy and loss plots side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'], label='Train', color='#2196F3', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val', color='#FF5722', linewidth=2, linestyle='--')
    axes[0].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='Train', color='#2196F3', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val', color='#FF5722', linewidth=2, linestyle='--')
    axes[1].set_title('Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [SAVED] Plot: {path}')


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print('\n' + '=' * 60)
    print('  DEVANAGARI CHARACTER CLASSIFIER - TRAINING')
    print('=' * 60)

    os.makedirs(BACKEND_DIR, exist_ok=True)

    # ---- Load data ----
    train_images, train_labels = load_dataset(TRAIN_DIR, 'Training Set (Train/)')
    test_images, test_labels = load_dataset(TEST_DIR, 'Test Set (Test/)')

    # Combine them since the local Train folder is incomplete (missing 43 classes!)
    all_images = np.concatenate((train_images, test_images))
    all_labels = np.concatenate((train_labels, test_labels))

    # ---- Encode labels ----
    print(f'\n--- Encoding labels ---')
    le = LabelEncoder()
    encoded = le.fit_transform(all_labels)
    num_classes = len(le.classes_)
    print(f'  Classes: {num_classes}')
    print(f'  Labels:  {list(le.classes_)}')

    np.save(LABELS_PATH, le.classes_)
    print(f'  [SAVED] Labels: {LABELS_PATH}')

    categorical = to_categorical(encoded, num_classes=num_classes)

    # ---- Train/val split ----
    print(f'\n--- Splitting 80/20 ---')
    X_train, X_val, y_train, y_val = train_test_split(
        all_images, categorical,
        test_size=0.2,
        random_state=42,
        stratify=encoded
    )
    print(f'  Train: {X_train.shape[0]} samples')
    print(f'  Val:   {X_val.shape[0]} samples')

    # ---- Build model ----
    print(f'\n--- Building CNN ({num_classes} classes) ---')
    model = build_model(num_classes)
    model.summary()

    # ---- Train ----
    print(f'\n--- Training: {EPOCHS} epochs, batch={BATCH_SIZE} ---')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # ---- Evaluate on validation ----
    print(f'\n--- Validation Results ---')
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f'  Val Accuracy: {val_acc * 100:.2f}%')
    print(f'  Val Loss:     {val_loss:.4f}')

    # ---- Evaluate on Test set ----
    print(f'\n--- Loading Test Set ---')
    test_encoded = le.transform(test_labels)
    test_categorical = to_categorical(test_encoded, num_classes=num_classes)

    print(f'\n--- Test Results ---')
    test_loss, test_acc = model.evaluate(test_images, test_categorical, verbose=0)
    print(f'  Test Accuracy: {test_acc * 100:.2f}%')
    print(f'  Test Loss:     {test_loss:.4f}')

    # ---- Save model ----
    model.save(MODEL_PATH)
    print(f'  [SAVED] Model: {MODEL_PATH}')

    # ---- Save plot ----
    save_plot(history, PLOT_PATH)

    # ---- Summary ----
    print('\n' + '=' * 60)
    print('  TRAINING COMPLETE')
    print('=' * 60)
    print(f'  Model:    {MODEL_PATH}')
    print(f'  Labels:   {LABELS_PATH}')
    print(f'  Plot:     {PLOT_PATH}')
    print(f'  Classes:  {num_classes}')
    print(f'  Val Acc:  {val_acc * 100:.2f}%')
    print(f'  Test Acc: {test_acc * 100:.2f}%')
    print(f'\n  Next: python app.py')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    main()
