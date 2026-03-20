"""
Train a MobileNetV2-based plant disease detection model
on the PlantVillage dataset.

Usage:
    python scripts/train_model.py --data_dir /path/to/PlantVillage --epochs 20

Dataset: Download from https://www.kaggle.com/datasets/emmarex/plantdisease
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json

# ── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 38
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = 'model/plant_disease_model.h5'
HISTORY_SAVE_PATH = 'model/training_history.json'

def build_model(num_classes: int) -> Model:
    """Build MobileNetV2 transfer learning model."""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    # Freeze base model initially
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=outputs), base_model

def get_data_generators(data_dir: str):
    """Create train/val/test data generators with augmentation."""
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )
    val_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

    train_gen = train_datagen.flow_from_directory(
        data_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training', shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        data_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation', shuffle=False
    )
    return train_gen, val_gen

def plot_history(history, save_path='model/training_plots.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['accuracy'], label='Train Acc')
    axes[0].plot(history['val_accuracy'], label='Val Acc')
    axes[0].set_title('Accuracy over Epochs')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history['loss'], label='Train Loss')
    axes[1].plot(history['val_loss'], label='Val Loss')
    axes[1].set_title('Loss over Epochs')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] Training plots saved to {save_path}")

def train(data_dir: str, epochs: int = 20, fine_tune_epochs: int = 10):
    os.makedirs('model', exist_ok=True)

    print("[INFO] Building model...")
    model, base_model = build_model(NUM_CLASSES)

    print("[INFO] Loading data...")
    train_gen, val_gen = get_data_generators(data_dir)

    print(f"[INFO] Found {train_gen.samples} training samples, {val_gen.samples} validation samples.")
    print(f"[INFO] Classes: {len(train_gen.class_indices)}")

    # Save class indices
    with open('model/class_indices.json', 'w') as f:
        json.dump(train_gen.class_indices, f, indent=2)

    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', verbose=1),
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7, verbose=1),
        TensorBoard(log_dir='model/logs')
    ]

    # Phase 1: Train only top layers
    print("\n[INFO] Phase 1: Training classification head...")
    model.compile(optimizer=Adam(LEARNING_RATE), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history1 = model.fit(
        train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks, verbose=1
    )

    # Phase 2: Fine-tune top layers of base model
    print("\n[INFO] Phase 2: Fine-tuning base model (last 30 layers)...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(optimizer=Adam(LEARNING_RATE / 10), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history2 = model.fit(
        train_gen, epochs=fine_tune_epochs, validation_data=val_gen,
        callbacks=callbacks, verbose=1
    )

    # Merge and save history
    combined = {
        k: history1.history[k] + history2.history[k]
        for k in history1.history
    }
    with open(HISTORY_SAVE_PATH, 'w') as f:
        json.dump(combined, f, indent=2)

    plot_history(combined)
    print(f"\n[INFO] Training complete. Model saved to {MODEL_SAVE_PATH}")
    val_acc = max(combined['val_accuracy'])
    print(f"[INFO] Best validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train plant disease detection model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to PlantVillage dataset directory')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--fine_tune_epochs', type=int, default=10)
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.fine_tune_epochs)
