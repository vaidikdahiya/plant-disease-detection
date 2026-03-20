"""
Evaluate the trained plant disease detection model.
Generates confusion matrix, classification report, and per-class accuracy.

Usage:
    python scripts/evaluate_model.py --data_dir /path/to/PlantVillage --test_split 0.1
"""

import os
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 32
MODEL_PATH = 'model/plant_disease_model.h5'


def evaluate(data_dir, test_split=0.1):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first.")

    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    test_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=test_split)
    test_gen = test_datagen.flow_from_directory(
        data_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation', shuffle=False
    )

    print("[INFO] Running inference on test set...")
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    class_names = list(test_gen.class_indices.keys())
    short_names = [c.split('___')[-1][:20] for c in class_names]

    acc = accuracy_score(y_true, y_pred)
    print(f"\n[RESULT] Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    print("\n[INFO] Classification Report:")
    report = classification_report(y_true, y_pred, target_names=short_names, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=short_names))

    # Save report
    os.makedirs('model', exist_ok=True)
    with open('model/evaluation_report.json', 'w') as f:
        json.dump({'accuracy': acc, 'report': report}, f, indent=2)

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=short_names, yticklabels=short_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig('model/confusion_matrix.png', dpi=150)
    print("[INFO] Confusion matrix saved to model/confusion_matrix.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--test_split', type=float, default=0.1)
    args = parser.parse_args()
    evaluate(args.data_dir, args.test_split)
