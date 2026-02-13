"""
UI Bug AI — CNN Model Training Script (MobileNetV2 Transfer Learning)
Train a TensorFlow/Keras model to classify UI bug screenshots.

Usage:
    python train_model.py

Dataset Structure Required:
    dataset/
    ├── Layout Broken/
    ├── Text Overflow/
    ├── Dark Mode Issue/
    ├── Alignment Issue/
    └── No Bug/
"""

import os
import json
import numpy as np
from datetime import datetime

# ── TensorFlow Setup ─────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ── Configuration ────────────────────────────────────────────────────
CONFIG = {
    "dataset_dir": "dataset",
    "model_save_path": "model/ui_bug_classifier.keras",
    "img_size": 128,
    "batch_size": 32,
    "epochs": 20,
    "validation_split": 0.2,
    "learning_rate": 0.0001,
    "class_names": [
        "Alignment Issue",
        "Dark Mode Issue",
        "Layout Broken",
        "No Bug",
        "Text Overflow"
    ]
}

IMG_SIZE = CONFIG["img_size"]


def build_model(num_classes: int) -> models.Sequential:
    """Build a MobileNetV2 transfer learning model."""
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze pretrained layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def load_dataset():
    """Load dataset from directory structure."""
    dataset_dir = CONFIG["dataset_dir"]

    if not os.path.exists(dataset_dir):
        print(f"\n❌ Dataset folder '{dataset_dir}' not found!")
        print(f"\nPlease create the following folder structure:")
        print(f"")
        print(f"  {dataset_dir}/")
        for cls in CONFIG["class_names"]:
            print(f"  ├── {cls}/")
            print(f"  │   ├── screenshot_001.png")
            print(f"  │   └── ...")
        print(f"")
        print(f"Each subfolder should contain PNG/JPG screenshots for that bug category.")
        print(f"Aim for at least 50–100 images per category for decent results.")
        return None, None

    print(f"\n📂 Loading dataset from: {dataset_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=CONFIG["validation_split"],
        subset="training",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=CONFIG["batch_size"],
        label_mode="categorical"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=CONFIG["validation_split"],
        subset="validation",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=CONFIG["batch_size"],
        label_mode="categorical"
    )

    class_names = train_ds.class_names
    print(f"✅ Found classes: {class_names}")
    print(f"📊 Training batches: {len(train_ds)} | Validation batches: {len(val_ds)}")

    # Normalize pixel values to [0, 1]
    normalization = layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    # Prefetch for performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def train():
    """Main training loop."""
    print("=" * 60)
    print("  🧠 UI Bug AI — Model Training (MobileNetV2)")
    print("=" * 60)
    print(f"  Backbone:      MobileNetV2 (ImageNet pretrained)")
    print(f"  Image Size:    {IMG_SIZE}×{IMG_SIZE}")
    print(f"  Batch Size:    {CONFIG['batch_size']}")
    print(f"  Epochs:        {CONFIG['epochs']}")
    print(f"  Learning Rate: {CONFIG['learning_rate']}")
    print(f"  EarlyStopping: patience=5")
    print("=" * 60)

    # Load data
    train_ds, val_ds = load_dataset()
    if train_ds is None:
        return

    # Build model
    num_classes = len(CONFIG["class_names"])
    model = build_model(num_classes)
    model.summary()

    # Callbacks
    os.makedirs(os.path.dirname(CONFIG["model_save_path"]), exist_ok=True)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        CONFIG["model_save_path"],
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # Train
    print(f"\n🚀 Starting training...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["epochs"],
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )

    # Save final model
    model.save(CONFIG["model_save_path"])

    # Save training history
    history_data = {
        "accuracy": [float(v) for v in history.history['accuracy']],
        "val_accuracy": [float(v) for v in history.history['val_accuracy']],
        "loss": [float(v) for v in history.history['loss']],
        "val_loss": [float(v) for v in history.history['val_loss']],
        "epochs_completed": len(history.history['accuracy']),
        "best_val_accuracy": float(max(history.history['val_accuracy'])),
        "trained_at": datetime.now().isoformat(),
        "config": {k: v for k, v in CONFIG.items() if k != "class_names"},
        "architecture": "MobileNetV2 + GlobalAvgPool + Dense(128) + Softmax(5)"
    }

    history_path = "model/training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2, default=str)

    # Final summary
    best_acc = max(history.history['val_accuracy']) * 100
    print(f"\n{'=' * 60}")
    print(f"  ✅ Training Complete!")
    print(f"  📈 Best Validation Accuracy: {best_acc:.1f}%")
    print(f"  💾 Model saved to: {CONFIG['model_save_path']}")
    print(f"  📊 History saved to: {history_path}")
    print(f"{'=' * 60}")
    print(f"\n  Restart the Flask server to use the trained model:")
    print(f"  python app.py")


if __name__ == '__main__':
    train()
