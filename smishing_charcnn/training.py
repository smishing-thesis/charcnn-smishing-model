"""Seeding, train/test split, and the training loop.

set_seeds gives deterministic results on CPU. TensorFlow does not guarantee
full determinism on GPU even with seeds fixed.
"""

from __future__ import annotations

import random

import matplotlib

matplotlib.use("Agg")  # no display available when run from the CLI on Windows
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def stratified_split(
    X: np.ndarray, y: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    validation_split: float,
    patience: int,
) -> keras.callbacks.History:
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    return model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=1,
    )


def save_history_plot(history: keras.callbacks.History, output_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["loss"], label="train")
    ax1.plot(history.history["val_loss"], label="validation")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax2.plot(history.history["accuracy"], label="train")
    ax2.plot(history.history["val_accuracy"], label="validation")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
