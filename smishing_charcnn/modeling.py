"""Char-CNN architecture. Defaults are from Seo et al. (2024), Table 3 -
do not change embedding_dim/conv_filters/kernel_size/dense_units."""

from __future__ import annotations

from tensorflow import keras


def build_model(
    vocab_size: int,
    maxlen: int,
    embedding_dim: int = 48,
    conv_filters: int = 192,
    kernel_size: int = 12,
    dense_units: int = 10,
) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(maxlen,), name="input"),
            keras.layers.Embedding(vocab_size, embedding_dim, name="embedding"),
            keras.layers.Conv1D(
                conv_filters, kernel_size, activation="relu", name="conv"
            ),
            keras.layers.GlobalMaxPooling1D(name="pooling"),
            keras.layers.Dense(dense_units, activation="relu", name="dense"),
            keras.layers.Dense(1, activation="sigmoid", name="output"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
