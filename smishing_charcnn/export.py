"""TFLite export, Keras/TFLite parity check, and the Android preprocessing
config.

The Android client must replicate the exported preprocessing JSON
character-for-character, or on-device predictions won't match training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
from tensorflow import keras

try:
    # ai-edge-litert is the maintained successor to tf.lite.Interpreter.
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    # Not installed here. tf.lite.Interpreter still works on TF 2.x
    # (deprecated, removed in TF 2.20) so we fall back instead of breaking.
    from tensorflow.lite import Interpreter


class ParityError(Exception):
    """Raised when TFLite predictions diverge from the Keras model beyond
    tolerance."""


def export_tflite(model: keras.Model, output_path: str | Path) -> bytes:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_bytes)
    return tflite_bytes


def load_interpreter(tflite_bytes: bytes) -> Interpreter:
    interpreter = Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    return interpreter


def predict_tflite(interpreter: Interpreter, x: np.ndarray) -> float:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details["index"], x.astype(np.float32))
    interpreter.invoke()
    return float(interpreter.get_tensor(output_details["index"])[0][0])


def verify_parity(
    keras_model: keras.Model,
    interpreter: Interpreter,
    cases: list[str],
    encode_fn: Callable[[str], list[int]],
    tolerance: float = 1e-3,
) -> None:
    """Compare Keras vs TFLite predictions on `cases`; raise ParityError if
    any pair diverges by more than `tolerance`."""
    for message in cases:
        x = np.array([encode_fn(message)], dtype=np.float32)
        keras_pred = float(keras_model.predict(x, verbose=0)[0][0])
        tflite_pred = predict_tflite(interpreter, x)
        if abs(keras_pred - tflite_pred) > tolerance:
            raise ParityError(
                f"Keras/TFLite prediction mismatch for message {message!r}: "
                f"keras={keras_pred:.6f} tflite={tflite_pred:.6f} "
                f"(tolerance={tolerance})"
            )


def export_preprocessing_config(
    path: str | Path,
    alphabet: str,
    char2idx: dict[str, int],
    maxlen: int,
    pad_idx: int,
    unk_idx: int,
    masking_rules: list[dict],
) -> None:
    """Write the JSON the Android/Kotlin client must replicate exactly."""
    config = {
        "alphabet": alphabet,
        "char2idx": char2idx,
        "maxlen": maxlen,
        "pad_idx": pad_idx,
        "unk_idx": unk_idx,
        "masking_rules": masking_rules,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
