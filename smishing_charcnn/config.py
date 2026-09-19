"""Application configuration: all paths and hyperparameters, loaded from YAML."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclasses.dataclass
class DataConfig:
    imc25_url: str = (
        "https://raw.githubusercontent.com/reportsmishing/"
        "Smishing-Dataset-IMC25/main/dataset/final_dataset_output.csv"
    )
    imc25_language: str = "spanish"
    legit_corpus_path: str = "data/raw/legit_es.csv"
    min_text_length: int = 15
    processed_dataset_path: str = "data/processed/dataset.csv"


@dataclasses.dataclass
class PreprocessingConfig:
    maxlen: int = 160


@dataclasses.dataclass
class ModelConfig:
    # From Seo et al. (2024), Table 3 - keep as-is to match the paper.
    embedding_dim: int = 48
    conv_filters: int = 192
    kernel_size: int = 12
    dense_units: int = 10


@dataclasses.dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 64
    validation_split: float = 0.1
    test_size: float = 0.2
    early_stopping_patience: int = 3
    model_path: str = "models/model.keras"
    split_cache_path: str = "models/test_split.npz"


@dataclasses.dataclass
class ExportConfig:
    tflite_path: str = "models/smishing_charcnn.tflite"
    preprocessing_config_path: str = "models/preprocessing_config.json"


@dataclasses.dataclass
class AppConfig:
    seed: int = 42
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = dataclasses.field(
        default_factory=PreprocessingConfig
    )
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    export: ExportConfig = dataclasses.field(default_factory=ExportConfig)


def _merge_dataclass(instance: Any, overrides: dict[str, Any]) -> Any:
    """Return a copy of `instance` with `overrides` applied (shallow, per field)."""
    updates: dict[str, Any] = {}
    for field in dataclasses.fields(instance):
        if field.name not in overrides:
            continue
        value = overrides[field.name]
        current = getattr(instance, field.name)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            updates[field.name] = _merge_dataclass(current, value)
        else:
            updates[field.name] = value
    return dataclasses.replace(instance, **updates)


def load_config(path: str | Path | None = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load AppConfig from a YAML file, falling back to defaults for any
    field the file does not set. `path=None` returns pure defaults."""
    config = AppConfig()
    if path is None:
        return config

    path = Path(path)
    if not path.exists():
        return config

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return _merge_dataclass(config, raw)
