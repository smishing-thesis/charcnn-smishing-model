"""Combine sources into the final labeled dataset.

Dedup and contradictory-label removal run here, before the train/test split
in training.py, to avoid leaking near-duplicate text across the split.
"""

from __future__ import annotations

import re

import pandas as pd


class DatasetValidationError(Exception):
    """Raised when the combined dataset cannot be safely trained on."""


def combine_sources(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True)


def filter_short_texts(df: pd.DataFrame, min_length: int) -> pd.DataFrame:
    return df[df["text"].str.len() >= min_length].reset_index(drop=True)


def normalize_for_dedup(text: str) -> str:
    """Collapse a message to a coarse, punctuation-free form used only to
    detect near-duplicates and label conflicts across sources."""
    t = text.lower()
    t = re.sub(r"[^a-z0-9áéíóúñü ]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def drop_duplicates_and_conflicts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove exact near-duplicates and messages with contradictory labels
    across sources, before any split. Returns (clean_df, stats)."""
    df = df.copy()
    df["_norm"] = df["text"].apply(normalize_for_dedup)

    label_counts = df.groupby("_norm")["y"].nunique()
    conflicting_norms = label_counts[label_counts > 1].index
    n_conflicts = len(conflicting_norms)
    df = df[~df["_norm"].isin(conflicting_norms)]

    n_before_dedup = len(df)
    df = df.drop_duplicates(subset="_norm", keep="first")
    n_duplicates = n_before_dedup - len(df)

    df = df.drop(columns="_norm").reset_index(drop=True)
    stats = {
        "conflicting_groups_dropped": int(n_conflicts),
        "duplicates_dropped": int(n_duplicates),
    }
    return df, stats


def validate_both_classes_present(df: pd.DataFrame) -> None:
    """Abort if the dataset has only one class. Never fills the missing
    class with synthetic data or oversampling."""
    n_classes = df["y"].nunique()
    if n_classes < 2:
        present = sorted(df["y"].unique().tolist())
        raise DatasetValidationError(
            f"Dataset has only {n_classes} class(es) after combining sources "
            f"(labels present: {present}). IMC25 alone only contains "
            "smishing (y=1); you must provide the local legitimate-message "
            "corpus (data/raw/legit_es.csv, label=0) to train a binary "
            "classifier. Synthetic data or oversampling will not be used to "
            "fill the missing class."
        )


def build_dataset(
    imc25_df: pd.DataFrame, legit_df: pd.DataFrame, min_length: int
) -> pd.DataFrame:
    """Full combination pipeline used by the `fetch-data` CLI stage."""
    df = combine_sources([imc25_df, legit_df])
    df = filter_short_texts(df, min_length)
    df, stats = drop_duplicates_and_conflicts(df)
    print(
        f"Dropped {stats['conflicting_groups_dropped']} conflicting-label "
        f"group(s) and {stats['duplicates_dropped']} duplicate(s) "
        "before any train/test split."
    )
    validate_both_classes_present(df)
    return df
