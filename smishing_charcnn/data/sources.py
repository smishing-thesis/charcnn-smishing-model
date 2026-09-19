"""Loaders for the two message sources: IMC25 and the local legit corpus.

Each loader is split into an IO half (`fetch_*` / `read_*`) and a pure
transformation half (`parse_*`) so parsing can be unit-tested without network
access or a real file on disk.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REQUIRED_LOCAL_COLUMNS = {"text", "label"}

# IMC25 anonymizes entities/links with placeholder tokens; map them to
# processable text instead of dropping them.
_IMC25_PLACEHOLDER_RULES = [
    (r"<NAMED_ENTITY>", " "),
    (r"<URL>", " http://url "),
    (r"<PHONE_NUMBER>", " 000000000 "),
    (r"<EMAIL_ADDRESS>", " mail@mail.com "),
    (r"<[A-Za-z_]+>", " "),  # any remaining placeholder token
    (r"\s+", " "),
]


def fetch_imc25_raw(url: str) -> pd.DataFrame:
    """Download the IMC25 dataset CSV. Thin IO wrapper, not unit-tested."""
    return pd.read_csv(url)


def parse_imc25(raw_df: pd.DataFrame, language: str = "spanish") -> pd.DataFrame:
    """Filter IMC25 to one language and normalize it to the common schema
    (text, y, source, lang). IMC25 contains only smishing, so y is always 1.
    """
    df = raw_df[raw_df["language"].astype(str).str.lower() == language.lower()]
    df = df[["text"]].copy()
    df["text"] = df["text"].astype(str)
    for pattern, replacement in _IMC25_PLACEHOLDER_RULES:
        df["text"] = df["text"].str.replace(pattern, replacement, regex=True)
    df["text"] = df["text"].str.strip()

    df["y"] = 1
    df["source"] = "IMC25"
    df["lang"] = "es"
    return df.reset_index(drop=True)


def load_imc25(url: str, language: str = "spanish") -> pd.DataFrame:
    return parse_imc25(fetch_imc25_raw(url), language=language)


def read_local_legit_csv(path: str | Path) -> pd.DataFrame:
    """Read the local legit corpus, or raise a clear error if it's missing
    instead of failing silently."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Local legitimate-message corpus not found at '{path}'.\n"
            "This corpus is not published and must be supplied manually: "
            "place a CSV with 'text' and 'label' columns (label=0 for "
            "legitimate messages) at that path. IMC25 alone only contains "
            "the smishing class, so the pipeline cannot train without it."
        )
    return pd.read_csv(path)


def parse_local_legit(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the local legit corpus to the common schema."""
    missing = REQUIRED_LOCAL_COLUMNS - set(raw_df.columns)
    if missing:
        raise ValueError(
            f"Local legit corpus is missing required column(s): {sorted(missing)}. "
            f"Expected columns: {sorted(REQUIRED_LOCAL_COLUMNS)}."
        )

    df = raw_df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["y"] = df["label"].astype(int)
    df = df.drop(columns="label")
    df["source"] = "LocalLegit"
    df["lang"] = "es"
    return df.reset_index(drop=True)


def load_local_legit_corpus(path: str | Path) -> pd.DataFrame:
    return parse_local_legit(read_local_legit_csv(path))
