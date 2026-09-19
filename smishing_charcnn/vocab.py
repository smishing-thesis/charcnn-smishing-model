"""Character vocabulary and encoding.

PAD_IDX and UNK_IDX are kept distinct so the model can tell "no more text"
apart from "an unrecognized character was here".
"""

from __future__ import annotations

import numpy as np

PAD_IDX = 0
UNK_IDX = 1

ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"
    "áéíóúüñ"
    "0"  # digits are normalized to 0 during masking, so only one slot is needed
    " .,;:!?¿¡"
    "'\"/\\|_@#$%^&*~`+-=<>()[]{}"
)


def build_char2idx(alphabet: str = ALPHABET) -> dict[str, int]:
    """Map each alphabet character to an index starting at 2 (0=pad, 1=unk)."""
    return {c: i + 2 for i, c in enumerate(alphabet)}


def vocab_size(alphabet: str = ALPHABET) -> int:
    return len(alphabet) + 2  # + pad + unk


def encode(text: str, char2idx: dict[str, int], maxlen: int) -> list[int]:
    """Encode `text` into a fixed-length sequence of character indices.

    Characters outside the alphabet map to UNK_IDX; anything past `maxlen` is
    truncated; anything shorter is right-padded with PAD_IDX.
    """
    seq = [char2idx.get(c, UNK_IDX) for c in text[:maxlen]]
    return seq + [PAD_IDX] * (maxlen - len(seq))


def encode_batch(
    texts: list[str], char2idx: dict[str, int], maxlen: int
) -> np.ndarray:
    return np.array([encode(t, char2idx, maxlen) for t in texts], dtype=np.int32)
