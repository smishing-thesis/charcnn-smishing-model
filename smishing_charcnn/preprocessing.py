"""Text masking applied before character encoding.

MASKING_RULES is an explicit, ordered list (not a plain function) so the same
rules can be serialized into the preprocessing JSON exported in export.py and
replicated by the Android client.
"""

from __future__ import annotations

import re

# pattern=None means the step isn't a regex substitution (only "lowercase" is).
MASKING_RULES: list[dict[str, str | None]] = [
    {"name": "lowercase", "pattern": None, "replacement": None},
    {"name": "url", "pattern": r"http\S+|www\.\S+", "replacement": " link "},
    {"name": "phone", "pattern": r"\b\d{7,}\b", "replacement": " call "},
    {"name": "digit", "pattern": r"\d", "replacement": "0"},
    {"name": "whitespace", "pattern": r"\s+", "replacement": " "},
]


def mask_text(text: str) -> str:
    """Apply MASKING_RULES, in order, to a single string."""
    t = str(text)
    for rule in MASKING_RULES:
        if rule["name"] == "lowercase":
            t = t.lower()
        else:
            t = re.sub(rule["pattern"], rule["replacement"], t)
    return t.strip()
