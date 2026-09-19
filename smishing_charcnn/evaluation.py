"""Metrics and the text-evasion robustness suite."""

from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["ham", "smishing"]
        ),
    }


# (expected_label, message) pairs: legitimate/smishing text in English and
# Spanish, plus leetspeak-obfuscated variants of the smishing ones.
ROBUSTNESS_CASES: list[tuple[str, str]] = [
    ("Legitimate", "hey are we still meeting for lunch tomorrow"),
    (
        "Legitimate",
        "your package will be delivered today between 2 and 4 pm",
    ),
    ("Smishing", "congratulations you won a free prize call now to claim"),
    ("Obfuscated", "c0ngratulati0ns y0u w0n a fr33 pr1ze ca11 n0w t0 c1aim"),
    (
        "Smishing",
        "urgent your account has been suspended verify at http://bit.ly/xyz",
    ),
    (
        "Obfuscated",
        "urg3nt y0ur acc0unt has b33n susp3nd3d v3rify at http://bit.ly/xyz",
    ),
    ("Legitimate", "hola, ya sali del trabajo, llego en 20 minutos"),
    (
        "Legitimate",
        "Tu codigo de verificacion es 458920. No lo compartas con nadie.",
    ),
    (
        "Legitimate",
        "Tu pedido llegara manana entre 2 y 4 pm. Gracias por comprar.",
    ),
    (
        "Smishing",
        "BCP: Su cuenta ha sido bloqueada por seguridad. Verifique aqui http://url",
    ),
    (
        "Smishing",
        "Interbank: detectamos un cargo no reconocido, ingrese a http://url",
    ),
    (
        "Smishing",
        "SUNAT: tiene una deuda pendiente. Regularice en http://url o sera multado",
    ),
    (
        "Obfuscated",
        "8CP: Su cu3nta ha sid0 bl0queada. Verifiqu3 aqui http://url",
    ),
    (
        "Obfuscated",
        "1nterbank: carg0 n0 rec0nocid0 detectad0, ingres3 a http://url",
    ),
]


def run_robustness_suite(
    predict_fn: Callable[[str], float], cases: list[tuple[str, str]] = ROBUSTNESS_CASES
) -> pd.DataFrame:
    """predict_fn takes a raw message and returns the smishing probability."""
    rows = []
    for label, message in cases:
        prob = float(predict_fn(message))
        rows.append(
            {
                "expected": label,
                "message": message,
                "probability": prob,
                "prediction": "smishing" if prob > 0.5 else "legitimate",
            }
        )
    return pd.DataFrame(rows)
