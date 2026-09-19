import pandas as pd
import pytest

from smishing_charcnn.data import dataset


def _row(text, y, source):
    return {"text": text, "y": y, "source": source, "lang": "es"}


def test_validate_both_classes_present_raises_on_single_class():
    df = pd.DataFrame([_row("solo smishing aqui", 1, "IMC25")])
    with pytest.raises(dataset.DatasetValidationError, match="only 1 class"):
        dataset.validate_both_classes_present(df)


def test_validate_both_classes_present_passes_with_two_classes():
    df = pd.DataFrame(
        [
            _row("mensaje de smishing", 1, "IMC25"),
            _row("mensaje legitimo", 0, "LocalLegit"),
        ]
    )
    dataset.validate_both_classes_present(df)  # should not raise


def test_drop_duplicates_and_conflicts_removes_exact_duplicates():
    df = pd.DataFrame(
        [
            _row("Hola, como estas?", 0, "LocalLegit"),
            _row("hola como estas", 0, "LocalLegit"),  # same after normalization
        ]
    )
    clean, stats = dataset.drop_duplicates_and_conflicts(df)
    assert len(clean) == 1
    assert stats["duplicates_dropped"] == 1
    assert stats["conflicting_groups_dropped"] == 0


def test_drop_duplicates_and_conflicts_removes_contradictory_labels():
    df = pd.DataFrame(
        [
            _row("mismo texto normalizado", 0, "LocalLegit"),
            _row("mismo texto normalizado", 1, "IMC25"),
        ]
    )
    clean, stats = dataset.drop_duplicates_and_conflicts(df)
    assert len(clean) == 0
    assert stats["conflicting_groups_dropped"] == 1


def test_filter_short_texts():
    df = pd.DataFrame([_row("hi", 0, "LocalLegit"), _row("a much longer message here", 0, "LocalLegit")])
    filtered = dataset.filter_short_texts(df, min_length=15)
    assert len(filtered) == 1


def test_build_dataset_raises_when_legit_corpus_is_empty():
    imc25_df = pd.DataFrame([_row("mensaje de smishing bastante largo", 1, "IMC25")])
    empty_legit = pd.DataFrame(columns=["text", "y", "source", "lang"])
    with pytest.raises(dataset.DatasetValidationError):
        dataset.build_dataset(imc25_df, empty_legit, min_length=15)
