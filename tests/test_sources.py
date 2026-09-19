import pandas as pd
import pytest

from smishing_charcnn.data import sources


def test_read_local_legit_csv_raises_clear_error_when_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="not published"):
        sources.read_local_legit_csv(missing_path)


def test_load_local_legit_corpus_from_fixture():
    df = sources.load_local_legit_corpus("tests/fixtures/legit_sample.csv")
    assert len(df) == 3
    assert set(df.columns) == {"text", "y", "source", "lang"}
    assert (df["y"] == 0).all()
    assert (df["source"] == "LocalLegit").all()


def test_parse_local_legit_raises_on_missing_columns():
    raw = pd.DataFrame({"text": ["hola"]})  # missing 'label'
    with pytest.raises(ValueError, match="missing required column"):
        sources.parse_local_legit(raw)


def test_parse_imc25_filters_by_language_and_sets_label():
    raw = pd.DataFrame(
        {
            "text": ["mensaje en espanol", "message in english"],
            "language": ["Spanish", "English"],
        }
    )
    df = sources.parse_imc25(raw, language="spanish")
    assert len(df) == 1
    assert df.iloc[0]["y"] == 1
    assert df.iloc[0]["source"] == "IMC25"


def test_parse_imc25_replaces_placeholders():
    raw = pd.DataFrame(
        {
            "text": [
                "Hola <NAMED_ENTITY>, visita <URL> o llama a <PHONE_NUMBER>, "
                "escribe a <EMAIL_ADDRESS>"
            ],
            "language": ["spanish"],
        }
    )
    df = sources.parse_imc25(raw, language="spanish")
    text = df.iloc[0]["text"]
    assert "<" not in text
    assert "http://url" in text
    assert "000000000" in text
    assert "mail@mail.com" in text


def test_parse_imc25_catch_all_covers_lowercase_and_unknown_tokens():
    raw = pd.DataFrame(
        {
            "text": ["visita <link> para mas info sobre <DATE_TIME>"],
            "language": ["spanish"],
        }
    )
    df = sources.parse_imc25(raw, language="spanish")
    assert "<" not in df.iloc[0]["text"]
