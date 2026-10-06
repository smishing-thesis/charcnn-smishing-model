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


def test_parse_local_legit_replaces_placeholders():
    raw = pd.DataFrame(
        {
            "text": [
                "Hola <NAMED_ENTITY>, visita <URL> o llama a <PHONE_NUMBER>, "
                "escribe a <EMAIL_ADDRESS>"
            ],
            "label": [0],
        }
    )
    df = sources.parse_local_legit(raw)
    text = df.iloc[0]["text"]
    assert "<" not in text
    assert "http://url" in text
    assert "000000000" in text
    assert "mail@mail.com" in text


def test_parse_local_legit_catch_all_covers_lowercase_and_unknown_tokens():
    raw = pd.DataFrame(
        {"text": ["visita <link> para mas info sobre <DATE_TIME>"], "label": [0]}
    )
    df = sources.parse_local_legit(raw)
    assert "<" not in df.iloc[0]["text"]


def test_imc25_and_local_legit_apply_identical_placeholder_rules():
    text = "Hola <NAMED_ENTITY>, tu paquete: <URL> llama <PHONE_NUMBER> <EMAIL_ADDRESS> <OTHER>"
    imc25 = sources.parse_imc25(
        pd.DataFrame({"text": [text], "language": ["spanish"]}), language="spanish"
    )
    legit = sources.parse_local_legit(pd.DataFrame({"text": [text], "label": [0]}))
    assert imc25.iloc[0]["text"] == legit.iloc[0]["text"]


def test_parse_local_legit_keeps_text_without_placeholders_unchanged():
    raw = pd.DataFrame({"text": ["  Tu codigo es 123456, no lo compartas  "], "label": [0]})
    df = sources.parse_local_legit(raw)
    assert df.iloc[0]["text"] == "Tu codigo es 123456, no lo compartas"
