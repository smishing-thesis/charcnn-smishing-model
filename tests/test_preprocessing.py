from smishing_charcnn.preprocessing import mask_text


def test_lowercases():
    assert mask_text("HOLA") == "hola"


def test_masks_urls():
    assert mask_text("visit http://bit.ly/xyz now") == "visit link now"
    assert mask_text("go to www.example.com today") == "go to link today"


def test_masks_long_digit_sequences_as_call():
    assert mask_text("llamar al 987654321 ahora") == "llamar al call ahora"


def test_normalizes_short_digits_to_zero():
    assert mask_text("codigo 4589") == "codigo 0000"


def test_collapses_whitespace():
    assert mask_text("hola    mundo") == "hola mundo"


def test_url_masking_runs_before_digit_normalization():
    # the URL rule must consume the digits inside the link before the
    # generic digit->0 rule would otherwise mangle them
    result = mask_text("visit http://example.com/123 now")
    assert "link" in result
    assert "123" not in result
