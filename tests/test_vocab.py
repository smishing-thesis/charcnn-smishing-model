from smishing_charcnn import vocab


def test_pad_and_unk_are_different_indices():
    assert vocab.PAD_IDX != vocab.UNK_IDX
    assert vocab.PAD_IDX == 0
    assert vocab.UNK_IDX == 1


def test_char2idx_starts_at_two():
    char2idx = vocab.build_char2idx()
    assert min(char2idx.values()) == 2
    assert vocab.PAD_IDX not in char2idx.values()
    assert vocab.UNK_IDX not in char2idx.values()


def test_vocab_size_accounts_for_pad_and_unk():
    assert vocab.vocab_size() == len(vocab.ALPHABET) + 2


def test_encode_pads_short_text_with_pad_idx():
    char2idx = vocab.build_char2idx()
    encoded = vocab.encode("ab", char2idx, maxlen=5)
    assert encoded == [
        char2idx["a"],
        char2idx["b"],
        vocab.PAD_IDX,
        vocab.PAD_IDX,
        vocab.PAD_IDX,
    ]


def test_encode_truncates_long_text():
    char2idx = vocab.build_char2idx()
    encoded = vocab.encode("abcdef", char2idx, maxlen=3)
    assert encoded == [char2idx["a"], char2idx["b"], char2idx["c"]]


def test_encode_maps_unknown_characters_to_unk_not_pad():
    char2idx = vocab.build_char2idx()
    encoded = vocab.encode("a中b", char2idx, maxlen=3)  # 中 not in alphabet
    assert encoded == [char2idx["a"], vocab.UNK_IDX, char2idx["b"]]


def test_encode_batch_shape():
    char2idx = vocab.build_char2idx()
    batch = vocab.encode_batch(["a", "bb"], char2idx, maxlen=4)
    assert batch.shape == (2, 4)
