from smishing_charcnn.config import AppConfig, load_config


def test_defaults_match_seo_et_al_hyperparameters():
    config = AppConfig()
    assert config.model.embedding_dim == 48
    assert config.model.conv_filters == 192
    assert config.model.kernel_size == 12
    assert config.model.dense_units == 10
    assert config.preprocessing.maxlen == 160
    assert config.training.test_size == 0.2
    assert config.training.validation_split == 0.1


def test_load_config_returns_defaults_when_file_missing(tmp_path):
    config = load_config(tmp_path / "does_not_exist.yaml")
    assert config == AppConfig()


def test_load_config_merges_overrides_from_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "training:\n  epochs: 5\n  batch_size: 16\ndata:\n  min_text_length: 20\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.training.epochs == 5
    assert config.training.batch_size == 16
    assert config.data.min_text_length == 20
    # unrelated fields keep their defaults
    assert config.model.embedding_dim == 48
    assert config.training.test_size == 0.2
