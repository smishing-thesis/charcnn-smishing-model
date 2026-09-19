"""Command-line entry point: fetch-data / train / evaluate / export / all."""

from __future__ import annotations

import argparse
import dataclasses
import json

import numpy as np
import pandas as pd
from tensorflow import keras

from smishing_charcnn import evaluation, export, modeling, preprocessing, training, vocab
from smishing_charcnn.config import AppConfig, load_config
from smishing_charcnn.data import dataset, sources


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the YAML config file."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smishing-charcnn")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser(
        "fetch-data", help="Download IMC25, load the local legit corpus, combine and validate."
    )
    _add_config_arg(fetch)
    fetch.add_argument("--imc25-url")
    fetch.add_argument("--imc25-language")
    fetch.add_argument("--legit-corpus-path")
    fetch.add_argument("--min-text-length", type=int)
    fetch.add_argument("--processed-dataset-path")

    train = subparsers.add_parser("train", help="Train the Char-CNN model.")
    _add_config_arg(train)
    train.add_argument("--processed-dataset-path")
    train.add_argument("--maxlen", type=int)
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--validation-split", type=float)
    train.add_argument("--test-size", type=float)
    train.add_argument("--patience", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument("--model-path")
    train.add_argument("--split-cache-path")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate the trained model.")
    _add_config_arg(evaluate)
    evaluate.add_argument("--model-path")
    evaluate.add_argument("--split-cache-path")
    evaluate.add_argument("--maxlen", type=int)
    evaluate.add_argument(
        "--metrics-output", help="Optional path to also save metrics as JSON."
    )

    export_p = subparsers.add_parser(
        "export", help="Export to TFLite, verify parity, write the preprocessing config."
    )
    _add_config_arg(export_p)
    export_p.add_argument("--model-path")
    export_p.add_argument("--maxlen", type=int)
    export_p.add_argument("--tflite-path")
    export_p.add_argument("--preprocessing-config-path")

    all_p = subparsers.add_parser("all", help="Run fetch-data, train, evaluate and export in order.")
    _add_config_arg(all_p)

    return parser


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    data_overrides = {
        k: v
        for k, v in {
            "imc25_url": getattr(args, "imc25_url", None),
            "imc25_language": getattr(args, "imc25_language", None),
            "legit_corpus_path": getattr(args, "legit_corpus_path", None),
            "min_text_length": getattr(args, "min_text_length", None),
            "processed_dataset_path": getattr(args, "processed_dataset_path", None),
        }.items()
        if v is not None
    }
    preprocessing_overrides = {
        k: v
        for k, v in {"maxlen": getattr(args, "maxlen", None)}.items()
        if v is not None
    }
    training_overrides = {
        k: v
        for k, v in {
            "epochs": getattr(args, "epochs", None),
            "batch_size": getattr(args, "batch_size", None),
            "validation_split": getattr(args, "validation_split", None),
            "test_size": getattr(args, "test_size", None),
            "early_stopping_patience": getattr(args, "patience", None),
            "model_path": getattr(args, "model_path", None),
            "split_cache_path": getattr(args, "split_cache_path", None),
        }.items()
        if v is not None
    }
    export_overrides = {
        k: v
        for k, v in {
            "tflite_path": getattr(args, "tflite_path", None),
            "preprocessing_config_path": getattr(args, "preprocessing_config_path", None),
        }.items()
        if v is not None
    }
    seed_override = getattr(args, "seed", None)

    config = dataclasses.replace(config, data=dataclasses.replace(config.data, **data_overrides))
    config = dataclasses.replace(
        config, preprocessing=dataclasses.replace(config.preprocessing, **preprocessing_overrides)
    )
    config = dataclasses.replace(
        config, training=dataclasses.replace(config.training, **training_overrides)
    )
    config = dataclasses.replace(
        config, export=dataclasses.replace(config.export, **export_overrides)
    )
    if seed_override is not None:
        config = dataclasses.replace(config, seed=seed_override)
    return config


def run_fetch_data(config: AppConfig) -> None:
    imc25_df = sources.load_imc25(config.data.imc25_url, config.data.imc25_language)
    legit_df = sources.load_local_legit_corpus(config.data.legit_corpus_path)
    df = dataset.build_dataset(imc25_df, legit_df, config.data.min_text_length)

    print(f"Final dataset: {len(df)} messages")
    print(pd.crosstab(df["source"], df["y"], margins=True))

    out_path = config.data.processed_dataset_path
    import os

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved to {out_path}")


def run_train(config: AppConfig) -> None:
    training.set_seeds(config.seed)

    df = pd.read_csv(config.data.processed_dataset_path)
    dataset.validate_both_classes_present(df)

    clean_texts = df["text"].apply(preprocessing.mask_text)
    char2idx = vocab.build_char2idx()
    X = vocab.encode_batch(clean_texts.tolist(), char2idx, config.preprocessing.maxlen)
    y = df["y"].values

    X_train, X_test, y_train, y_test = training.stratified_split(
        X, y, config.training.test_size, config.seed
    )
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    model = modeling.build_model(
        vocab.vocab_size(),
        config.preprocessing.maxlen,
        config.model.embedding_dim,
        config.model.conv_filters,
        config.model.kernel_size,
        config.model.dense_units,
    )
    model.summary()

    history = training.train_model(
        model,
        X_train,
        y_train,
        config.training.epochs,
        config.training.batch_size,
        config.training.validation_split,
        config.training.early_stopping_patience,
    )

    import os

    os.makedirs(os.path.dirname(config.training.model_path) or ".", exist_ok=True)
    model.save(config.training.model_path)
    print(f"Model saved to {config.training.model_path}")

    history_plot_path = os.path.join(
        os.path.dirname(config.training.model_path) or ".", "training_history.png"
    )
    training.save_history_plot(history, history_plot_path)
    print(f"Training curves saved to {history_plot_path}")

    os.makedirs(os.path.dirname(config.training.split_cache_path) or ".", exist_ok=True)
    np.savez(config.training.split_cache_path, X_test=X_test, y_test=y_test)
    print(f"Test split cached at {config.training.split_cache_path}")


def run_evaluate(config: AppConfig, metrics_output: str | None = None) -> None:
    model = keras.models.load_model(config.training.model_path)
    split = np.load(config.training.split_cache_path)
    X_test, y_test = split["X_test"], split["y_test"]

    probs = model.predict(X_test, verbose=0).flatten()
    pred = (probs > 0.5).astype(int)
    metrics = evaluation.compute_metrics(y_test, pred)

    for key in ["accuracy", "precision", "recall", "f1"]:
        print(f"{key.capitalize():<10}: {metrics[key]:.4f}")
    print(
        f"\nConfusion matrix -> TN={metrics['true_negatives']} "
        f"FP={metrics['false_positives']} "
        f"FN={metrics['false_negatives']} (most severe) "
        f"TP={metrics['true_positives']}"
    )
    print(f"\n{metrics['classification_report']}")

    char2idx = vocab.build_char2idx()

    def predict_fn(message: str) -> float:
        clean = preprocessing.mask_text(message)
        x = np.array([vocab.encode(clean, char2idx, config.preprocessing.maxlen)])
        return float(model.predict(x, verbose=0)[0][0])

    robustness_df = evaluation.run_robustness_suite(predict_fn)
    print("\nRobustness suite:")
    print(robustness_df.to_string(index=False))

    if metrics_output:
        with open(metrics_output, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in metrics.items() if k != "classification_report"}, f, indent=2)
        print(f"\nMetrics saved to {metrics_output}")


def run_export(config: AppConfig) -> None:
    model = keras.models.load_model(config.training.model_path)
    tflite_bytes = export.export_tflite(model, config.export.tflite_path)
    kb = len(tflite_bytes) / 1024
    print(f"TFLite model size: {kb:.1f} KB (Seo et al. 2024 reference: 127 KB)")

    char2idx = vocab.build_char2idx()

    def encode_fn(message: str) -> list[int]:
        return vocab.encode(preprocessing.mask_text(message), char2idx, config.preprocessing.maxlen)

    interpreter = export.load_interpreter(tflite_bytes)
    sample_cases = [msg for _, msg in evaluation.ROBUSTNESS_CASES]
    export.verify_parity(model, interpreter, sample_cases, encode_fn)
    print(f"Parity check passed on {len(sample_cases)} robustness cases.")

    export.export_preprocessing_config(
        config.export.preprocessing_config_path,
        vocab.ALPHABET,
        char2idx,
        config.preprocessing.maxlen,
        vocab.PAD_IDX,
        vocab.UNK_IDX,
        preprocessing.MASKING_RULES,
    )
    print(f"Preprocessing config saved to {config.export.preprocessing_config_path}")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config = _apply_overrides(config, args)

    if args.command == "fetch-data":
        run_fetch_data(config)
    elif args.command == "train":
        run_train(config)
    elif args.command == "evaluate":
        run_evaluate(config, getattr(args, "metrics_output", None))
    elif args.command == "export":
        run_export(config)
    elif args.command == "all":
        run_fetch_data(config)
        run_train(config)
        run_evaluate(config)
        run_export(config)


if __name__ == "__main__":
    main()
