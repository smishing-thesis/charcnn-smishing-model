# charcnn-smishing-model

On-device smishing detection for Spanish SMS. A character-level CNN
reimplementing the hyperparameters of Seo, J. W., Lee, J. S., Kim, H., Lee,
J., Han, S., Cho, J., & Lee, C. H. (2024). *On-device smishing classifier
resistant to text evasion attack*. IEEE Access, 12, 4762-4779 (CC BY 4.0),
adapted to the Latin alphabet and Spanish, exported to TFLite for offline
inference on low-end Android devices targeting the Peruvian mobile ecosystem.

## Data sources

- **[IMC25](https://github.com/reportsmishing/Smishing-Dataset-IMC25)**
  (Agarwal et al., 2025, CC BY 4.0) — real Spanish smishing messages,
  downloaded automatically. It contains only the smishing class.
- **Local legitimate-message corpus** — not published anywhere and must be
  supplied manually. Place a CSV at `data/raw/legit_es.csv` with columns
  `text` and `label` (`label=0` for legitimate messages). The pipeline
  aborts with an explicit error if this file is missing, and aborts again if
  the combined dataset ends up with only one class: no synthetic data or
  oversampling is used to fill the missing class.

No real SMS content is ever committed to this repository (see `.gitignore`).

## Installation

Requires Python 3.12 on Windows, CPU only (TensorFlow 2.18 has no native GPU
support on Windows past TF 2.10, so nothing extra needs to be configured).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Project structure

```
smishing_charcnn/
├── config.py          # all paths/hyperparameters (config.yaml + CLI overrides)
├── data/
│   ├── sources.py      # IMC25 loader, local legit corpus loader
│   └── dataset.py       # combine, dedup, conflict resolution, class validation
├── preprocessing.py    # masking rules (URLs, phone numbers, digits, case)
├── vocab.py            # character alphabet and encoding (pad=0, unk=1)
├── modeling.py          # Char-CNN architecture (Seo et al. Table 3)
├── training.py          # seeding, stratified split, training loop
├── evaluation.py        # metrics + text-evasion robustness suite
├── export.py            # TFLite export, Keras/TFLite parity check, Android config
└── cli.py               # command-line entry point
```

Every module can be imported and unit-tested independently; see `tests/`.

## Usage

Each stage can run on its own or all together, always via `config.yaml`
(overridable per-flag):

```bash
python main.py fetch-data    # download IMC25 + local corpus, combine, validate
python main.py train         # preprocess, encode, split 80/20, train
python main.py evaluate      # metrics + robustness suite on the held-out test set
python main.py export        # TFLite export + parity check + Android preprocessing JSON
python main.py all           # run all four stages in order
```

Example override: `python main.py train --epochs 30 --batch-size 32`.

Outputs land under `models/` (gitignored): `model.keras`, `test_split.npz`,
`training_history.png`, `smishing_charcnn.tflite`, `preprocessing_config.json`.

### Preprocessing config for Android

`export` writes `models/preprocessing_config.json` with the character
alphabet, `char2idx` map, `maxlen`, the padding/unknown-character indices,
and the ordered masking rules applied during training. The Android/Kotlin
client must replicate this exact preprocessing character-for-character,
otherwise on-device predictions will not match the trained model.

## Reproducibility

Seeds (Python, numpy, TensorFlow) are fixed via `training.set_seeds(seed)`
(default `seed: 42` in `config.yaml`). This gives deterministic results on
CPU, the target platform for this project. TensorFlow does not guarantee
full determinism on GPU even with seeds fixed, due to non-deterministic GPU
kernel implementations — not a concern here since training runs on CPU only.

## Testing

```bash
pytest
```

Tests for `vocab`, `preprocessing`, `data.dataset` and `data.sources` do not
require TensorFlow. `tests/fixtures/legit_sample.csv` contains a handful of
invented (non-real) Spanish messages used only to test the local-corpus
loader.

## Citation

Seo, J. W., Lee, J. S., Kim, H., Lee, J., Han, S., Cho, J., & Lee, C. H.
(2024). On-device smishing classifier resistant to text evasion attack. IEEE
Access, 12, 4762-4779.
