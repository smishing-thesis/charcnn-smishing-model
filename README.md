# charcnn-smishing-model

On-device smishing detection for Spanish SMS. Character-level CNN reimplementing
Seo et al. (2024), *On-device smishing classifier resistant to text evasion
attack* (IEEE Access 12, 4762-4779, CC BY 4.0), exported to TFLite for
low-end Android devices.

## Install

Python 3.12, Windows.

```bash
python -m venv .venv
source .venv/Scripts/activate      # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

- **IMC25** — Spanish smishing messages, downloaded automatically (smishing-only).
- **Local legit corpus** — not published: a CSV at
  `data/raw/legit_es.csv` with columns `text`, `label` (`label=0`). Required
  before `fetch-data`/`train`; no synthetic data is generated to replace it.

## Usage

```bash
python main.py fetch-data    # download + combine + validate
python main.py train         # preprocess, encode, split, train
python main.py evaluate      # metrics + robustness suite
python main.py export        # TFLite export + parity check + Android config
python main.py all           # run all four
```

All paths/hyperparameters come from `config.yaml`, overridable per flag, e.g.
`python main.py train --epochs 30`.

Outputs go to `models/` (gitignored): `model.keras`, `smishing_charcnn.tflite`,
`preprocessing_config.json` (alphabet, char2idx, maxlen, masking rules — the
Android client must replicate this exactly), `training_history.png`.

## Structure

```
smishing_charcnn/
├── config.py        # paths/hyperparameters
├── data/
│   ├── sources.py    # IMC25 + local legit corpus loaders
│   └── dataset.py     # combine, dedup, class validation
├── preprocessing.py  # masking rules
├── vocab.py          # character encoding (pad=0, unk=1)
├── modeling.py        # Char-CNN architecture
├── training.py         # seeding, split, training loop
├── evaluation.py       # metrics + robustness suite
├── export.py           # TFLite export + parity check
└── cli.py               # entry point
```

## Test

```bash
pytest
```

## Citation

Seo, J. W., Lee, J. S., Kim, H., Lee, J., Han, S., Cho, J., & Lee, C. H.
(2024). On-device smishing classifier resistant to text evasion attack. IEEE
Access, 12, 4762-4779.
