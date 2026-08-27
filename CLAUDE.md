# CLAUDE.md

## Project

Pure premium estimation framework for motor insurance using the French MTPL dataset. Two-stage frequency-severity model benchmarking GLM, XGBoost, and PyTorch neural network.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Python 3.14, venv at `./venv/`.

## Structure

```
notebooks/01_eda.ipynb          # EDA — run first
notebooks/02_glm.ipynb          # Poisson GLM (frequency) + Gamma GLM (severity)
notebooks/03_xgboost.ipynb      # XGBoost
notebooks/04_neural_net.ipynb   # PyTorch neural net
notebooks/05_comparison.ipynb   # Model comparison + pure premium estimator
data/                           # Raw and cleaned CSVs
models/                         # Saved model files
results/                        # Output charts, metric tables, saved predictions
dashboard/                      # Flask backend serving predictions from models/
```

## Publishing

Quarto site. To render locally:

```bash
quarto render
```

Output goes to `_site/`. Notebooks are rendered via Quarto — keep `_quarto.yml` and `notebooks.qmd` in sync when adding new notebooks.

## Modelling conventions

- Frequency target: `ClaimNb` (Poisson), with `log(Exposure)` as offset
- Severity target: `ClaimAmount`, capped at 99th percentile (16,451 EUR) — the cap is applied
  **in notebook 01**, so `claims_cleaned.csv` arrives already winsorized (265 claims sit at
  exactly 16,451). Notebooks 02–04 must not cap again.
- Categorical encoding: one-hot with `drop_first=True`
- Charts saved to `results/`, models saved to `models/`
- Notebooks 01–04 run independently, sharing only the cleaned CSVs in `data/`

## Run order and the artifact contract

Notebook 05 **fits nothing**. It is a pure summarisation notebook that loads what 02/03/04
produce, so it imports numpy, pandas and matplotlib only — no torch, xgboost, statsmodels
or sklearn. Refitting there previously duplicated all six model fits and stalled the
kernel.

That makes 05 the one notebook with a hard prerequisite. Run order:

```
01 → 02 → 03 → 04 → 05      # 05 requires the others; 02 must precede 03/04's comparisons
```

Each modelling notebook ends with a save cell writing:

| File | Written by | Contents |
|---|---|---|
| `results/test_split.npz` | 02 only | shared test-set definition: actuals, exposure, BonusMalus, DrivAge, row indices |
| `results/predictions_{glm,xgb,nn}.npz` | 02/03/04 | `sev_test_pred`, `freq_test_rate`, `sev_on_freq_test` |
| `results/metrics_{glm,xgb,nn}.json` | 02/03/04 | severity `{MAE, RMSE, GammaDeviance}`, frequency `{MAE, RMSE, PoissonDeviance}` |
| `models/*` | 02/03/04 | fitted objects, plus `nn_{sev,freq}_scaler.pkl` and `preprocessing.pkl` |

Two rules that keep this sound:

- **`sev_on_freq_test` is the severity model scored on the *frequency* test rows.** Pure
  premium is `freq_test_rate * sev_on_freq_test`, and that product is only meaningful when
  both halves refer to the same policy. It must be computed inside the modelling notebook,
  where the rows are unambiguous — never by positionally slicing the full matrix in 05,
  since `train_test_split` shuffles and the tail slice is a different set of policies.
- **All splits use `random_state=42`**, frequency and severity alike, so the three models
  are compared on identical rows.

`models/` feeds `dashboard/backend/models_loader.py` as well as reproducibility — if
notebook 01 changes the cleaned CSVs, rerun 02–04 or the dashboard serves predictions from
models trained on stale data.

## Context

This is a portfolio/demonstration project showing the accuracy-interpretability tradeoff in insurance pricing. Regulatory explainability (APRA context) matters — don't optimise for accuracy alone.
