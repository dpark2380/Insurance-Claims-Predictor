# Insurance Claims Predictor — Project Brief

A reference document describing the project's purpose, content, and technical
implementation. Intended as context for downstream prompts that will draft
resume bullets, interview talking points, or a portfolio writeup framed
around **insurance pricing roles** (e.g. Allianz, IAG, Suncorp, QBE,
Munich Re graduate pricing programs).

---

## 1. Why this project exists

### The framing problem in the prior resume

The previous "car pricing" project demonstrated technical capability — scraping,
XGBoost, a React dashboard — but it doesn't speak the language of the team that
would hire for a pricing role. A pricing actuary at Allianz does not care
that you can build an ML pipeline; they care whether you understand:

1. The *purpose* of the model (charge each policyholder a fair, sustainable price)
2. The *trade-offs* between models in a regulated context (accuracy vs interpretability vs defensibility)
3. The *business consequence* of model choices (under-pricing tail risk → solvency issues; over-pricing → adverse selection / lost market share)
4. How to *communicate* model decisions to non-technical stakeholders

This project is built to demonstrate all four. It is structured deliberately as
the same workflow a graduate at a pricing desk would run: load loss data,
benchmark a regulatory-friendly baseline against modern ML, justify a hybrid
production design, and deliver an interactive comparison artifact for
non-technical reviewers.

### The actuarial concept being modelled

The fundamental quantity in insurance pricing is **pure premium**:

```
Pure Premium = E[Frequency] × E[Severity | claim occurred]
              = P(claim)    × E[claim cost | claim]
```

It is the expected annual loss per policy *before* expenses, profit margin, or
capital loading. Every motor insurer in Australia and Europe estimates this
quantity in some form. The choice of model determines:

- How fairly individual policyholders are priced (segmentation power)
- How resilient the book is to adverse selection
- How the model performs at the tails (where solvency capital is set)
- Whether the model can pass APRA / EIOPA regulatory review

This is the bread-and-butter problem the project solves. The dataset
(French motor third-party liability, `freMTPL2freq` + `freMTPL2sev`) is the
canonical academic benchmark for this exact task.

---

## 2. What the project actually contains

A five-notebook analysis, a comparison study, and a production-style interactive
dashboard. Organised top-to-bottom in the order a pricing analyst would work.

### Notebooks (in `notebooks/`)

| # | Notebook | Purpose | Key actuarial concept |
|---|----------|---------|------------------------|
| 01 | `01_eda.ipynb` | Exploratory data analysis on 679,513 policies and 26,439 claims | Loss distribution shape; feature-vs-frequency relationships |
| 02 | `02_glm.ipynb` | Industry-standard baseline: Poisson GLM (frequency) + Gamma GLM (severity), with `log(Exposure)` as offset | Two-stage frequency-severity decomposition; exposure as offset; multiplicative rating factors |
| 03 | `03_xgboost.ipynb` | Gradient boosted trees for both stages — same target, more flexible functional form | Non-linear effects; tree-based monotonicity constraints |
| 04 | `04_neural_net.ipynb` | PyTorch feedforward network — most flexible, least interpretable | Tabular deep learning trade-offs |
| 05 | `05_comparison.ipynb` | Head-to-head comparison + pure premium synthesis + risk discrimination | Gini / Lorenz curves; regulatory interpretability assessment |

### Dashboard (in `dashboard/`)

A FastAPI + React app that lets a non-technical user enter a policy profile
and see all three models' predicted pure premium side-by-side, with per-feature
attribution (SHAP for XGBoost, coefficient × value for GLM, gradient × input
for the neural net). This is the artifact you would demo in an interview —
it shows the end-to-end pipeline and communicates the model trade-off visually.

### Quarto site (`_quarto.yml`, `notebooks.qmd`)

The whole analysis publishes as a static site via Quarto. Demonstrates the
ability to communicate technical work to a broader audience — the skill a
pricing team needs when presenting to underwriting, finance, or APRA.

---

## 3. Key findings — the substance worth talking about

These are the project's *load-bearing* outputs. Every resume bullet or
interview answer should be traceable back to one of these.

### Finding 1: Pure premium estimates diverge substantially across model families

| Model      | Mean pure premium | Frequency Gini | Pure Premium Gini |
|------------|-------------------|----------------|--------------------|
| GLM        | **€220**          | 0.265          | 0.110              |
| XGBoost    | €135              | **0.381**      | **0.179**          |
| Neural Net | €129              | 0.321          | 0.103              |

**Why this matters:** the GLM systematically over-prices the portfolio by
~60% relative to XGBoost. In a competitive market this is the difference
between gaining and losing market share. The GLM's frequency Gini being
0.265 vs XGBoost's 0.381 (a **44% relative improvement** in risk
discrimination) means XGBoost can charge high-risk drivers more and
low-risk drivers less, capturing the same expected loss with better
fairness.

### Finding 2: The GLM extrapolates dangerously in the BonusMalus tail

| BonusMalus band | GLM | XGBoost | Neural Net |
|---|---|---|---|
| 50–60 (clean)    | €170   | €106 | €107 |
| 150+ (worst)     | **€3,317** | €786 | €1,168 |

The GLM's log-link multiplies expected claims by `exp(β·ΔBM)` — for
sparse tail observations this produces premiums 4× higher than XGBoost.
This is not a bug, it's the log-linear assumption colliding with sparse
data. Real pricing teams cap GLM predictions or use credibility blending
for exactly this reason. The finding shows you understand *why* GLMs are
not just "a baseline" but a parametric assumption with consequences.

### Finding 3: Severity is data-limited, not model-limited

All three severity models converge to similar MAE (€938–€1,079). Adding
complexity from GLM → XGBoost → neural net moves the needle by ~2%.
**The ceiling is the data, not the model.** Policy features predict
*whether* a claim happens but not *how much* it costs. A more accurate
severity model is not the optimisation target — better external data
(vehicle valuations, repair cost indices) would be.

### Finding 4: Hybrid design wins — and the neural net loses

The recommended production design is **XGBoost for frequency, GLM for
severity** — best of both. The neural net trails XGBoost on every
frequency metric, ties on severity, and has the *worst* end-to-end pure
premium Gini (0.103 — behind even the GLM at 0.110). On structured
tabular insurance data, gradient boosted trees beat neural networks
consistently. Knowing when *not* to reach for deep learning is a senior
judgment call.

### Finding 5: Regulatory defensibility shapes the choice

Under APRA Prudential Standard GPS 116, premium rating factors must be
documented and defensible. The hybrid is regulator-friendly:

- **GLM severity** — standard actuarial practice, coefficients audit-ready
- **XGBoost frequency** — SHAP for individual explanations, monotonicity constraints on BonusMalus to guarantee worse driving never lowers premium
- **Neural net** — would need significantly more explanation tooling; not worth the cost given it does not outperform XGBoost

---

## 4. Technical implementation details

For prompts that need accurate details about *how* things were built.

### Data

- **Dataset:** French Motor Third-Party Liability (`freMTPL2freq`, `freMTPL2sev`) via OpenML
- **Size:** 679,513 policies, 26,439 claims, 3.9% claim rate
- **Features:** VehPower (4–15), VehAge (0–100), DrivAge (18–100), BonusMalus (50–230), Density (1–27,000), Area (A–F), VehBrand (B1–B14), VehGas (Regular/Diesel), Region (R11–R94, 22 levels)
- **Targets:** `ClaimNb` for frequency (Poisson); `ClaimAmount` for severity (Gamma), capped at the 99th percentile (€16,451)
- **Encoding:** One-hot with `drop_first=True`, severity matrix reindexed to the frequency schema so a single input vector drives both stages
- **Split:** 80/20 train/test, `random_state=42`, same split used by all six models for direct comparability

### Models

| Stage / target | GLM | XGBoost | Neural Net |
|---|---|---|---|
| **Frequency** (`ClaimNb`, Poisson) | `statsmodels.GLM`, Poisson family, log link, `log(Exposure)` offset | `XGBRegressor(objective='count:poisson')`, 500 trees, depth 4, sample-weighted by exposure, early stopping | PyTorch `Linear(42→64→32→1)` + ReLU + Dropout(0.2); 20 epochs, batch 4096; trained on `ClaimNb/Exposure` with exposure-weighted MSE |
| **Severity** (`ClaimAmount`, Gamma) | `statsmodels.GLM`, Gamma family, log link | `XGBRegressor(objective='reg:squarederror')`, 500 trees, depth 4, trained on `log(ClaimAmount)`, early stopping | Same architecture as frequency net, trained on `log1p(ClaimAmount)`, 30 epochs batch 256 |

### Metrics

- **Frequency:** MAE, RMSE, Poisson Deviance (actuarially correct distributional metric)
- **Severity:** MAE, RMSE, Gamma Deviance (calibration to the Gamma family)
- **End-to-end:** Mean pure premium, pure premium distribution, Gini coefficient (frequency Gini and pure premium Gini), Lorenz curves, decile lift charts
- **Risk segmentation:** Mean pure premium by BonusMalus band and by driver age band

### Dashboard architecture

```
scripts/
├── 01_train_glm_xgb.py    GLM + XGBoost (~8s)
├── 02_train_nn.py         Neural net (~5s)
├── 03_compute_stats.py    Portfolio stats (~1s)
└── train_all.sh           Wrapper running all three sequentially

dashboard/backend/         FastAPI on :8000
├── main.py                Three routes: /api/health, /api/schema, /api/predict
├── models_loader.py       Loads pickled artifacts at startup; runs inference + SHAP
└── schemas.py             Pydantic request/response models

dashboard/frontend/        React + Vite on :5173
└── src/components/        PolicyForm, PremiumCards, BandContext, AttributionGrid

models/                    All artifacts loaded once at backend startup
├── glm_freq.pkl, glm_sev.pkl       (statsmodels coefficient arrays)
├── xgb_freq.json, xgb_sev.json     (native XGBoost format)
├── nn_freq.pt, nn_sev.pt           (torch state dicts + scaler params)
├── preprocessing.pkl                (feature schema, vocab, ranges)
├── portfolio_stats.json             (mean PP per model, band means)
└── shap_background.npy              (100-row reference for NN attribution)
```

**Attribution methods** differ per model and are all reported in the UI:
- **GLM:** `β_i × x_i` per feature — exact, analytic
- **XGBoost:** TreeSHAP via `shap.TreeExplainer` — exact for tree ensembles
- **Neural net:** `gradient × (input − background mean)` — a SHAP-style approximation

**Note on the macOS quirk:** training is split into three sequential Python
processes because importing `torch` after `xgboost` in the same interpreter
causes an OpenMP runtime deadlock on Apple Silicon. Each subprocess gets a
fresh interpreter. The serving backend imports both safely because it only
does single-sample inference.

### Stack

| Layer | Tools |
|---|---|
| Modelling | `statsmodels` (GLM), `xgboost`, `torch`, `scikit-learn` |
| Explainability | `shap` (TreeExplainer for XGBoost) |
| Data | `pandas`, `numpy` |
| API | `fastapi`, `uvicorn`, `pydantic` v2 |
| Frontend | React 18, Vite, Recharts |
| Publishing | Quarto (renders the notebooks as a static site) |
| Environment | Python 3.14, Node 18+, macOS / Linux |

---

## 5. Skills demonstrated — mapped to a pricing-role JD

For mapping bullets to specific job requirements.

| Required skill (typical pricing JD) | How this project demonstrates it |
|---|---|
| GLM modelling for insurance pricing | Poisson frequency + Gamma severity GLMs with `log(Exposure)` offset, the textbook actuarial setup |
| Understanding of frequency-severity decomposition | The entire project is structured around it; comparison notebook synthesises both stages into pure premium |
| Familiarity with ML methods used by modern pricing teams (GBM, neural nets) | XGBoost and PyTorch implementations alongside the GLM baseline, with calibration and Gini comparison |
| Model selection in a regulated context | Notebook 05 part 4: explicit accuracy vs interpretability radar chart, APRA GPS 116 framing in the conclusion |
| Communication of technical results to non-technical audiences | Interactive dashboard with cards, band context, and per-feature attribution; Quarto publishing for the writeup |
| Python programming | All notebooks, training scripts, FastAPI backend |
| Data manipulation (pandas/SQL) | EDA notebook; one-hot encoding pipeline that aligns severity to frequency schema |
| Risk segmentation analysis | BonusMalus band table, driver age band table, Gini coefficients |
| Knowledge of common metrics (deviance, Gini, lift) | All used in notebook 05, reported in the dashboard |
| Working with industry-standard datasets | French MTPL is the canonical benchmark in academic actuarial literature |

---

## 6. Resume-bullet starter ideas

Rough drafts to be polished by a downstream prompt. Each bullet should
lead with **business impact** and back-fill with technique.

- *Benchmarked Poisson/Gamma GLM, XGBoost, and PyTorch neural network on French MTPL pure premium estimation — found XGBoost delivered a 44% relative improvement in risk-discrimination Gini (0.381 vs 0.265) over the regulatory baseline.*
- *Quantified the cost of GLM tail extrapolation, showing the log-link produced predictions 4× higher than XGBoost in the BonusMalus 150+ band — a finding directly informing the recommended hybrid pricing design.*
- *Designed a hybrid two-stage pricer (XGBoost frequency + GLM severity) framed against APRA Prudential Standard GPS 116, balancing predictive lift against regulatory interpretability.*
- *Demonstrated that on structured tabular insurance data, neural networks under-perform gradient boosted trees on every frequency metric (Poisson Deviance 44,368 vs 43,579) — a pragmatic finding against the "deeper is better" default.*
- *Built a FastAPI + React dashboard letting non-technical reviewers enter a policy profile and compare all three models' pure premium estimates side-by-side, with per-feature SHAP attribution.*
- *Identified severity as data-limited rather than model-limited (all three models within 2% on MAE), redirecting effort toward feature engineering on external vehicle/repair-cost data.*

---

## 7. Why this replaces the car-pricing project

| | Car pricing project | This project |
|---|---|---|
| Domain relevance to insurance roles | Adjacent — uses similar ML tooling but not actuarial | Direct — pure premium estimation is the central task of P&C pricing |
| Demonstrates regulatory awareness | No | Yes — APRA / GPS 116 framing, interpretability radar, defensibility analysis |
| Uses canonical actuarial dataset | No | Yes — French MTPL is the academic benchmark |
| Two-stage frequency-severity decomposition | No (single regression) | Yes — entire project is built on it |
| Distributional metrics (deviance, Gini, lift) | RMSE/MAE only | Poisson/Gamma deviance, Gini, Lorenz, decile lift |
| Risk-segmentation analysis | No | BonusMalus and DrivAge band breakdowns |
| Interpretability story | Implicit | Explicit — SHAP for XGBoost, coefficients for GLM, gradient × input for NN, side-by-side in the dashboard |
| Business framing | Consumer use case | Insurer use case (the same work a pricing analyst does day-to-day) |

The car-pricing project is well-built but reads as a "data science project."
This project reads as an "actuarial pricing analysis." For a pricing graduate
role at Allianz, the difference is exactly the framing the resume is
currently missing.

---

## 8. Tone notes for downstream prompts

When writing resume bullets or interview answers off this brief:

- Lead with the *finding* or *business consequence*, not the *technique*.
  - Bad: "Built a Poisson GLM with exposure offset."
  - Good: "Established a regulatory-baseline frequency model, then quantified
    its 44%-relative-Gini gap to XGBoost — supporting a hybrid production design."
- Use *actuarial language* where appropriate: pure premium, exposure, risk
  discrimination, segmentation power, distributional fit, calibration,
  tail behaviour, credibility, loss ratio.
- Reference *real numbers* from this project where possible — they signal
  that the work was actually done.
- Tie technical choices to *regulatory* and *commercial* consequences (APRA
  GPS 116, adverse selection, capital adequacy, market share).
- Avoid resume clichés like "leveraged" / "utilised" — pricing teams write
  technical memos, they don't write LinkedIn posts.
