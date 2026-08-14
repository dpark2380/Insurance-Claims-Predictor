# Insurance Pure Premium Dashboard

Interactive comparison of the GLM, XGBoost, and Neural Net models from
`notebooks/05_comparison.ipynb`. Enter a policy profile, see all three
models' pure premium predictions side-by-side with per-feature attribution.

## Architecture

```
dashboard/
├── backend/         FastAPI service (port 8000)
│   ├── main.py             — API routes
│   ├── models_loader.py    — loads pickled models, runs inference + SHAP
│   └── schemas.py          — Pydantic request/response models
└── frontend/        React + Vite (port 5173)
    └── src/components/     — PolicyForm, PremiumCards, BandContext, AttributionGrid
```

Models live in `../models/` (six trained models + portfolio stats + SHAP
background sample). They are trained once by `scripts/train_all.sh` and
loaded into memory at FastAPI startup.

## One-time setup

From the project root:

```bash
# 1. Create the venv and install Python dependencies (see ../requirements.txt)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Train all six models (~15 seconds end-to-end)
bash scripts/train_all.sh

# 3. Install frontend deps
cd dashboard/frontend && npm install && cd -
```

## Running

You need two terminals — one for the backend, one for the frontend.

```bash
# Terminal 1 — backend
cd dashboard/backend
source ../../venv/bin/activate
uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd dashboard/frontend
npm run dev
```

Open <http://localhost:5173>.

## API

- `GET  /api/health`  — readiness check
- `GET  /api/schema`  — categorical vocab, numeric ranges, band stats
- `POST /api/predict` — `Policy` in, all-three-model `PredictionResponse` out

Example:

```bash
curl -s -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"VehPower":6,"VehAge":5,"DrivAge":22,"BonusMalus":120,"Density":2000,"Exposure":1.0,"Area":"C","VehBrand":"B1","VehGas":"Regular","Region":"R24"}'
```

## Attribution methods

| Model       | Method                                       |
|-------------|----------------------------------------------|
| GLM         | Coefficient × value (exact, analytic)        |
| XGBoost     | TreeSHAP (`shap.TreeExplainer`, exact)       |
| Neural Net  | Gradient × (input − background mean)         |

All three are reported on the frequency model's predictions, since that
is where the three models diverge most meaningfully (severity is data-limited).

## Note on macOS / OpenMP

Training is split into three sequential Python processes
(`01_train_glm_xgb.py`, `02_train_nn.py`, `03_compute_stats.py`) to avoid
a deadlock between XGBoost's and PyTorch's OpenMP runtimes when imported
in the same interpreter. Each runs in a fresh process, total cost ~15s.

The serving backend imports both safely because PyTorch only does
single-sample inference (no parallel workers).
