"""Computes portfolio stats (mean pure premium per model, band means) using
   the already-trained models on the held-out test slice."""
import os, pickle, json, warnings, time
import numpy as np
import pandas as pd
import xgboost as xgb
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
t0 = time.time()


def log(msg): print(f"[{time.time()-t0:5.1f}s] {msg}", flush=True)


log("Loading test slice + models + NN preds...")
td = np.load(os.path.join(MODELS_DIR, '_test_data.npz'))
X_te_all, exp_te_all, bm_te, age_te = td['X_te_all'], td['exp_te_all'], td['bm_te'], td['age_te']

with open(os.path.join(MODELS_DIR, 'glm_freq.pkl'), 'rb') as f:
    glm_freq_params = pickle.load(f)['params']
with open(os.path.join(MODELS_DIR, 'glm_sev.pkl'), 'rb') as f:
    glm_sev_params = pickle.load(f)['params']

xgb_freq = xgb.XGBRegressor(); xgb_freq.load_model(os.path.join(MODELS_DIR, 'xgb_freq.json'))
xgb_sev  = xgb.XGBRegressor(); xgb_sev.load_model(os.path.join(MODELS_DIR, 'xgb_sev.json'))

nn_preds = np.load(os.path.join(MODELS_DIR, '_nn_test_preds.npz'))
nn_freq_rate, nn_sev_pred = nn_preds['nn_freq_rate'], nn_preds['nn_sev_pred']

log("Generating predictions...")
# GLM (manual since we only saved the params)
X_const = np.column_stack([np.ones(X_te_all.shape[0], dtype=np.float64), X_te_all])
glm_freq_count = np.exp(X_const @ glm_freq_params + np.log(exp_te_all))
glm_freq_rate  = glm_freq_count / exp_te_all
glm_sev_pred   = np.exp(X_const @ glm_sev_params)
glm_pp         = glm_freq_rate * glm_sev_pred

xgb_freq_rate = xgb_freq.predict(X_te_all)
xgb_sev_pred  = np.exp(xgb_sev.predict(X_te_all))
xgb_pp        = xgb_freq_rate * xgb_sev_pred

nn_pp = nn_freq_rate * nn_sev_pred

# ── Bands ─────────────────────────────────────────────────────────────────────
bm_bins  = pd.cut(bm_te,  bins=[49,60,80,100,150,230], labels=['50-60','61-80','81-100','101-150','150+'])
age_bins = pd.cut(age_te, bins=[17,25,35,45,55,65,100], labels=['18-25','26-35','36-45','46-55','56-65','65+'])

def band_means(pp, bins, labels):
    out = {}
    for lab in labels:
        m = np.asarray(bins == lab)
        if m.sum(): out[lab] = float(np.round(pp[m].mean(), 2))
    return out

stats = {
    'mean_pure_premium': {
        'GLM': float(np.round(glm_pp.mean(), 2)),
        'XGBoost': float(np.round(xgb_pp.mean(), 2)),
        'Neural Net': float(np.round(nn_pp.mean(), 2)),
    },
    'mean_frequency_rate': {
        'GLM': float(np.round(glm_freq_rate.mean(), 5)),
        'XGBoost': float(np.round(xgb_freq_rate.mean(), 5)),
        'Neural Net': float(np.round(nn_freq_rate.mean(), 5)),
    },
    'mean_severity': {
        'GLM': float(np.round(glm_sev_pred.mean(), 2)),
        'XGBoost': float(np.round(xgb_sev_pred.mean(), 2)),
        'Neural Net': float(np.round(nn_sev_pred.mean(), 2)),
    },
    'pp_by_bonus_malus_band': {
        'bands': ['50-60','61-80','81-100','101-150','150+'],
        'GLM':        band_means(glm_pp, bm_bins, ['50-60','61-80','81-100','101-150','150+']),
        'XGBoost':    band_means(xgb_pp, bm_bins, ['50-60','61-80','81-100','101-150','150+']),
        'Neural Net': band_means(nn_pp,  bm_bins, ['50-60','61-80','81-100','101-150','150+']),
    },
    'pp_by_age_band': {
        'bands': ['18-25','26-35','36-45','46-55','56-65','65+'],
        'GLM':        band_means(glm_pp, age_bins, ['18-25','26-35','36-45','46-55','56-65','65+']),
        'XGBoost':    band_means(xgb_pp, age_bins, ['18-25','26-35','36-45','46-55','56-65','65+']),
        'Neural Net': band_means(nn_pp,  age_bins, ['18-25','26-35','36-45','46-55','56-65','65+']),
    },
}

with open(os.path.join(MODELS_DIR, 'portfolio_stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)

# Clean up intermediate files
for tmp in ['_train_data.npz', '_test_data.npz', '_nn_test_preds.npz']:
    p = os.path.join(MODELS_DIR, tmp)
    if os.path.exists(p): os.remove(p)

log("Step 3 done. All artifacts in models/.")
