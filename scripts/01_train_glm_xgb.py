"""Fits GLM + XGBoost frequency/severity. Saves models + the test slice we need
   for the NN script and stats script."""
import os, pickle, warnings, time
import numpy as np
import pandas as pd
import xgboost as xgb
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
np.random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
t0 = time.time()


def log(msg): print(f"[{time.time()-t0:5.1f}s] {msg}", flush=True)


log("Loading data...")
df_policies = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/policies_cleaned.csv'))
df_claims   = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/claims_cleaned.csv'))

features = ['VehPower','VehAge','DrivAge','BonusMalus','Density','Area','VehBrand','VehGas','Region']
cat_cols = ['Area','VehBrand','VehGas','Region']

df_sev_enc = pd.get_dummies(df_claims[features + ['ClaimAmount']], columns=cat_cols, drop_first=True)
df_freq_enc = pd.get_dummies(df_policies[features + ['ClaimNb']], columns=cat_cols, drop_first=True)

freq_cols = [c for c in df_freq_enc.columns if c != 'ClaimNb']
df_sev_aligned = df_sev_enc.drop(columns=['ClaimAmount']).reindex(columns=freq_cols, fill_value=0)

X_sev = np.ascontiguousarray(df_sev_aligned.values.astype(np.float32))
y_sev = df_sev_enc['ClaimAmount'].values.astype(np.float32)
X_freq = np.ascontiguousarray(df_freq_enc[freq_cols].values.astype(np.float32))
y_freq = df_freq_enc['ClaimNb'].values.astype(np.float32)
exposure = df_policies['Exposure'].values

X_sev_tr, X_sev_te, y_sev_tr, y_sev_te = train_test_split(X_sev, y_sev, test_size=0.2, random_state=42)
X_fr_tr, X_fr_te, y_fr_tr, y_fr_te, exp_tr, exp_te = train_test_split(
    X_freq, y_freq, exposure, test_size=0.2, random_state=42)

y_rate_tr = y_fr_tr / exp_tr
y_rate_te = y_fr_te / exp_te

log(f"  features: {len(freq_cols)}")

# ── Vocab / ranges for preprocessing artifact ─────────────────────────────────
vocab = {
    'Area':     sorted(df_policies['Area'].unique().tolist()),
    'VehBrand': sorted(df_policies['VehBrand'].unique().tolist()),
    'VehGas':   sorted(df_policies['VehGas'].unique().tolist()),
    'Region':   sorted(df_policies['Region'].unique().tolist()),
}
display_vocab = {
    'Area':     vocab['Area'],
    'VehBrand': vocab['VehBrand'],
    'VehGas':   [v.strip("'") for v in vocab['VehGas']],
    'Region':   vocab['Region'],
}
numeric_ranges = {
    'VehPower':   [int(df_policies['VehPower'].min()),   int(df_policies['VehPower'].max())],
    'VehAge':     [int(df_policies['VehAge'].min()),     int(df_policies['VehAge'].max())],
    'DrivAge':    [int(df_policies['DrivAge'].min()),    int(df_policies['DrivAge'].max())],
    'BonusMalus': [int(df_policies['BonusMalus'].min()), int(df_policies['BonusMalus'].max())],
    'Density':    [int(df_policies['Density'].min()),    int(df_policies['Density'].max())],
}

preprocessing = {
    'feature_columns': freq_cols,
    'numeric_features': ['VehPower','VehAge','DrivAge','BonusMalus','Density'],
    'categorical_features': cat_cols,
    'vocab': vocab,
    'display_vocab': display_vocab,
    'numeric_ranges': numeric_ranges,
}
with open(os.path.join(MODELS_DIR, 'preprocessing.pkl'), 'wb') as f:
    pickle.dump(preprocessing, f)


# ── GLM ───────────────────────────────────────────────────────────────────────
log("Fitting GLM severity (Gamma)...")
glm_sev = sm.GLM(y_sev_tr, sm.add_constant(X_sev_tr),
                 family=sm.families.Gamma(link=sm.families.links.Log())).fit()
log("Fitting GLM frequency (Poisson)...")
glm_freq = sm.GLM(y_fr_tr, sm.add_constant(X_fr_tr),
                  family=sm.families.Poisson(), offset=np.log(exp_tr)).fit()

with open(os.path.join(MODELS_DIR, 'glm_sev.pkl'), 'wb') as f:
    pickle.dump({'params': np.asarray(glm_sev.params)}, f)
with open(os.path.join(MODELS_DIR, 'glm_freq.pkl'), 'wb') as f:
    pickle.dump({'params': np.asarray(glm_freq.params)}, f)


# ── XGBoost ───────────────────────────────────────────────────────────────────
log("Fitting XGBoost severity...")
xgb_sev = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05,
    max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42,
    early_stopping_rounds=20, eval_metric='rmse')
xgb_sev.fit(X_sev_tr, np.log(y_sev_tr), eval_set=[(X_sev_te, np.log(y_sev_te))], verbose=False)
xgb_sev.save_model(os.path.join(MODELS_DIR, 'xgb_sev.json'))

log("Fitting XGBoost frequency...")
xgb_freq = xgb.XGBRegressor(objective='count:poisson', n_estimators=500, learning_rate=0.05,
    max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42,
    early_stopping_rounds=20, eval_metric='poisson-nloglik')
xgb_freq.fit(X_fr_tr, y_rate_tr, sample_weight=exp_tr,
             eval_set=[(X_fr_te, y_rate_te)], verbose=False)
xgb_freq.save_model(os.path.join(MODELS_DIR, 'xgb_freq.json'))


# ── Test slice + train slice for downstream scripts ───────────────────────────
np.random.seed(0)
bg_idx = np.random.choice(len(X_fr_tr), size=100, replace=False)
np.save(os.path.join(MODELS_DIR, 'shap_background.npy'), X_fr_tr[bg_idx])

# Save train slices for NN step
np.savez(os.path.join(MODELS_DIR, '_train_data.npz'),
         X_sev_tr=X_sev_tr, y_sev_tr=y_sev_tr,
         X_fr_tr=X_fr_tr, y_rate_tr=y_rate_tr, exp_tr=exp_tr)

# Save test slice for stats step
n_train = len(X_fr_tr)
np.savez(os.path.join(MODELS_DIR, '_test_data.npz'),
         X_te_all=X_freq[n_train:],
         exp_te_all=exposure[n_train:],
         bm_te=df_policies['BonusMalus'].values[n_train:],
         age_te=df_policies['DrivAge'].values[n_train:])

log("Step 1 done.")
