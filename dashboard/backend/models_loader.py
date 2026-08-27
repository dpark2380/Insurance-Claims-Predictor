"""Loads trained models from models/ and provides a single `predict()` entrypoint.

Each model returns (frequency_rate, severity_eur, pure_premium, attributions).
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import json, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import shap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')


class ClaimNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)


class Bundle:
    """All models + explainers + metadata, loaded once at startup."""

    def __init__(self):
        with open(os.path.join(MODELS_DIR, 'preprocessing.pkl'), 'rb') as f:
            self.preprocessing = pickle.load(f)
        self.feature_columns = self.preprocessing['columns']
        self.categorical_features = self.preprocessing['cat_cols']
        self.numeric_features = [c for c in self.preprocessing['features']
                                  if c not in self.categorical_features]

        with open(os.path.join(MODELS_DIR, 'portfolio_stats.json')) as f:
            self.portfolio_stats = json.load(f)

        # Plain coefficient arrays: [const, then one per column in self.feature_columns].
        with open(os.path.join(MODELS_DIR, 'glm_freq.pkl'), 'rb') as f:
            self.glm_freq_params = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'glm_sev.pkl'), 'rb') as f:
            self.glm_sev_params = pickle.load(f)

        self.xgb_freq = xgb.XGBRegressor()
        self.xgb_freq.load_model(os.path.join(MODELS_DIR, 'xgb_freq.json'))
        self.xgb_sev = xgb.XGBRegressor()
        self.xgb_sev.load_model(os.path.join(MODELS_DIR, 'xgb_sev.json'))

        self.nn_freq, self.nn_freq_mean, self.nn_freq_scale = self._load_nn('nn_freq.pt', 'nn_freq_scaler.pkl')
        self.nn_sev,  self.nn_sev_mean,  self.nn_sev_scale  = self._load_nn('nn_sev.pt', 'nn_sev_scaler.pkl')

        # SHAP explainers
        self.bg = np.load(os.path.join(MODELS_DIR, 'shap_background.npy'))
        self.xgb_freq_explainer = shap.TreeExplainer(self.xgb_freq)
        self.xgb_sev_explainer  = shap.TreeExplainer(self.xgb_sev)

    def _load_nn(self, fname, scaler_fname):
        # Saved as a raw state_dict (torch.save(model.state_dict(), ...)); the fitted
        # StandardScaler lives separately since a network alone can't rescale new inputs.
        state_dict = torch.load(os.path.join(MODELS_DIR, fname), weights_only=False)
        input_dim = state_dict['net.0.weight'].shape[1]
        net = ClaimNet(input_dim)
        net.load_state_dict(state_dict)
        net.eval()
        with open(os.path.join(MODELS_DIR, scaler_fname), 'rb') as f:
            scaler = pickle.load(f)
        return net, np.asarray(scaler.mean_, dtype=np.float32), np.asarray(scaler.scale_, dtype=np.float32)

    # ── Encoding ──────────────────────────────────────────────────────────────
    def encode(self, policy: dict) -> np.ndarray:
        """Convert a policy dict to the one-hot feature vector used at training."""
        row = {col: 0.0 for col in self.feature_columns}
        for nf in self.numeric_features:
            row[nf] = float(policy[nf])
        for cf in self.categorical_features:
            value = policy[cf]
            # The training data has VehGas values like "'Regular'" with quotes — handle both
            for candidate in [value, f"'{value}'"]:
                col = f"{cf}_{candidate}"
                if col in row:
                    row[col] = 1.0
                    break
        return np.array([row[c] for c in self.feature_columns], dtype=np.float32)

    # ── GLM ───────────────────────────────────────────────────────────────────
    def _glm_predict(self, x, params, offset=0.0):
        """log-link Poisson/Gamma: μ = exp(β0 + βᵀx + offset)."""
        x_aug = np.concatenate([[1.0], x])  # add constant
        linear = float(x_aug @ params) + offset
        return float(np.exp(linear))

    def _glm_attributions(self, x, params, top_n=8):
        """Per-feature contribution to the linear predictor (log scale).

        Returns list of (feature, contribution_in_eur_units, raw_value).
        Contribution is the additive effect on log(μ); we convert to a
        multiplicative factor minus 1 for interpretability.
        """
        contribs = []
        # params[0] is intercept; skip it
        for i, col in enumerate(self.feature_columns):
            beta = float(params[i + 1])
            val = float(x[i])
            if val == 0.0 and col.split('_')[0] in self.categorical_features:
                continue  # one-hot column off
            contrib = beta * val
            contribs.append((col, contrib, val))
        contribs.sort(key=lambda t: abs(t[1]), reverse=True)
        return contribs[:top_n]

    def predict_glm(self, x: np.ndarray, exposure: float):
        freq_rate  = self._glm_predict(x, self.glm_freq_params, offset=0.0)
        sev_amount = self._glm_predict(x, self.glm_sev_params,  offset=0.0)
        pp = freq_rate * sev_amount
        # Combine attributions: use the frequency model's attributions as the dominant signal
        freq_attr = self._glm_attributions(x, self.glm_freq_params)
        return {
            'frequency_rate': freq_rate,
            'severity_eur': sev_amount,
            'pure_premium_eur': pp,
            'attributions': freq_attr,
        }

    # ── XGBoost ───────────────────────────────────────────────────────────────
    def predict_xgb(self, x: np.ndarray, exposure: float):
        X = x.reshape(1, -1)
        freq_rate  = float(self.xgb_freq.predict(X)[0])
        sev_amount = float(np.exp(self.xgb_sev.predict(X)[0]))
        pp = freq_rate * sev_amount
        shap_vals = self.xgb_freq_explainer.shap_values(X)[0]
        contribs = sorted(
            [(self.feature_columns[i], float(shap_vals[i]), float(x[i]))
             for i in range(len(self.feature_columns))
             if not (x[i] == 0.0 and self.feature_columns[i].split('_')[0] in self.categorical_features)],
            key=lambda t: abs(t[1]), reverse=True
        )[:8]
        return {
            'frequency_rate': freq_rate,
            'severity_eur': sev_amount,
            'pure_premium_eur': pp,
            'attributions': contribs,
        }

    # ── Neural Net ────────────────────────────────────────────────────────────
    def _nn_attributions(self, net, x, mean, scale, top_n=8):
        """Gradient × (input - background) — a simple SHAP-like attribution."""
        x_scaled = (x - mean) / scale
        x_t = torch.from_numpy(x_scaled.astype(np.float32)).unsqueeze(0)
        x_t.requires_grad_(True)
        net.eval()
        pred = net(x_t)
        pred.backward()
        grad = x_t.grad[0].detach().numpy()
        # Reference: average of background sample (already in raw feature space)
        bg_mean = self.bg.mean(axis=0)
        bg_scaled = (bg_mean - mean) / scale
        contrib = grad * (x_scaled - bg_scaled)
        contribs = sorted(
            [(self.feature_columns[i], float(contrib[i]), float(x[i]))
             for i in range(len(self.feature_columns))
             if not (x[i] == 0.0 and self.feature_columns[i].split('_')[0] in self.categorical_features)],
            key=lambda t: abs(t[1]), reverse=True
        )[:top_n]
        return contribs

    def predict_nn(self, x: np.ndarray, exposure: float):
        x_freq_scaled = (x - self.nn_freq_mean) / self.nn_freq_scale
        x_sev_scaled  = (x - self.nn_sev_mean)  / self.nn_sev_scale
        with torch.no_grad():
            freq_rate = max(0.0, float(self.nn_freq(
                torch.from_numpy(x_freq_scaled.astype(np.float32)).unsqueeze(0)).item()))
            sev_amount = float(np.expm1(self.nn_sev(
                torch.from_numpy(x_sev_scaled.astype(np.float32)).unsqueeze(0)).item()))
        pp = freq_rate * sev_amount
        contribs = self._nn_attributions(self.nn_freq, x, self.nn_freq_mean, self.nn_freq_scale)
        return {
            'frequency_rate': freq_rate,
            'severity_eur': sev_amount,
            'pure_premium_eur': pp,
            'attributions': contribs,
        }

    # ── Unified entrypoint ────────────────────────────────────────────────────
    def predict_all(self, policy: dict) -> list[dict]:
        x = self.encode(policy)
        exposure = float(policy.get('Exposure', 1.0))
        results = []
        for name, fn in [('GLM', self.predict_glm), ('XGBoost', self.predict_xgb), ('Neural Net', self.predict_nn)]:
            r = fn(x, exposure)
            portfolio_mean = self.portfolio_stats['mean_pure_premium'][name]
            r['model'] = name
            r['portfolio_mean_pure_premium'] = portfolio_mean
            r['delta_vs_portfolio'] = r['pure_premium_eur'] - portfolio_mean
            r['attributions'] = [
                {'feature': self._pretty_feature(f), 'contribution': c, 'value': self._pretty_value(f, v)}
                for f, c, v in r['attributions']
            ]
            results.append(r)
        return results

    def _pretty_feature(self, col: str) -> str:
        for cf in self.categorical_features:
            if col.startswith(cf + '_'):
                return f"{cf} = {col[len(cf)+1:].strip(chr(39))}"
        return col

    def _pretty_value(self, col: str, v: float):
        if col in self.numeric_features:
            return v
        return 1 if v else 0
