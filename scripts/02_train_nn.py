"""Fits NN frequency/severity. No xgboost or statsmodels in this process."""
import os, warnings, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
t0 = time.time()


def log(msg): print(f"[{time.time()-t0:5.1f}s] {msg}", flush=True)


class ClaimNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)


def train_nn(X_tr, y_tr, epochs, batch_size, weights=None):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_tr).astype(np.float32)
    X_t = torch.from_numpy(Xs)
    y_t = torch.from_numpy(np.asarray(y_tr, dtype=np.float32)).unsqueeze(1)
    w_t = torch.from_numpy(np.asarray(weights, dtype=np.float32)).unsqueeze(1) if weights is not None else None
    net = ClaimNet(Xs.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = X_t.shape[0]
    for epoch in range(epochs):
        net.train()
        idx = torch.randperm(n)
        for s in range(0, n, batch_size):
            b = idx[s:s+batch_size]
            opt.zero_grad()
            pred = net(X_t[b])
            if w_t is not None:
                loss = (w_t[b] * (pred - y_t[b]) ** 2).mean()
            else:
                loss = ((pred - y_t[b]) ** 2).mean()
            loss.backward()
            opt.step()
    return net, scaler


log("Loading train slices...")
d = np.load(os.path.join(MODELS_DIR, '_train_data.npz'))
X_sev_tr, y_sev_tr = d['X_sev_tr'], d['y_sev_tr']
X_fr_tr, y_rate_tr, exp_tr = d['X_fr_tr'], d['y_rate_tr'], d['exp_tr']

log(f"Training NN severity ({X_sev_tr.shape[0]} rows, 30 epochs)...")
nn_sev, sev_scaler = train_nn(X_sev_tr, np.log1p(y_sev_tr), epochs=30, batch_size=256)
nn_sev.eval()

log(f"Training NN frequency ({X_fr_tr.shape[0]} rows, 20 epochs)...")
nn_freq, freq_scaler = train_nn(X_fr_tr, y_rate_tr, epochs=20, batch_size=4096,
                                weights=exp_tr.astype(np.float32))
nn_freq.eval()

torch.save({'state_dict': nn_sev.state_dict(),  'input_dim': X_sev_tr.shape[1],
            'mean':  sev_scaler.mean_.astype(np.float32).tolist(),
            'scale': sev_scaler.scale_.astype(np.float32).tolist()},
           os.path.join(MODELS_DIR, 'nn_sev.pt'))
torch.save({'state_dict': nn_freq.state_dict(), 'input_dim': X_fr_tr.shape[1],
            'mean':  freq_scaler.mean_.astype(np.float32).tolist(),
            'scale': freq_scaler.scale_.astype(np.float32).tolist()},
           os.path.join(MODELS_DIR, 'nn_freq.pt'))

# ── NN predictions on the held-out test slice (saved for step 3) ──────────────
log("Generating NN test-slice predictions...")
td = np.load(os.path.join(MODELS_DIR, '_test_data.npz'))
X_te_all = td['X_te_all']
with torch.no_grad():
    nn_freq_rate = np.clip(
        nn_freq(torch.from_numpy(((X_te_all - freq_scaler.mean_) / freq_scaler.scale_).astype(np.float32))).numpy().flatten(), 0, None)
    nn_sev_pred = np.expm1(
        nn_sev(torch.from_numpy(((X_te_all - sev_scaler.mean_) / sev_scaler.scale_).astype(np.float32))).numpy().flatten())
np.savez(os.path.join(MODELS_DIR, '_nn_test_preds.npz'),
         nn_freq_rate=nn_freq_rate, nn_sev_pred=nn_sev_pred)

log("Step 2 done.")
