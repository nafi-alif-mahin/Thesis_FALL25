import numpy as np
import random
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score

# ── CONFIG ───────────────────────────────────────────────────────────────
DATA_DIR = "../data"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)
BEST_PARAMS_FILE = os.path.join(MODEL_DIR, "best_adtcn_params.json")
BEST_MODEL_FILE  = os.path.join(MODEL_DIR, "best_adtcn.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tuning settings (keep small for testing)
POP_SIZE       = 6
MAX_ITER       = 8
EPOCHS_PER_CANDIDATE = 4   # short training per individual
BATCH_SIZE     = 64

# Search space
PARAM_RANGES = {
    'hidden_dim':   (32, 128),
    'lstm_layers':  (1, 3),
    'dropout':      (0.05, 0.45),
    'lr':           (5e-4, 5e-3)
}

PARAM_KEYS = list(PARAM_RANGES.keys())

# ── LOAD DATA (once) ─────────────────────────────────────────────────────
def load_datasets():
    def safe_load(fn):
        p = os.path.join(DATA_DIR, fn)
        try:
            return np.load(p, allow_pickle=False).astype(np.float32)
        except:
            arr = np.load(p, allow_pickle=True)
            if arr.dtype == object:
                arr = np.asarray(arr.tolist(), dtype=np.float32)
            return arr.astype(np.float32)

    X_tr = safe_load("train_X.npy")
    y_tr = safe_load("train_y.npy")
    X_va = safe_load("val_X.npy")
    y_va = safe_load("val_y.npy")

    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    va_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
    return tr_ds, va_ds

train_ds, val_ds = load_datasets()

# ── ADTCN MODEL ──────────────────────────────────────────────────────────
class ADTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(hidden_dim*2, 1)
        self.fc   = nn.Linear(hidden_dim*2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        w = torch.softmax(self.attn(out), dim=1)
        ctx = torch.sum(out * w, dim=1)
        return self.fc(ctx).squeeze(-1)   # logits

# ── FITNESS = train short & evaluate MCC-heavy ───────────────────────────
def evaluate_candidate(params):
    hidden = int(round(params[0]))          # hidden_dim
    layers = int(round(params[1]))          # lstm_layers
    drop   = params[2]                      # dropout
    lr     = params[3]                      # learning rate

    model = ADTCN(input_dim=6, hidden_dim=hidden,
                  lstm_layers=layers, dropout=drop).to(DEVICE)

    # Class weight (same as before)
    y_np = train_ds.tensors[1].numpy()
    fraud_r = y_np.mean() if len(y_np) > 0 else 0.01
    pos_w = torch.tensor([max(1.0, 1.0 / fraud_r)]).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = optim.Adam(model.parameters(), lr=lr)

    tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    best_mcc = -1.0
    for ep in range(EPOCHS_PER_CANDIDATE):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

        # Quick validation
        model.eval()
        pr, tr = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(DEVICE)
                lg = model(xb)
                p = torch.sigmoid(lg).cpu().numpy() > 0.5
                pr.extend(p.astype(int))
                tr.extend(yb.numpy().astype(int))

        pr = np.array(pr)
        tr = np.array(tr)
        if len(np.unique(tr)) < 2:
            mcc = 0.0
        else:
            mcc = matthews_corrcoef(tr, pr)

        best_mcc = max(best_mcc, mcc)

    # Fitness (paper style – higher is better)
    acc = accuracy_score(tr, pr)
    prec = precision_score(tr, pr, zero_division=0)
    fpr = ((tr == 0) & (pr == 1)).sum() / (tr == 0).sum() if (tr == 0).sum() > 0 else 0
    fitness = acc + best_mcc + prec - fpr

    return fitness   # minimize negative fitness (DBBOA minimizes)

# ── DB-BOA CLASS ─────────────────────────────────────────────────────────
class DBBOA:
    def __init__(self, obj_func, dim, bounds, pop_size=10):
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = np.array(bounds)          # shape (dim, 2)
        self.pop_size = pop_size

        # Population
        self.pop = np.random.uniform(self.bounds[:,0], self.bounds[:,1], (pop_size, dim))
        self.fitness = np.array([obj_func(ind) for ind in self.pop])

        self.best_idx = np.argmax(self.fitness)     # ← max
        self.best_pos = self.pop[self.best_idx].copy()
        self.best_fit = self.fitness[self.best_idx]

    def optimize(self, max_iter=15):
        for it in range(max_iter):
            print(f"Iter {it+1}/{max_iter} | Best fitness: {self.best_fit:.4f}")

            mean_fit = np.mean(self.fitness)
            for i in range(self.pop_size):
                if self.fitness[i] > mean_fit:  # better than average → butterfly (exploration)
                    c = 0.01
                    a = 0.1 * (1 - it / max_iter)  # decay
                    fragrance = c * (self.fitness[i] ** a)  # now safe (positive)

                    r = np.random.rand()
                    if r < 0.8:
                        self.pop[i] += (r ** 2) * (self.best_pos - self.pop[i]) * fragrance
                    else:
                        jk = random.sample(range(self.pop_size), 2)
                        self.pop[i] += (r ** 2) * (self.pop[jk[0]] - self.pop[jk[1]]) * fragrance
                else:
                    # worse → billiards exploitation
                    direction = self.best_pos - self.pop[i]
                    self.pop[i] += np.random.uniform(-0.5, 1.5) * direction

                self.pop[i] = np.clip(self.pop[i], self.bounds[:,0], self.bounds[:,1])
                new_fit = self.obj_func(self.pop[i])
                self.fitness[i] = new_fit

                if new_fit > self.best_fit:
                    self.best_fit = new_fit
                    self.best_pos = self.pop[i].copy()
                    self.best_idx = i

        return self.best_pos, self.best_fit



# ── MAIN TUNING ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Starting DB-BOA tuning on {DEVICE}")
    bounds_list = [PARAM_RANGES[k] for k in PARAM_KEYS]
    optimizer = DBBOA(obj_func=evaluate_candidate,
                      dim=len(PARAM_KEYS),
                      bounds=bounds_list,
                      pop_size=POP_SIZE)

    best_solution, best_real_fitness = optimizer.optimize(max_iter=MAX_ITER)

    # Convert to dict
    best_dict = {k: float(best_solution[i]) for i, k in enumerate(PARAM_KEYS)}
    best_dict['hidden_dim'] = int(round(best_dict['hidden_dim']))
    best_dict['lstm_layers'] = int(round(best_dict['lstm_layers']))

    print("\nBest hyperparameters:")
    print(best_dict)
    print(f"Best fitness (higher better): {best_real_fitness:.4f}")

    with open(BEST_PARAMS_FILE, 'w') as f:
        json.dump(best_dict, f, indent=4)
    print(f"Saved: {BEST_PARAMS_FILE}")




