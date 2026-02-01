# scripts/train_best_adtcn.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# ── CONFIG ───────────────────────────────────────────────────────────────
DATA_DIR = "../data"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 40               # longer final training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Best params from DB-BOA
best_params = {
    "hidden_dim": 58,
    "lstm_layers": 3,
    "dropout": 0.3653,
    "lr": 0.001764
}

# ── SAFE LOAD ────────────────────────────────────────────────────────────
def safe_load_npy(fn):
    p = os.path.join(DATA_DIR, fn)
    try:
        return np.load(p, allow_pickle=False).astype(np.float32)
    except:
        arr = np.load(p, allow_pickle=True)
        if arr.dtype == object:
            arr = np.asarray(arr.tolist(), dtype=np.float32)
        return arr.astype(np.float32)

X_train = safe_load_npy("train_X.npy")
y_train = safe_load_npy("train_y.npy")
X_val   = safe_load_npy("val_X.npy")
y_val   = safe_load_npy("val_y.npy")

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# ── MODEL ────────────────────────────────────────────────────────────────
class ADTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(6, best_params["hidden_dim"], best_params["lstm_layers"],
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(best_params["dropout"])
        self.attn = nn.Linear(best_params["hidden_dim"]*2, 1)
        self.fc   = nn.Linear(best_params["hidden_dim"]*2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        w = torch.softmax(self.attn(out), dim=1)
        ctx = torch.sum(out * w, dim=1)
        return self.fc(ctx).squeeze(-1)  # logits

model = ADTCN().to(DEVICE)

fraud_r = y_train.mean()
pos_weight = torch.tensor([max(1.0, 1.0 / fraud_r)]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])

# ── TRAIN ────────────────────────────────────────────────────────────────
best_mcc = -1.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Eval
    model.eval()
    pr, tr = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            lg = model(xb)
            p = torch.sigmoid(lg).cpu().numpy() > 0.5
            pr.extend(p.astype(int))
            tr.extend(yb.numpy().astype(int))

    pr = np.array(pr)
    tr = np.array(tr)
    mcc = matthews_corrcoef(tr, pr) if len(np.unique(tr)) > 1 else 0
    acc = accuracy_score(tr, pr)
    prec = precision_score(tr, pr, zero_division=0)
    rec = recall_score(tr, pr, zero_division=0)
    f1 = f1_score(tr, pr, zero_division=0)

    print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | "
          f"ACC: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")

    if mcc > best_mcc:
        best_mcc = mcc
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_adtcn.pth"))
        print(f"  → Saved best model (MCC {mcc:.4f})")

print(f"\nFinal training done. Best MCC: {best_mcc:.4f}")
print("Model saved as: models/best_adtcn.pth")



