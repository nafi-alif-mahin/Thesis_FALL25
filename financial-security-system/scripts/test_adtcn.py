import torch
import torch.nn as nn        
import numpy as np
import os

# ── CONFIG ───────────────────────────────────────────────────────────────
DATA_DIR = "../data"
MODEL_DIR = "../models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── SAFE LOAD FUNCTION ───────────────────────────────────────────────────
def safe_load_npy(filename):
    path = os.path.join(DATA_DIR, filename)
    try:
        return np.load(path, allow_pickle=False).astype(np.float32)
    except ValueError:
        print(f"Warning: allow_pickle needed for {filename}")
        arr = np.load(path, allow_pickle=True)
        if arr.dtype == object:
            arr = np.asarray(arr.tolist(), dtype=np.float32)
        return arr.astype(np.float32)

# ── MODEL CLASS (must match exactly the trained one) ─────────────────────
class ImprovedADTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        logits = self.fc(context)
        return logits.squeeze(-1)  # raw logits

# ── LOAD MODEL ───────────────────────────────────────────────────────────
model = ImprovedADTCN(input_dim=6).to(DEVICE)
model_path = os.path.join(MODEL_DIR, "adtcn_improved.pth")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Improved model not found: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
print(f"Improved model loaded successfully from: {model_path}")

# ── FRAUD SCORE FUNCTION ─────────────────────────────────────────────────
def get_fraud_score(seq):
    """seq: (seq_len, features) or (1, seq_len, features)"""
    if seq.ndim == 2:
        seq = seq[np.newaxis, ...]
    seq_tensor = torch.from_numpy(seq.astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        logits = model(seq_tensor)
        prob = torch.sigmoid(logits).cpu().item()
    return prob

# ── TEST ON VALIDATION ───────────────────────────────────────────────────
print("\nLoading validation data...")
X_val = safe_load_npy("val_X.npy")
y_val = safe_load_npy("val_y.npy")

print(f"Loaded {X_val.shape[0]} validation sequences")

print("\nExample fraud scores (first 10 sequences):")
for i in range(min(10, len(X_val))):
    score = get_fraud_score(X_val[i])
    true_label = int(y_val[i])
    pred_class = "FRAUD" if score > 0.5 else "NORMAL"
    print(f"Seq {i:3d} | True: {true_label} | Prob: {score:.4f} → {pred_class}")

# Top 5 highest probabilities
probs = [get_fraud_score(X_val[i]) for i in range(len(X_val))]
top_indices = np.argsort(probs)[-5:][::-1]  # descending

print("\nTop 5 highest fraud probability sequences:")
for idx in top_indices:
    score = probs[idx]
    true = int(y_val[idx])
    print(f"  Index {idx:4d} | True: {true} | Prob: {score:.4f}")



