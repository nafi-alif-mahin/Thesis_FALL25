import torch
import torch.nn as nn
import numpy as np
import os
import json
from datetime import datetime

# ── CONFIG ───────────────────────────────────────────────────────────────

# Make paths relative to THIS script file's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")
LOGS_DIR = os.path.join(SCRIPT_DIR, "..", "logs")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAUD_THRESHOLD = 0.40  # adjust based on your preference

REPUTATION_FILE = os.path.join(LOGS_DIR, "reputation.json")

# ── LOAD IMPROVED MODEL ──────────────────────────────────────────────────
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
        return logits.squeeze(-1)

model = ImprovedADTCN(input_dim=6).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "adtcn_improved.pth"), map_location=DEVICE))
model.eval()

# ── FRAUD SCORE FUNCTION ─────────────────────────────────────────────────
def get_fraud_score(sequence):
    """sequence: np.array (seq_len, features)"""
    if sequence.ndim == 2:
        sequence = sequence[np.newaxis, ...]
    seq_tensor = torch.from_numpy(sequence.astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        logits = model(seq_tensor)
        prob = torch.sigmoid(logits).cpu().item()
    return prob

# ── REPUTATION MANAGEMENT ────────────────────────────────────────────────
def load_reputation():
    if os.path.exists(REPUTATION_FILE):
        with open(REPUTATION_FILE, 'r') as f:
            return json.load(f)
    # Initialize with your main Ganache account
    return {"0xDC04300D97Ae2fd2a06a5A371457dE3316B1E38b": 100}

def save_reputation(rep_dict):
    print("DEBUG: Inside save_reputation()")
    print("DEBUG: Target full path:", REPUTATION_FILE)
    
    os.makedirs(os.path.dirname(REPUTATION_FILE), exist_ok=True)
    
    # Atomic write pattern (safer)
    tmp_path = REPUTATION_FILE + ".tmp"
    try:
        with open(tmp_path, 'w') as f:
            json.dump(rep_dict, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, REPUTATION_FILE)  # atomic move
        print(f"Reputation saved to {REPUTATION_FILE}")
        if os.path.exists(REPUTATION_FILE):
            print("DEBUG: File confirmed exists, size:", os.path.getsize(REPUTATION_FILE), "bytes")
    except Exception as e:
        print(f"ERROR during save: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

def update_reputation(proposer_address, is_fault=False, points_success=15, points_fault=-30):
    print("DEBUG: Entering update_reputation()")
    
    rep = load_reputation()
    current = rep.get(proposer_address, 100)
    
    if is_fault:
        rep[proposer_address] = max(0, current + points_fault)
        print(f"Reputation penalized for {proposer_address}: {current} → {rep[proposer_address]}")
    else:
        rep[proposer_address] = current + points_success
        print(f"Reputation bonus for {proposer_address}: {current} → {rep[proposer_address]}")
    
    print(f"DEBUG: Attempting to save reputation to {REPUTATION_FILE}")
    save_reputation(rep)

# ── VALIDATION FUNCTION ──────────────────────────────────────────────────
def validate_transaction(sequence, proposer_address="0xDC04300D97Ae2fd2a06a5A371457dE3316B1E38b"):
    """
    sequence: numpy array of shape (10, 6)
    proposer_address: Ethereum address of the proposing node/leader
    """
    fraud_prob = get_fraud_score(sequence)
    decision = "ACCEPT" if fraud_prob < FRAUD_THRESHOLD else "REJECT"
    
    print(f"\nValidation at {datetime.now()}")
    print(f"  Fraud probability: {fraud_prob:.4f}")
    print(f"  Decision: {decision} (threshold = {FRAUD_THRESHOLD})")
    print(f"  Proposer: {proposer_address}")
    
    # Simulate reputation update
    is_fault = (decision == "REJECT")
    update_reputation(proposer_address, is_fault=is_fault)
    
    return {
        "fraud_prob": fraud_prob,
        "decision": decision,
        "proposer": proposer_address,
        "timestamp": datetime.now().isoformat()
    }

# ── EXAMPLE USAGE ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    print("Reputation file will be saved to:", REPUTATION_FILE)
    
    # Load one validation sequence as example
    try:
        X_val = np.load(os.path.join(DATA_DIR, "val_X.npy"), allow_pickle=True).astype(np.float32)
        
        print("Testing validation on 5 random validation sequences:")
        for i in np.random.choice(len(X_val), 5, replace=False):
            seq = X_val[i]
            true_label = int(np.load(os.path.join(DATA_DIR, "val_y.npy"), allow_pickle=True)[i])
            result = validate_transaction(seq)
            print(f"  True label was: {'FRAUD' if true_label else 'NORMAL'}")
            print("-"*60)
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        print("Make sure the data folder structure is correct relative to this script.")



