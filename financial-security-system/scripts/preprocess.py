import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from datetime import datetime, timedelta
import random

# ── CONFIG ───────────────────────────────────────────────────────────────
OUTPUT_DIR = "../data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_TX = 25000               # total transactions
SEQ_LEN = 10               # sequence length per sender for ADTCN
FRAUD_RATE = 0.015         # ~1.5% fraud

np.random.seed(42)
random.seed(42)

# ── Generate synthetic data ─────────────────────────────────────────────
start_date = datetime(2024, 1, 1)

data = {
    'timestamp': [start_date + timedelta(hours=random.randint(0, 365*24*2)) for _ in range(N_TX)],
    'amount': np.random.lognormal(mean=5, sigma=1.8, size=N_TX).clip(5, 15000),
    'sender': [f"sender_{random.randint(1, 3000)}" for _ in range(N_TX)],  # ~3000 unique senders
    'receiver': [f"receiver_{random.randint(1, 5000)}" for _ in range(N_TX)],
    'type': np.random.choice(['payment', 'trade', 'transfer'], N_TX, p=[0.55, 0.30, 0.15]),
}

df = pd.DataFrame(data)
df = df.sort_values('timestamp').reset_index(drop=True)

# Add basic time features
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek

# Inject fraud (simple rule-based for now)
fraud_idx = np.random.choice(df.index, size=int(FRAUD_RATE * N_TX), replace=False)
df['label'] = 0
df.loc[fraud_idx, 'label'] = 1

# Make fraud more detectable: very high amount + unusual type/time
df.loc[fraud_idx, 'amount'] *= np.random.uniform(4, 25, len(fraud_idx))
df.loc[fraud_idx, 'type'] = np.random.choice(['trade', 'transfer'], len(fraud_idx))
# Find indices that are BOTH fraudulent AND happen at night (hour 2,3,4)
night_fraud_idx = np.intersect1d(fraud_idx, df.index[df['hour'].isin([2,3,4])])

# Apply multiplier only to those rows
df.loc[night_fraud_idx, 'amount'] *= 1.5
print(f"Generated {len(df)} transactions. Fraud rate: {df['label'].mean():.4f}")

# ── Normalize ────────────────────────────────────────────────────────────
scaler_amount = StandardScaler()
df['amount_norm'] = scaler_amount.fit_transform(df[['amount']])

# One-hot type
df_type = pd.get_dummies(df['type'], prefix='type')
df = pd.concat([df, df_type], axis=1)

# ── Create sequences ─────────────────────────────────────────────────────
feature_cols = ['amount_norm', 'hour', 'dayofweek'] + [c for c in df.columns if c.startswith('type_')]

sequences = []
labels = []

grouped = df.groupby('sender')

for sender, group in grouped:
    group = group.sort_values('timestamp')
    if len(group) < SEQ_LEN:
        continue
    for i in range(len(group) - SEQ_LEN + 1):
        seq = group.iloc[i:i+SEQ_LEN][feature_cols].values
        # Label = 1 if ANY tx in the sequence is fraud (common in fraud detection)
        seq_label = 1 if group.iloc[i:i+SEQ_LEN]['label'].any() else 0
        sequences.append(seq)
        labels.append(seq_label)

X = np.array(sequences)
y = np.array(labels)

print(f"Created {len(X)} sequences. Shape: {X.shape}, Fraud rate: {y.mean():.4f}")

# ── Split & Save ─────────────────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

np.save(os.path.join(OUTPUT_DIR, 'train_X.npy'), X_train)
np.save(os.path.join(OUTPUT_DIR, 'train_y.npy'), y_train)
np.save(os.path.join(OUTPUT_DIR, 'val_X.npy'),   X_val)
np.save(os.path.join(OUTPUT_DIR, 'val_y.npy'),   y_val)
np.save(os.path.join(OUTPUT_DIR, 'test_X.npy'),  X_test)
np.save(os.path.join(OUTPUT_DIR, 'test_y.npy'),  y_test)

# Also save original CSV for reference/dashboard
df.to_csv(os.path.join(OUTPUT_DIR, 'transactions.csv'), index=False)

print("Done! Files saved in data/:")
print(" - transactions.csv")
print(" - train_X.npy, train_y.npy, etc.")
