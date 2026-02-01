import numpy as np
import torch
import json
import os
import pandas as pd
import time
from web3 import Web3
from datetime import datetime

# Import existing validation logic and DB-BOA selector
from validate_tx import get_fraud_score, update_reputation
from dbboa_selector import select_leader_dbboa
from eth_utils import keccak, to_bytes

# ── CONFIG ───────────────────────────────────────────────────────────────
RPC_URL = "http://127.0.0.1:8545"
DATA_DIR = "data"
LOGS_DIR = "logs"
ACTIVITY_LOG = os.path.join(LOGS_DIR, "activity_log.csv")
CONFIG_FILE = os.path.join(LOGS_DIR, "config.json")
FRAUD_THRESHOLD = 0.40

# Replace with actual deployed addresses
LEDGER_ADDRESS = "0x923E9b1B9c600Eeaf91A39B3bf405f957B5377E6" 
ESCROW_ADDRESS = "0x4B3CDCD9f4f661eEef8528e3A66f4D59c944d973" 

# Consortium Leader (Default/Fallback)
LEADER_ADDRESS = "0xDC04300D97Ae2fd2a06a5A371457dE3316B1E38b"
LEADER_KEY = "0x94997c9a76fb07e5fc3da45b61b086ae20b8381ae3cbef3b4b33f084fe27f2fc"

w3 = Web3(Web3.HTTPProvider(RPC_URL))

# ── CONFIG LOADER ────────────────────────────────────────────────────────
def load_dynamic_config():
    """Reads the tuning parameters set in the Streamlit Sidebar"""
    if not os.path.exists(CONFIG_FILE):
        # Default fallback if UI hasn't created the file yet
        return {"block_size": 1, "tx_rate": 2.0, "noise_level": 0.05, "iterations": 50}
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

# ── LOGGING LOGIC ────────────────────────────────────────────────────────
def log_activity(prob, decision, tx_hash="N/A", leader="N/A"):
    """Saves AI decisions and the DB-BOA selected leader to CSV"""
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "selected_leader": leader,
        "fraud_prob": round(float(prob), 4),
        "decision": decision,
        "tx_hash": tx_hash if decision == "ACCEPT" else "BLOCKED"
    }
    df = pd.DataFrame([new_entry])
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    
    header = not os.path.exists(ACTIVITY_LOG)
    df.to_csv(ACTIVITY_LOG, mode='a', index=False, header=header)

# ── LOAD ABIs ────────────────────────────────────────────────────────────
ledger_abi = [{"inputs":[{"name":"txHash","type":"bytes32"}],"name":"storeTx","outputs":[],"stateMutability":"nonpayable","type":"function"}]
escrow_abi = [{"inputs":[],"name":"release","outputs":[],"stateMutability":"nonpayable","type":"function"}]

ledger_contract = w3.eth.contract(address=LEDGER_ADDRESS, abi=ledger_abi)
escrow_contract = w3.eth.contract(address=ESCROW_ADDRESS, abi=escrow_abi)

# ── ORCHESTRATION LOGIC ──────────────────────────────────────────────────
def process_transaction_block(sequence_idx, iterations=50, noise=0.05):
    # STEP 3a & 5: SELECT LEADER DYNAMICALLY VIA DB-BOA (Using Tuned Parameters)
    # Note: Ensure your dbboa_selector.py can accept these arguments
    current_leader = select_leader_dbboa(iterations=iterations, noise_level=noise)
    print(f"\n--- DB-BOA Selected Leader for Block {sequence_idx}: {current_leader} ---")

    X_val = np.load(os.path.join(DATA_DIR, "val_X.npy"), allow_pickle=True)
    # Loop to prevent index out of bounds
    seq = X_val[sequence_idx % len(X_val)]
    
    # AI Inference (Step 4 & 6)
    prob = get_fraud_score(seq)
    decision = "ACCEPT" if prob < FRAUD_THRESHOLD else "REJECT"
    
    print(f"AI Fraud Probability: {prob:.4f} | Final Decision: {decision}")

    actual_tx_hash = "N/A"

    if decision == "ACCEPT":
        current_nonce = w3.eth.get_transaction_count(LEADER_ADDRESS)
        tx_hash_bytes = keccak(to_bytes(text=str(seq.tolist())))
        actual_tx_hash = tx_hash_bytes.hex()
        
        print(f"Committing Hash to DataLedger via {current_leader[:10]}...")
        txn_l = ledger_contract.functions.storeTx(tx_hash_bytes).build_transaction({
            'from': LEADER_ADDRESS,
            'nonce': current_nonce,
            'gas': 200000,
            'gasPrice': w3.to_wei('20', 'gwei')
        })
        signed_l = w3.eth.account.sign_transaction(txn_l, LEADER_KEY)
        w3.eth.send_raw_transaction(signed_l.raw_transaction)

        print("Releasing Funds via Escrow...")
        txn_e = escrow_contract.functions.release().build_transaction({
            'from': LEADER_ADDRESS,
            'nonce': current_nonce + 1,
            'gas': 200000,
            'gasPrice': w3.to_wei('20', 'gwei')
        })
        signed_e = w3.eth.account.sign_transaction(txn_e, LEADER_KEY)
        w3.eth.send_raw_transaction(signed_e.raw_transaction)
        
        update_reputation(current_leader, is_fault=False)
        print("✅ Transaction Secured and Settled.")
    else:
        update_reputation(current_leader, is_fault=True)
        print(f"❌ Transaction Blocked. {current_leader[:10]} penalized.")

    log_activity(prob, decision, actual_tx_hash, current_leader)

# ── TUNABLE MAIN LOOP ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 DB-BOA + ADTCN Engine Started. Waiting for UI Tuning...")
    block_counter = 0

    try:
        while True:
            # 1. Load the latest tuning parameters from the UI
            config = load_dynamic_config()
            BLOCK_SIZE = config.get('block_size', 1)
            TX_RATE = config.get('tx_rate', 2.0)
            NOISE = config.get('noise_level', 0.05)
            ITERS = config.get('iterations', 50)

            print(f"\n[Config Update] Block Size: {BLOCK_SIZE} | Rate: {TX_RATE}s | Iters: {ITERS}")

            # 2. Process transactions based on tuned Block Size
            for _ in range(BLOCK_SIZE):
                process_transaction_block(block_counter, iterations=ITERS, noise=NOISE)
                block_counter += 1

            # 3. Wait for the tuned Transaction Rate
            time.sleep(TX_RATE)
            
    except KeyboardInterrupt:
        print("\n🛑 Engine stopped by user.")
