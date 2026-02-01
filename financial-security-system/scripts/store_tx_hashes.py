import os
import json
import pandas as pd
from web3 import Web3
from eth_abi import abi
from eth_utils import keccak, to_bytes

# ── CONFIG ───────────────────────────────────────────────────────────────
RPC_URL = "http://127.0.0.1:8545"
CONTRACT_ADDRESS = "0x923E9b1B9c600Eeaf91A39B3bf405f957B5377E6"  # ← your latest deployed address
DATA_FILE = "../data/transactions.csv"
NUM_TX_TO_STORE = 5  # how many to store for testing

ACCOUNT_ADDRESS = "0xDC04300D97Ae2fd2a06a5A371457dE3316B1E38b"
PRIVATE_KEY = "0x94997c9a76fb07e5fc3da45b61b086ae20b8381ae3cbef3b4b33f084fe27f2fc"

# ── CONNECT ──────────────────────────────────────────────────────────────
w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    raise Exception("Ganache not connected!")

account = w3.eth.account.from_key(PRIVATE_KEY)

# ── LOAD ABI (minimal, just what we need) ────────────────────────────────
abi_json = [
    {
        "inputs": [{"internalType": "bytes32", "name": "txHash", "type": "bytes32"}],
        "name": "storeTx",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": False, "internalType": "uint256", "name": "id", "type": "uint256"},
            {"indexed": False, "internalType": "bytes32", "name": "hash", "type": "bytes32"}
        ],
        "name": "TxStored",
        "type": "event"
    }
]

contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=abi_json)

# ── READ SOME TRANSACTIONS & HASH ────────────────────────────────────────
df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} transactions from CSV")

for i in range(min(NUM_TX_TO_STORE, len(df))):
    row = df.iloc[i]
    # Create a simple string representation of the tx (can be more detailed later)
    tx_str = f"{row['timestamp']}|{row['sender']}|{row['receiver']}|{row['amount']}|{row['type']}"
    tx_bytes = to_bytes(text=tx_str)
    tx_hash = keccak(tx_bytes)  # keccak256

    print(f"\nStoring tx {i}:")
    print(f"  Original: {tx_str[:80]}...")
    print(f"  Hash: {tx_hash.hex()}")

    # Build tx
    txn = contract.functions.storeTx(tx_hash).build_transaction({
        'from': account.address,
        'nonce': w3.eth.get_transaction_count(account.address),
        'gas': 100000,
        'gasPrice': w3.to_wei('20', 'gwei'),
    })

    signed_txn = account.sign_transaction(txn)
    tx_hash_sent = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash_sent)

    print(f"  Stored! Tx hash on-chain: {tx_hash_sent.hex()}")
    print(f"  Gas used: {receipt.gasUsed}")

print("\nDone. You can now check Ganache console for TxStored events.")

