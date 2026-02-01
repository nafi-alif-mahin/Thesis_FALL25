import json
import os
from web3 import Web3
from solcx import compile_source, install_solc

# ── CONFIG ───────────────────────────────────────────────────────────────
RPC_URL = "http://127.0.0.1:8545"  # Your Ganache RPC
CONTRACT_FILE = "../contracts/DataLedger.sol"
SOLC_VERSION = "0.8.0"  # Matches pragma in .sol

# Use the first Ganache account (replace with your actual private key if needed)
ACCOUNT_0_ADDRESS = "0xDC04300D97Ae2fd2a06a5A371457dE3316B1E38b"
ACCOUNT_0_PRIVATE_KEY = "0x94997c9a76fb07e5fc3da45b61b086ae20b8381ae3cbef3b4b33f084fe27f2fc"

# ── COMPILE CONTRACT ─────────────────────────────────────────────────────
print("Installing Solidity compiler version", SOLC_VERSION, "...")
install_solc(SOLC_VERSION)

with open(CONTRACT_FILE, "r") as f:
    source_code = f.read()

compiled_sol = compile_source(source_code, output_values=['abi', 'bin'], solc_version=SOLC_VERSION)
contract_interface = compiled_sol['<stdin>:DataLedger']

abi = contract_interface['abi']
bytecode = contract_interface['bin']

print("Contract compiled successfully.")

# ── CONNECT TO GANACHE & DEPLOY ──────────────────────────────────────────
w3 = Web3(Web3.HTTPProvider(RPC_URL))

if not w3.is_connected():
    raise Exception("Cannot connect to Ganache! Is it running?")

print("Connected to Ganache.")

account = w3.eth.account.from_key(ACCOUNT_0_PRIVATE_KEY)

# Build transaction
contract = w3.eth.contract(abi=abi, bytecode=bytecode)
construct_txn = contract.constructor().build_transaction({
    'from': account.address,
    'nonce': w3.eth.get_transaction_count(account.address),
    'gas': 2000000,
    'gasPrice': w3.to_wei('20', 'gwei'),
})

# Sign & send
signed_txn = account.sign_transaction(construct_txn)
tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)

print("Transaction sent. Waiting for receipt...")
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

contract_address = tx_receipt.contractAddress

print("\n" + "="*60)
print("DataLedger contract DEPLOYED SUCCESSFULLY!")
print("Contract Address:", contract_address)
print("Transaction Hash:", tx_hash.hex())
print("Deployed from account:", account.address)
print("="*60)

# Optional: Save address to a file for later use
with open("../data/contract_address.txt", "w") as f:
    f.write(contract_address)

print("Contract address also saved to data/contract_address.txt")
