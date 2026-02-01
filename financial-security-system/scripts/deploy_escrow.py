from web3 import Web3
from solcx import compile_source
import solcx

# ── CONFIG ───────────────────────────────────────────────────────────────
RPC_URL = "http://127.0.0.1:8545"
GANACHE_ACCOUNT = "0xDC04300D97Ae2fd2a06a5A371457dE3316B1E38b"
PRIVATE_KEY = "0x94997c9a76fb07e5fc3da45b61b086ae20b8381ae3cbef3b4b33f084fe27f2fc"  # your account 0

# Example buyer/seller (use Ganache accounts 1 and 2)
BUYER = "0xAa3228850e79B90882CF91cd67D3813E3515b95F"
SELLER = "0xA7aA403a7A47C9eAec8b03f1c2ec3E88254Bb8c2"
VALIDATOR = "0xDC04300D97Ae2fd2a06a5A371457dE3316B1E38b" # Your Main Account

# ── COMPILE & DEPLOY ─────────────────────────────────────────────────────
w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    raise Exception("Ganache not running!")

account = w3.eth.account.from_key(PRIVATE_KEY)

with open("contracts/Escrow.sol", "r") as f:
    source = f.read()


solcx.set_solc_version('0.8.0') # Add this line before compiling
compiled = compile_source(source, output_values=['abi', 'bin'], solc_version='0.8.0')
interface = compiled['<stdin>:Escrow']
contract = w3.eth.contract(abi=interface['abi'], bytecode=interface['bin'])

# Deploy with 1 ETH deposit
construct_tx = contract.constructor(BUYER, SELLER, VALIDATOR).build_transaction({
    'from': account.address,
    'value': w3.to_wei(1, 'ether'),
    'gas': 3000000,
    'gasPrice': w3.to_wei('20', 'gwei'),
    'nonce': w3.eth.get_transaction_count(account.address),
})

signed = account.sign_transaction(construct_tx)
tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

escrow_address = receipt.contractAddress

print("\nEscrow contract deployed!")
print("Address:", escrow_address)
print("Tx hash:", tx_hash.hex())
print("Deposited: 1 ETH")
print("Buyer:", BUYER)
print("Seller:", SELLER)



