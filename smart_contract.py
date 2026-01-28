import hashlib
import time
import random

class FinancialSecurityContract:
    """
    Python implementation of the 'FinancialSecurity.sol' Smart Contract.
    Manages Escrow, ADTCN Verification, and Settlement.
    """
    def __init__(self, owner_address):
        # Simulate Blockchain Deployment
        self.owner = owner_address
        self.contract_address = f"0x{hashlib.sha256(str(time.time()).encode()).hexdigest()[:40]}"
        self.balance = 0
        self.state = "DEPLOYED"  # States: DEPLOYED -> INITIATED -> APPROVED -> SETTLED or REVERTED
        self.logs = []
        self._log_event("ContractDeployed", f"Address: {self.contract_address}")

    def initiate_trade(self, buyer, seller, amount):
        """Step 1: Buyer deposits funds (Escrow)"""
        self.buyer = buyer
        self.seller = seller
        self.balance = amount
        self.state = "INITIATED"

        # Simulate Gas Cost
        gas_used = 45000 + random.randint(0, 5000)
        self._log_event("TradeInitiated", f"Amount: {amount} | Buyer: {buyer} | Gas: {gas_used}")
        return True

    def adtcn_oracle_check(self, fraud_score, threshold=0.5):
        """Step 2: The Smart Contract asks ADTCN (The Oracle) if trade is safe"""
        if fraud_score < threshold:
            self.state = "APPROVED"
            self._log_event("OracleUpdate", f"ADTCN Approved (Score: {fraud_score:.4f})")
            return True
        else:
            self.state = "REVERTED"
            self._log_event("OracleUpdate", f"ADTCN Rejected (Risk: {fraud_score:.4f})")
            return False

    def settle_trade(self):
        """Step 3: Release funds to Seller if Approved"""
        if self.state == "APPROVED":
            self.state = "SETTLED"
            transferred_amount = self.balance
            self.balance = 0
            self._log_event("TradeSettled", f"Released {transferred_amount} to {self.seller}")
            return True
        else:
            self._log_event("SettlementFailed", "Contract Logic Reverted: Not Approved")
            return False

    def _log_event(self, event_name, details):
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] [SMART_CONTRACT] {event_name}: {details}")

    def get_logs(self):
        return self.logs
