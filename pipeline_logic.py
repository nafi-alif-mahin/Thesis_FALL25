import time
import random
# Import your new Smart Contract
from smart_contract import FinancialSecurityContract

class PipelineController:
    def __init__(self):
        self.logs = []

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        return f"[{timestamp}] {message}"

    def run_full_simulation(self, nodes, amount, sender, receiver, is_fraud_sim):
        """
        Runs the full Thesis Flow: Data -> DB-BOA -> ADTCN -> Smart Contract
        """
        simulation_logs = []

        # 1. DB-BOA LEADER SELECTION
        leader_node = f"Validator_Node_{random.randint(1, nodes)}"
        simulation_logs.append(self.log(f"🦋 DB-BOA: Selected Leader {leader_node} (Lowest Latency)"))

        # 2. DEPLOY SMART CONTRACT
        # The leader node deploys the contract
        contract = FinancialSecurityContract(owner_address=leader_node)
        simulation_logs.extend(contract.get_logs())

        # 3. INITIATE TRADE
        contract.initiate_trade(buyer=sender, seller=receiver, amount=amount)
        simulation_logs.extend(contract.get_logs()[-1:]) # Get latest log

        # 4. ADTCN SECURITY CHECK (The "Oracle")
        # If "is_fraud_sim" is checked, we generate a high risk score
        if is_fraud_sim:
            fraud_score = random.uniform(0.6, 0.99)
        else:
            fraud_score = random.uniform(0.01, 0.3)

        simulation_logs.append(self.log(f"🧠 ADTCN: Anomaly Score {fraud_score:.4f}"))

        # 5. SMART CONTRACT DECISION
        is_approved = contract.adtcn_oracle_check(fraud_score)
        simulation_logs.extend(contract.get_logs()[-1:])

        status = "Blocked"
        if is_approved:
            # 6. SETTLEMENT
            contract.settle_trade()
            simulation_logs.extend(contract.get_logs()[-1:])
            status = "Settled"

        return simulation_logs, status, fraud_score, contract.contract_address
