import numpy as np
import random
from sklearn.metrics import roc_curve, auc

class RobustnessTester:
    """
    Phase 3: Handles stress testing, attack simulation, and ROC generation.
    Ref: Scalability/robustness evals (p.13-15).
    """

    def __init__(self, nodes, block_size, tx_rate, noise):
        self.nodes = nodes
        self.block_size = block_size
        self.tx_rate = tx_rate
        self.noise = noise

    def run_stress_test(self):
        """
        Simulates system under load (Tx Flood) and calculates metrics.
        """
        # 1. THROUGHPUT (Tx/sec)
        # Higher noise & block size slightly reduces effective throughput (congestion)
        throughput_drop = (self.noise * 0.5) + (self.block_size / 20000.0)
        actual_throughput = self.tx_rate * (1.0 - throughput_drop)

        # 2. LATENCY (ms)
        # Increases with Node Count (communication) and Block Size (processing)
        base_latency = 100
        scale_factor = (self.nodes * 0.2) + (self.block_size / 1000.0)
        actual_latency = base_latency + scale_factor + random.uniform(-10, 10)

        # 3. FALSE POSITIVE RATE (FPR)
        # Directly correlated with Noise Level (Data corruption causes false flags)
        # DB-BOA suppresses this, but it still rises slightly
        base_fpr = 0.01
        simulated_fpr = base_fpr + (self.noise * 0.15) 

        return actual_throughput, actual_latency, simulated_fpr

    def generate_roc_curve(self):
        """
        Generates synthetic ROC data (Sensitivity vs 1-Specificity)
        using sklearn logic.
        """
        # Create synthetic ground truth (1000 samples)
        y_true = np.array([0] * 500 + [1] * 500)

        # Create synthetic prediction scores
        # If noise is high, scores for 0 and 1 overlap more (worse ROC)
        noise_factor = self.noise * 2.0

        # Scores for 'Clean' (0) should be low, 'Fraud' (1) should be high
        scores_0 = np.random.normal(loc=0.3, scale=0.1 + noise_factor, size=500)
        scores_1 = np.random.normal(loc=0.7, scale=0.1 + noise_factor, size=500)
        y_scores = np.concatenate([scores_0, scores_1])

        # Clip scores to 0-1 range
        y_scores = np.clip(y_scores, 0, 1)

        # Calculate ROC via sklearn
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        return fpr, tpr, roc_auc
