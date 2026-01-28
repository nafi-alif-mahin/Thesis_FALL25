import numpy as np
import random

class BaselineSimulator:
    """
    Simulates the performance of baseline algorithms (BOA, MBO, WSA) 
    compared to DB-BOA based on the paper's trends (p.14-19).
    """

    def __init__(self, num_nodes):
        self.nodes = num_nodes

    def get_latency_data(self):
        """
        Returns average latency (ms) for Leader Selection at current node count.
        Paper Trend: DB-BOA is 2-36% faster.
        """
        # Base latency grows with nodes
        base = 100 + (self.nodes * 0.15) 

        # Add random jitter for realism
        noise = random.uniform(-5, 5)

        return {
            "DB-BOA (Ours)": base * 0.75 + noise,       # Best performance
            "Standard BOA": base * 0.90 + noise,        # Slower
            "MBO": base * 1.05 + noise,                 # Slower
            "WSA": base * 1.15 + noise                  # Slowest
        }

    def get_convergence_data(self, iterations=50):
        """
        Returns fitness curves over iterations.
        Paper Trend: DB-BOA converges 72-82% faster.
        """
        x = list(range(iterations))

        # DB-BOA: Rapid exponential decay (finds solution fast)
        db_boa = [1.0 * np.exp(-0.2 * i) + 0.1 for i in x]

        # BOA: Slower decay
        boa = [1.0 * np.exp(-0.1 * i) + 0.25 for i in x]

        # MBO & WSA: Very slow / Linear-like decay
        mbo = [1.0 * np.exp(-0.05 * i) + 0.35 for i in x]
        wsa = [1.0 * np.exp(-0.04 * i) + 0.40 for i in x]

        return x, db_boa, boa, mbo, wsa
