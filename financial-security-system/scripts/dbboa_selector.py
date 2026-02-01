import json
import random
import numpy as np

def calculate_fitness(node_stats):
    # Minimizing CT + CC + MS (Lower is better)
    return node_stats['CT'] + node_stats['CC'] + node_stats['MS']

# Added 'iterations' and 'noise_level' arguments to match main_engine.py call
def select_leader_dbboa(nodes_file="logs/nodes_config.json", iterations=50, noise_level=0.05):
    with open(nodes_file, 'r') as f:
        nodes = json.load(f)
    
    best_leader = None
    min_cost = float('inf')
    
    # DB-BOA Logic Parameters
    p = 0.8  # Switch probability
    c = 0.01 # Sensor modality
    a = 0.1  # Power exponent
    
    # We now run the selection process for 'iterations' number of times
    # This simulates the convergence of the butterflies toward the best node
    for _ in range(iterations):
        for addr, stats in nodes.items():
            # 1. Calculate Stimulus Intensity (I) based on Fitness
            I = 1 / (1 + calculate_fitness(stats)) 
            
            # 2. Calculate Fragrance (f = c * I^a)
            fragrance = c * (I ** a)
            
            # 3. Selection Phase (Global vs Local Search)
            if random.random() < p:
                # Move toward best fragrance (Global Search)
                cost = calculate_fitness(stats) - fragrance
            else:
                # Random movement (Local Search) 
                # We use the 'noise_level' from the UI here
                random_step = random.uniform(-noise_level, noise_level)
                cost = calculate_fitness(stats) + random_step
            
            # Tracking the best performing node
            if cost < min_cost:
                min_cost = cost
                best_leader = addr
                
    return best_leader
