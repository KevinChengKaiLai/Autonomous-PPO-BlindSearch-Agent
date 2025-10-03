"""
Evaluation Module
Contains functions for evaluating trained agents.
"""

import numpy as np
from environment import BlindSearchEnv


def evaluate_agent(model, n_eval_episodes: int = 100, render: bool = False):
    """Evaluate trained agent"""
    
    print("\n=== Evaluating Agent Performance ===")
    
    # Use multiple different seed ranges for evaluation
    seed_ranges = [
        (0, 99),      # Seed range seen during training
        (100, 199),   # New seed range
        (200, 299),   # Another new seed range
    ]
    
    all_results = []
    
    for start_seed, end_seed in seed_ranges:
        env = BlindSearchEnv(grid_size=50, sigma=0.05, max_steps=5000)
        
        successes = 0
        steps_list = []
        
        for episode in range(n_eval_episodes):
            seed = start_seed + episode
            obs, _ = env.reset(seed=seed)
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                if render and episode < 5:
                    env.render()
            
            if info['success']:
                successes += 1
                steps_list.append(info['steps'])
        
        results = {
            'seed_range': (start_seed, end_seed),
            'success_rate': successes / n_eval_episodes,
            'avg_steps': np.mean(steps_list) if steps_list else float('inf')
        }
        all_results.append(results)
        env.close()
    
    # Calculate average results
    avg_success_rate = np.mean([r['success_rate'] for r in all_results])
    print(f"Multi-range evaluation average success rate: {avg_success_rate:.2%}")
    
    return all_results

