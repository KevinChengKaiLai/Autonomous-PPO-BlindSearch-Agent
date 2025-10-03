"""
Main Program Module
Entry point for training and evaluating blind search agents.
"""

import pickle
import matplotlib.pyplot as plt

from training import train_blind_search_agent
from evaluation import evaluate_agent


def main():
    """Main program: Train and evaluate blind search agent"""
    
    # Training parameters
    TOTAL_TIMESTEPS = 1e7
    ALGORITHM = 'PPO'
    
    # Train model
    model, trajectories, progress_callback, model_path = train_blind_search_agent(
        total_timesteps=TOTAL_TIMESTEPS,
        algorithm=ALGORITHM
    )
    
    # Evaluate model
    results = evaluate_agent(model, n_eval_episodes=100, render=False)
    
    # Save results data locally
    results_data = {
        'results': results,
        'trajectories_count': len(trajectories),
        'model_path': model_path,
        'algorithm': ALGORITHM,
        'timesteps': TOTAL_TIMESTEPS
    }
    
    results_path = 'training_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results_data, f)
    
    print(f"Training results saved locally: {results_path}")
    print(f"Model path: {model_path}")
    
    return model, trajectories, results, model_path


def compare_algorithms():
    """Compare performance of different RL algorithms"""
    
    algorithms = ['PPO', 'DQN', 'SAC']
    results = {}
    
    for algo in algorithms:
        print(f"\n{'='*50}")
        print(f"Training {algo}")
        print(f"{'='*50}")
        
        model, trajectories, _ = train_blind_search_agent(
            total_timesteps=200000,  # Fewer steps for quick comparison
            algorithm=algo
        )
        
        results[algo] = evaluate_agent(model, n_eval_episodes=50)
        results[algo]['trajectories'] = len(trajectories)
    
    # Visualize comparison results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    algos = list(results.keys())
    success_rates = [results[a]['success_rate'] for a in algos]
    avg_steps = [results[a]['avg_steps'] for a in algos]
    
    ax1.bar(algos, success_rates)
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Algorithm Success Rate Comparison')
    ax1.set_ylim(0, 1)
    
    ax2.bar(algos, avg_steps)
    ax2.set_ylabel('Mean Steps')
    ax2.set_title('Mean Steps Comparison')
    
    plt.tight_layout()
    plt.show()
    
    return results


if __name__ == "__main__":
    # Run main program
    model, trajectories, results, model_path = main()
    
    # If you want to compare different algorithms, uncomment the next line
    # comparison_results = compare_algorithms()

