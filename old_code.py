import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from typing import Dict, List, Tuple, Optional
import torch
from collections import deque
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, suitable for Colab

# ===== Custom Gym Environment =====

class BlindSearchEnv(gym.Env):
    """Blind Search Environment - Compatible with Gymnasium Interface"""
    
    def __init__(self, grid_size: int = 50, sigma: float = 0.05, max_steps: int = 5000):
        super().__init__()
        self.grid_size = grid_size
        self.sigma = sigma
        self.max_steps = max_steps
        
        # Define action space: 8 discrete directions
        self.action_space = spaces.Discrete(8)
        
        # Define observation space
        # [agent_x, agent_y, visit_map(10x10), time_ratio, last_direction(8)]
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(2 + 100 + 1 + 8,), 
            dtype=np.float32
        )
        
        # Direction vectors
        self.directions = np.array([
            [1, 0], [1, 1], [0, 1], [-1, 1],
            [-1, 0], [-1, -1], [0, -1], [1, -1]
        ], dtype=np.float32) 

        
        norms = np.linalg.norm(self.directions, axis=1, keepdims=True)
        self.directions = self.directions / norms
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        # Agent starts from bottom-left corner
        self.agent_pos = np.array([0.0, 0.0])
        
        # Randomly generate target position
        margin = 0
        self.target_pos = np.array([
            self.np_random.uniform(margin, 1.0 - margin),
            self.np_random.uniform(margin, 1.0 - margin)
        ])
        
        # Initialize visit map and trajectory
        self.visited = np.zeros((self.grid_size, self.grid_size))
        self.trajectory = [self.agent_pos.copy()]
        self.steps = 0
        self.last_direction = 0
        
        # Mark initial position as visited
        self._mark_visited(self.agent_pos)
        
        return self._get_observation(), {}
    
    def step(self, action: int):
        """Execute action"""
        # Movement speed
        speed = 2.0 / self.grid_size
        
        # Update position
        new_pos = self.agent_pos + self.directions[action] * speed
        new_pos = np.clip(new_pos, 0, 1)
        
        # Calculate visit state before movement
        old_coverage = np.sum(self.visited > 0)
        
        self.agent_pos = new_pos
        self.trajectory.append(self.agent_pos.copy())
        self.steps += 1
        self.last_direction = action
        
        # Update visit map
        is_new_cell = self._mark_visited(self.agent_pos)
        new_coverage = np.sum(self.visited > 0)
        
        # Check if target is found
        distance = np.linalg.norm(self.agent_pos - self.target_pos)
        found = distance < self.sigma
        
        # Calculate reward
        reward = self._calculate_reward(
            found, distance, is_new_cell, 
            old_coverage, new_coverage
        )
        
        # Termination conditions
        terminated = found
        truncated = self.steps >= self.max_steps
        
        info = {
            'distance': distance,
            'coverage': new_coverage / (self.grid_size ** 2),
            'success': found,
            'steps': self.steps
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _calculate_reward(self, found: bool, distance: float, 
                         is_new_cell: bool, old_coverage: int, 
                         new_coverage: int) -> float:
        """Calculate reward function"""
        if found:
            # Reward for finding target, with bonus based on efficiency
            base_reward = 1000
            efficiency_bonus = max(0, (self.max_steps - self.steps) / self.max_steps) * 500
            return base_reward + efficiency_bonus
        
        # Base step penalty
        reward = -1
        
        # Reward for exploring new areas
        if is_new_cell:
            reward += 5
        
        # Reward for coverage increase
        coverage_increase = (new_coverage - old_coverage) / (self.grid_size ** 2)
        reward += coverage_increase * 100
        
        # Implicit reward based on distance to target (for guidance, but agent cannot directly sense target)
        # This shouldn't exist in true blind search, but helps accelerate training
        # reward -= distance * 0.1
        
        return reward
    
    def _mark_visited(self, pos: np.ndarray) -> bool:
        """Mark visited position, return if it's a new cell"""
        grid_x = int(pos[0] * self.grid_size)
        grid_y = int(pos[1] * self.grid_size)
        grid_x = min(grid_x, self.grid_size - 1)
        grid_y = min(grid_y, self.grid_size - 1)
        
        is_new = self.visited[grid_y, grid_x] == 0
        self.visited[grid_y, grid_x] += 1
        
        return is_new
    
    def _get_observation(self) -> np.ndarray:
        """Get observation"""
        obs = []
        
        # Agent position
        obs.extend(self.agent_pos.tolist())
        
        # Visit map (downsampled to 10x10)        
        # Improvement: normalize visit map, preserve frequency information
        visit_map = self.visited.reshape(10, 5, 10, 5).mean(axis=(1, 3))
        max_visits = np.max(visit_map) if np.max(visit_map) > 0 else 1
        visit_map = np.clip(visit_map / max_visits, 0, 1)

        obs.extend(visit_map.flatten().tolist())
        # Time ratio
        time_ratio = self.steps / self.max_steps
        obs.append(time_ratio)
        
        # Last direction (one-hot)
        direction_onehot = [0] * 8
        direction_onehot[self.last_direction] = 1
        obs.extend(direction_onehot)
        
        return np.array(obs, dtype=np.float32)
    
    def render(self, mode='human'):
        """Render environment"""
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            # Remove plt.ion() - not needed in Colab
        
        self.ax.clear()
        
        # Draw visit heatmap
        self.ax.imshow(self.visited.T, cmap='Blues', alpha=0.5, 
                      origin='lower', extent=[0, 1, 0, 1])
        
        # Draw trajectory
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.7, linewidth=2)
        
        # Draw Agent
        self.ax.plot(self.agent_pos[0], self.agent_pos[1], 'ro', markersize=10)
        
        # Draw target
        circle = plt.Circle(self.target_pos, self.sigma, color='green', 
                          fill=False, linewidth=2)
        self.ax.add_patch(circle)
        self.ax.plot(self.target_pos[0], self.target_pos[1], 'g*', markersize=15)
        
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect('equal')
        self.ax.set_title(f'Steps: {self.steps} | Coverage: {np.sum(self.visited > 0) / (self.grid_size ** 2):.1%}')
        self.ax.grid(True, alpha=0.3)
        
        # Remove plt.pause() - it will freeze in Colab
        plt.draw()
        plt.show()  # Show directly instead
        
    def close(self):
        """Close environment"""
        if hasattr(self, 'fig'):
            plt.close(self.fig)

# ===== Custom Callback Functions =====

class TrajectoryCollectorCallback(BaseCallback):
    """Callback to collect successful trajectories"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.successful_trajectories = []
        self.episode_count = 0
        self.success_count = 0
        
    def _on_step(self) -> bool:
        # Check if any environment is done
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            self.episode_count += 1
            
            # If target found successfully, save trajectory
            if info.get('success', False):
                self.success_count += 1
                # Get underlying environment (bypass Monitor wrapper)
                env = self.training_env.envs[0]
                if hasattr(env, 'env'):  # If it's a Monitor-wrapped environment
                    env = env.env
                self.successful_trajectories.append({
                    'trajectory': env.trajectory.copy(),
                    'steps': info['steps'],
                    'target': env.target_pos.copy()
                })
                
                if self.verbose > 0 and self.success_count % 10 == 0:
                    print(f"Collected {self.success_count} successful trajectories")
        
        return True

class ProgressCallback(BaseCallback):
    """Callback for training progress"""
    
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get training statistics
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                mean_length = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])
                
                self.episode_rewards.append(mean_reward)
                self.episode_lengths.append(mean_length)
                
                if self.verbose > 0:
                    print(f"Steps: {self.n_calls} | Mean Reward: {mean_reward:.2f} | Mean Length: {mean_length:.2f}")
        
        return True

# ===== Training Functions =====

import os
import pickle

def train_blind_search_agent(total_timesteps: int = 500000, 
                           algorithm: str = 'PPO',
                           render_freq: int = 0):
    """Train blind search agent"""
    
    print(f"=== Training Blind Search Agent with Stable-Baselines3 {algorithm} ===\n")
    
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"[GPU] Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        print("[CPU] GPU not available, using CPU")
    
    # Create environment - using fixed seed sequence
    env = BlindSearchEnv(grid_size=50, sigma=0.05, max_steps=5000)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # Create evaluation environment - using same fixed seed sequence
    eval_env = BlindSearchEnv(grid_size=50, sigma=0.05, max_steps=5000)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Select algorithm
    if algorithm == 'PPO':
        model = PPO(
            'MlpPolicy',
            env,
            device=device,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            policy_kwargs=dict(
                net_arch=[256, 256, 256],
                activation_fn=torch.nn.ReLU
            ),
            verbose=1,
            tensorboard_log="./blind_search_tensorboard/")
    elif algorithm == 'DQN':
        model = DQN(
            'MlpPolicy',
            env,
            learning_rate=1e-3,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            verbose=1,
            tensorboard_log="./blind_search_tensorboard/"
        )
    elif algorithm == 'SAC':
        model = SAC(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./blind_search_tensorboard/"
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Create callback functions
    trajectory_callback = TrajectoryCollectorCallback(verbose=1)
    progress_callback = ProgressCallback(check_freq=5000)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./blind_search_best_model/',
        log_path='./blind_search_logs/',
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    callbacks = [trajectory_callback, progress_callback, eval_callback]
    
    # Train model
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    print(f"\nTraining complete! Collected {len(trajectory_callback.successful_trajectories)} successful trajectories")
    
    # Save model locally
    model_path = f"blind_search_{algorithm.lower()}_final"
    model.save(model_path)
    print(f"Model saved locally: {model_path}")
    
    return model, trajectory_callback.successful_trajectories, progress_callback, model_path

# ===== Evaluation Functions =====

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

# ===== Trajectory Analysis Functions =====

def analyze_trajectories(trajectories: List[Dict], model=None):
    """Analyze learned trajectories"""
    
    print(f"\n=== Analyzing {len(trajectories)} Successful Trajectories ===")
    
    if len(trajectories) == 0:
        print("No successful trajectories to analyze")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Trajectory overlay plot
    ax = axes[0, 0]
    for i, traj_data in enumerate(trajectories[:30]):  # Show first 30
        traj = np.array(traj_data['trajectory'])
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.3, linewidth=1)
    ax.set_title('Overlay Trajectories (Top 30)')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # 2. Average visit heatmap
    ax = axes[0, 1]
    grid_size = 50
    avg_visited = np.zeros((grid_size, grid_size))
    
    for traj_data in trajectories:
        visited = np.zeros((grid_size, grid_size))
        traj = np.array(traj_data['trajectory'])
        for pos in traj:
            x = int(pos[0] * grid_size)
            y = int(pos[1] * grid_size)
            x = min(x, grid_size - 1)
            y = min(y, grid_size - 1)
            visited[y, x] = 1
        avg_visited += visited
    
    avg_visited /= len(trajectories)
    im = ax.imshow(avg_visited.T, cmap='hot', origin='lower', extent=[0, 1, 0, 1])
    ax.set_title('Mean Visited Heatmap')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)
    
    # 3. Steps distribution
    ax = axes[1, 0]
    steps = [t['steps'] for t in trajectories]
    ax.hist(steps, bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Steps Distribution (Mean: {np.mean(steps):.1f})')
    ax.grid(True, alpha=0.3)
    
    # 4. Target position distribution
    ax = axes[1, 1]
    targets = np.array([t['target'] for t in trajectories])
    scatter = ax.scatter(targets[:, 0], targets[:, 1], 
                        c=[t['steps'] for t in trajectories],
                        cmap='viridis', alpha=0.6)
    ax.set_xlabel('Target X')
    ax.set_ylabel('Target Y')
    ax.set_title('Target Position vs Steps')
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.colorbar(scatter, ax=ax, label='Steps')
    
    plt.tight_layout()
    plt.show()
    
    # If model exists, show an example run
    if model is not None:
        print("\nShowing a test run...")
        env = BlindSearchEnv(grid_size=50, sigma=0.05, max_steps=5000)
        obs, _ = env.reset()  # Each episode uses a different fixed seed
        
        for step in range(5000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 100 == 0:
                env.render()
            
            if terminated or truncated:
                print(f"Done! Steps: {info['steps']}, Success: {info['success']}")
                break
        
        env.render()
        # Remove input() - cannot be used in Colab
        print("Test run completed!")
        env.close()

# ===== Main Program =====

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

# ===== Compare Different Algorithms =====

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