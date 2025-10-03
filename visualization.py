"""
Visualization Module
Contains functions for trajectory analysis and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from environment import BlindSearchEnv


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

