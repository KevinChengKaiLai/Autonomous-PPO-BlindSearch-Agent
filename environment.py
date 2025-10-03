"""
Blind Search Environment Module
Contains the custom Gymnasium environment for blind search tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, suitable for Colab


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

