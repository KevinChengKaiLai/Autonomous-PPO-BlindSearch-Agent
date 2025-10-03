"""
Custom Callback Functions Module
Contains custom callbacks for training monitoring and trajectory collection.
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


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

