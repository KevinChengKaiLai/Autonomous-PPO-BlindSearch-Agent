"""
Training Module
Contains functions for training reinforcement learning agents.
"""

import torch
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from environment import BlindSearchEnv
from callbacks import TrajectoryCollectorCallback, ProgressCallback


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

