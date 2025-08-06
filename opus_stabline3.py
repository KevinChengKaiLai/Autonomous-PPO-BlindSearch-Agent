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
matplotlib.use('Agg')  # 非互動式後端，適合Colab

# ===== 自定義 Gym 環境 =====

class BlindSearchEnv(gym.Env):
    """盲目搜索環境 - 符合 Gymnasium 接口"""
    
    def __init__(self, grid_size: int = 50, sigma: float = 0.05, max_steps: int = 5000):
        super().__init__()
        self.grid_size = grid_size
        self.sigma = sigma
        self.max_steps = max_steps
        
        # 定義動作空間：8個離散方向
        self.action_space = spaces.Discrete(8)
        
        # 定義觀察空間
        # [agent_x, agent_y, visit_map(10x10), time_ratio, last_direction(8)]
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(2 + 100 + 1 + 8,), 
            dtype=np.float32
        )
        
        # 方向向量
        self.directions = np.array([
            [1, 0], [1, 1], [0, 1], [-1, 1],
            [-1, 0], [-1, -1], [0, -1], [1, -1]
        ], dtype=np.float32) 

        
        norms = np.linalg.norm(self.directions, axis=1, keepdims=True)
        self.directions = self.directions / norms
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        """reset environment"""
        super().reset(seed=seed)
        
        # Agent 從左下角開始
        self.agent_pos = np.array([0.0, 0.0])
        
        # 隨機生成目標位置
        margin = 0
        self.target_pos = np.array([
            self.np_random.uniform(margin, 1.0 - margin),
            self.np_random.uniform(margin, 1.0 - margin)
        ])
        
        # 初始化訪問地圖和軌跡
        self.visited = np.zeros((self.grid_size, self.grid_size))
        self.trajectory = [self.agent_pos.copy()]
        self.steps = 0
        self.last_direction = 0
        
        # 記錄初始位置為已訪問
        self._mark_visited(self.agent_pos)
        
        return self._get_observation(), {}
    
    def step(self, action: int):
        """execute action"""
        # 移動速度
        speed = 2.0 / self.grid_size
        
        # 更新位置
        new_pos = self.agent_pos + self.directions[action] * speed
        new_pos = np.clip(new_pos, 0, 1)
        
        # 計算移動前的訪問狀態
        old_coverage = np.sum(self.visited > 0)
        
        self.agent_pos = new_pos
        self.trajectory.append(self.agent_pos.copy())
        self.steps += 1
        self.last_direction = action
        
        # 更新訪問地圖
        is_new_cell = self._mark_visited(self.agent_pos)
        new_coverage = np.sum(self.visited > 0)
        
        # 檢查是否找到目標
        distance = np.linalg.norm(self.agent_pos - self.target_pos)
        found = distance < self.sigma
        
        # 計算獎勵
        reward = self._calculate_reward(
            found, distance, is_new_cell, 
            old_coverage, new_coverage
        )
        
        # 終止條件
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
        """calculate reward function"""
        if found:
            # 找到目標的獎勵，根據步數給予額外獎勵
            base_reward = 1000
            efficiency_bonus = max(0, (self.max_steps - self.steps) / self.max_steps) * 500
            return base_reward + efficiency_bonus
        
        # 基礎步數懲罰
        reward = -1
        
        # 探索新區域獎勵
        if is_new_cell:
            reward += 5
        
        # 覆蓋率增長獎勵
        coverage_increase = (new_coverage - old_coverage) / (self.grid_size ** 2)
        reward += coverage_increase * 100
        
        # 距離目標的隱式獎勵（用於引導，但不能直接感知目標）
        # 這個在實際盲目搜索中不應該有，但有助於加速訓練
        # reward -= distance * 0.1
        
        return reward
    
    def _mark_visited(self, pos: np.ndarray) -> bool:
        """mark visited position, return if it's a new cell"""
        grid_x = int(pos[0] * self.grid_size)
        grid_y = int(pos[1] * self.grid_size)
        grid_x = min(grid_x, self.grid_size - 1)
        grid_y = min(grid_y, self.grid_size - 1)
        
        is_new = self.visited[grid_y, grid_x] == 0
        self.visited[grid_y, grid_x] += 1
        
        return is_new
    
    def _get_observation(self) -> np.ndarray:
        """get observation"""
        obs = []
        
        # Agent 位置
        obs.extend(self.agent_pos.tolist())
        
        # 訪問地圖（降採樣到 10x10）        
        # 🔧 改進：訪問地圖歸一化，保留頻率信息
        visit_map = self.visited.reshape(10, 5, 10, 5).mean(axis=(1, 3))
        max_visits = np.max(visit_map) if np.max(visit_map) > 0 else 1
        visit_map = np.clip(visit_map / max_visits, 0, 1)

        obs.extend(visit_map.flatten().tolist())
        # 時間比例
        obs.append(self.steps / self.max_steps)
        
        # 上一個方向（one-hot）
        direction_onehot = [0] * 8
        direction_onehot[self.last_direction] = 1
        obs.extend(direction_onehot)
        
        return np.array(obs, dtype=np.float32)
    
    def render(self, mode='human'):
        """render environment"""
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            # 移除 plt.ion() - 在Colab中不需要
        
        self.ax.clear()
        
        # 繪製訪問熱力圖
        self.ax.imshow(self.visited.T, cmap='Blues', alpha=0.5, 
                      origin='lower', extent=[0, 1, 0, 1])
        
        # 繪製軌跡
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.7, linewidth=2)
        
        # 繪製 Agent
        self.ax.plot(self.agent_pos[0], self.agent_pos[1], 'ro', markersize=10)
        
        # 繪製目標
        circle = plt.Circle(self.target_pos, self.sigma, color='green', 
                          fill=False, linewidth=2)
        self.ax.add_patch(circle)
        self.ax.plot(self.target_pos[0], self.target_pos[1], 'g*', markersize=15)
        
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect('equal')
        self.ax.set_title(f'Steps: {self.steps} | Coverage: {np.sum(self.visited > 0) / (self.grid_size ** 2):.1%}')
        self.ax.grid(True, alpha=0.3)
        
        # 移除 plt.pause() - 在Colab中會卡住
        plt.draw()
        plt.show()  # 改為直接顯示
        
    def close(self):
        """close environment"""
        if hasattr(self, 'fig'):
            plt.close(self.fig)

# ===== 自定義回調函數 =====

class TrajectoryCollectorCallback(BaseCallback):
    """collect successful trajectories callback"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.successful_trajectories = []
        self.episode_count = 0
        self.success_count = 0
        
    def _on_step(self) -> bool:
        # 檢查是否有環境完成
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            self.episode_count += 1
            
            # 如果成功找到目標，保存軌跡
            if info.get('success', False):
                self.success_count += 1
                # 獲取底層環境（繞過Monitor包裝）
                env = self.training_env.envs[0]
                if hasattr(env, 'env'):  # 如果是Monitor包裝的環境
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
    """training progress callback"""
    
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # 獲取訓練統計
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                mean_length = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])
                
                self.episode_rewards.append(mean_reward)
                self.episode_lengths.append(mean_length)
                
                if self.verbose > 0:
                    print(f"Steps: {self.n_calls} | Mean Reward: {mean_reward:.2f} | Mean Length: {mean_length:.2f}")
        
        return True

# ===== 訓練函數 =====

import os
import pickle

def train_blind_search_agent(total_timesteps: int = 500000, 
                           algorithm: str = 'PPO',
                           render_freq: int = 0):
    """train blind search agent"""
    
    print(f"=== use Stable-Baselines3 {algorithm} to train blind search agent ===\n")
    
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        print("⚠️  GPU not available, using CPU")
    
    # 創建環境 - 使用固定 seed 序列
    env = BlindSearchEnv(grid_size=50, sigma=0.05, max_steps=5000)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # 創建評估環境 - 使用相同的固定 seed 序列
    eval_env = BlindSearchEnv(grid_size=50, sigma=0.05, max_steps=5000)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # 選擇算法
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
    
    # 創建回調函數
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
    
    # 訓練模型
    print("start training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    print(f"\ntraining done! collected {len(trajectory_callback.successful_trajectories)} successful trajectories")
    
    # 保存模型到本地
    model_path = f"blind_search_{algorithm.lower()}_final"
    model.save(model_path)
    print(f"模型已保存到本地: {model_path}")
    
    return model, trajectory_callback.successful_trajectories, progress_callback, model_path

# ===== 評估函數 =====

def evaluate_agent(model, n_eval_episodes: int = 100, render: bool = False):
    """evaluate trained agent"""
    
    print("\n=== evaluate agent performance ===")
    
    # 使用多個不同的 seed 範圍進行評估
    seed_ranges = [
        (0, 99),      # 訓練時看到的 seed 範圍
        (100, 199),   # 新的 seed 範圍
        (200, 299),   # 另一個新的 seed 範圍
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
    
    # 計算平均結果
    avg_success_rate = np.mean([r['success_rate'] for r in all_results])
    print(f"多範圍評估平均成功率: {avg_success_rate:.2%}")
    
    return all_results

# ===== 軌跡分析函數 =====

def analyze_trajectories(trajectories: List[Dict], model=None):
    """分析學習到的軌跡"""
    
    print(f"\n=== analyze {len(trajectories)} successful trajectories ===")
    
    if len(trajectories) == 0:
        print("no successful trajectories to analyze")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 軌跡疊加圖
    ax = axes[0, 0]
    for i, traj_data in enumerate(trajectories[:30]):  # 顯示前30條
        traj = np.array(traj_data['trajectory'])
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.3, linewidth=1)
    ax.set_title('overlay trajectories (top 30)')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # 2. 平均訪問熱力圖
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
    ax.set_title('mean visited heatmap')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)
    
    # 3. 步數分布
    ax = axes[1, 0]
    steps = [t['steps'] for t in trajectories]
    ax.hist(steps, bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('steps')
    ax.set_ylabel('frequency')
    ax.set_title(f'steps distribution (mean: {np.mean(steps):.1f})')
    ax.grid(True, alpha=0.3)
    
    # 4. 目標位置分布
    ax = axes[1, 1]
    targets = np.array([t['target'] for t in trajectories])
    scatter = ax.scatter(targets[:, 0], targets[:, 1], 
                        c=[t['steps'] for t in trajectories],
                        cmap='viridis', alpha=0.6)
    ax.set_xlabel('target X')
    ax.set_ylabel('target Y')
    ax.set_title('target position vs steps')
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.colorbar(scatter, ax=ax, label='steps')
    
    plt.tight_layout()
    plt.show()
    
    # 如果有模型，展示一個示例運行
    if model is not None:
        print("\nshow a test run...")
        env = BlindSearchEnv(grid_size=50, sigma=0.05, max_steps=5000)
        obs, _ = env.reset()  # 每個 episode 使用不同的固定 seed
        
        for step in range(5000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 100 == 0:
                env.render()
            
            if terminated or truncated:
                print(f"done! steps: {info['steps']}, success: {info['success']}")
                break
        
        env.render()
        # 移除 input() - 在Colab中無法使用
        print("Test run completed!")
        env.close()

# ===== 主程序 =====

def main():
    """主程序：訓練和評估盲目搜索智能體"""
    
    # 訓練參數
    TOTAL_TIMESTEPS = 1e7
    ALGORITHM = 'PPO'
    
    # 訓練模型
    model, trajectories, progress_callback, model_path = train_blind_search_agent(
        total_timesteps=TOTAL_TIMESTEPS,
        algorithm=ALGORITHM
    )
    
    # 評估模型
    results = evaluate_agent(model, n_eval_episodes=100, render=False)
    
    # 保存結果數據到本地
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
    
    print(f"訓練結果已保存到本地: {results_path}")
    print(f"模型路徑: {model_path}")
    
    return model, trajectories, results, model_path

# ===== 比較不同算法 =====

def compare_algorithms():
    """比較不同RL算法的性能"""
    
    algorithms = ['PPO', 'DQN', 'SAC']
    results = {}
    
    for algo in algorithms:
        print(f"\n{'='*50}")
        print(f"訓練 {algo}")
        print(f"{'='*50}")
        
        model, trajectories, _ = train_blind_search_agent(
            total_timesteps=200000,  # 較少的步數用於快速比較
            algorithm=algo
        )
        
        results[algo] = evaluate_agent(model, n_eval_episodes=50)
        results[algo]['trajectories'] = len(trajectories)
    
    # 可視化比較結果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    algos = list(results.keys())
    success_rates = [results[a]['success_rate'] for a in algos]
    avg_steps = [results[a]['avg_steps'] for a in algos]
    
    ax1.bar(algos, success_rates)
    ax1.set_ylabel('success rate')
    ax1.set_title('algorithm success rate comparison')
    ax1.set_ylim(0, 1)
    
    ax2.bar(algos, avg_steps)
    ax2.set_ylabel('mean steps')
    ax2.set_title('mean steps comparison')
    
    plt.tight_layout()
    plt.show()
    
    return results

if __name__ == "__main__":
    # 運行主程序
    model, trajectories, results, model_path = main()
    
    # 如果想比較不同算法，取消註釋下一行
    # comparison_results = compare_algorithms()