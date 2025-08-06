import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端
import matplotlib.pyplot as plt
import os
from opus_stabline3 import BlindSearchEnv
from stable_baselines3 import PPO

def auto_run_trajectory_analysis():
    """自動運行軌跡分析"""
    
    print("=== 自動軌跡分析 ===")
    
    # 載入模型
    model_path = "blind_search_ppo_final"
    print(f"載入模型: {model_path}")
    
    if os.path.exists(model_path + '.zip'):
        custom_objects = {
            "learning_rate": 3e-4,
            "clip_range": 0.2,
            "lr_schedule": None
        }
        model = PPO.load(model_path, custom_objects=custom_objects)
        print("✅ 模型載入成功!")
    else:
        print("❌ 模型文件不存在!")
        return
    
    # 1. 單軌跡詳細分析
    print("\n=== 1. 單軌跡詳細分析 ===")
    record_and_visualize_trajectory(model)
    
    # 2. 多軌跡比較分析
    print("\n=== 2. 多軌跡比較分析 ===")
    run_multiple_episodes_analysis(model, n_episodes=5)
    
    print("\n✅ 所有分析完成！")
    print("生成的圖片文件:")
    print("- trajectory_visualization.png (單軌跡詳細分析)")
    print("- multiple_trajectories_comparison.png (多軌跡比較)")

def record_and_visualize_trajectory(model):
    """記錄並視覺化模型軌跡"""
    
    # 創建環境
    env = BlindSearchEnv(grid_size=50, sigma=0.05, max_steps=5000)
    obs, _ = env.reset(seed=42)
    
    print(f"目標位置: {env.target_pos}")
    print("開始記錄軌跡...")
    
    # 記錄軌跡數據
    step = 0
    done = False
    trajectory = [env.agent_pos.copy()]
    distances = []
    coverages = []
    
    while not done:
        # 模型預測動作
        action, _ = model.predict(obs, deterministic=True)
        
        # 執行動作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 記錄數據
        trajectory.append(env.agent_pos.copy())
        distances.append(info['distance'])
        coverages.append(info['coverage'])
        
        # 顯示進度
        if step % 100 == 0:
            print(f"Step {step}: 距離目標 {info['distance']:.3f}, 覆蓋率 {info['coverage']:.1%}")
        
        done = terminated or truncated
        step += 1
        
        # 如果找到目標，立即停止
        if terminated and info['success']:
            print(f"🎉 找到目標! 步數: {step}")
            break
    
    # 顯示最終結果
    print(f"總步數: {step}")
    print(f"成功找到目標: {info['success']}")
    print(f"最終距離: {info['distance']:.3f}")
    print(f"覆蓋率: {info['coverage']:.1%}")
    
    # 創建視覺化
    create_trajectory_visualization(trajectory, env.target_pos, distances, coverages, info)
    
    env.close()

def create_trajectory_visualization(trajectory, target_pos, distances, coverages, info):
    """創建軌跡視覺化"""
    
    trajectory = np.array(trajectory)
    
    # 創建多個子圖
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 完整軌跡圖
    ax = axes[0, 0]
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.7, label='Search Trajectory')
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'ro', markersize=10, label='Start Position')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'go', markersize=10, label='Final Position')
    ax.plot(target_pos[0], target_pos[1], 'g*', markersize=15, label='Target Position')
    
    # 繪製目標圓圈
    circle = plt.Circle(target_pos, 0.05, color='green', fill=False, linewidth=2)
    ax.add_patch(circle)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(f'Complete Search Trajectory (Steps: {len(trajectory)-1})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 距離變化圖
    ax = axes[0, 1]
    ax.plot(distances, 'r-', linewidth=2)
    ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='Target Radius')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Distance to Target')
    ax.set_title('Distance to Target Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 標記最接近點
    if info['success']:
        min_dist_idx = np.argmin(distances)
        ax.plot(min_dist_idx, distances[min_dist_idx], 'go', markersize=10, label='Closest Point')
        ax.legend()
    
    # 3. 覆蓋率變化圖
    ax = axes[0, 2]
    ax.plot(coverages, 'purple', linewidth=2)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Coverage')
    ax.set_title('Coverage Over Time')
    ax.grid(True, alpha=0.3)
    
    # 4. 速度分析
    ax = axes[1, 0]
    speeds = []
    for i in range(1, len(trajectory)):
        speed = np.linalg.norm(trajectory[i] - trajectory[i-1])
        speeds.append(speed)
    
    ax.plot(speeds, 'orange', linewidth=2)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Movement Speed')
    ax.set_title('Movement Speed Over Time')
    ax.grid(True, alpha=0.3)
    
    # 5. 方向分析
    ax = axes[1, 1]
    directions = []
    for i in range(1, len(trajectory)):
        direction = np.arctan2(trajectory[i][1] - trajectory[i-1][1], 
                              trajectory[i][0] - trajectory[i-1][0])
        directions.append(direction)
    
    # 將方向轉換為度數並標準化
    directions_deg = np.array(directions) * 180 / np.pi
    directions_deg = (directions_deg + 360) % 360
    
    ax.hist(directions_deg, bins=16, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Movement Direction (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Movement Direction Distribution')
    ax.grid(True, alpha=0.3)
    
    # 6. 軌跡統計
    ax = axes[1, 2]
    stats_data = [
        len(trajectory)-1,  # 總步數
        info['coverage'] * 100,  # 覆蓋率
        min(distances) if distances else 0,  # 最小距離
        max(distances) if distances else 0,  # 最大距離
        np.mean(speeds) if speeds else 0,  # 平均速度
    ]
    stats_labels = ['Total Steps', 'Coverage (%)', 'Min Distance', 'Max Distance', 'Avg Speed']
    
    bars = ax.bar(stats_labels, stats_data, color=['blue', 'green', 'red', 'orange', 'purple'], alpha=0.7)
    ax.set_title('Trajectory Statistics')
    ax.tick_params(axis='x', rotation=45)
    
    # 在柱狀圖上添加數值標籤
    for bar, value in zip(bars, stats_data):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存圖片
    plt.savefig('trajectory_visualization.png', dpi=300, bbox_inches='tight')
    print("✅ 軌跡視覺化已保存為 'trajectory_visualization.png'")
    
    plt.close()

def run_multiple_episodes_analysis(model, n_episodes=5):
    """運行多個episode並分析"""
    
    print(f"運行 {n_episodes} 個episode...")
    
    all_trajectories = []
    all_results = []
    
    for episode in range(n_episodes):
        print(f"Episode {episode + 1}/{n_episodes}")
        
        # 創建環境
        env = BlindSearchEnv(grid_size=50, sigma=0.05, max_steps=5000)
        obs, _ = env.reset(seed=episode)
        
        step = 0
        done = False
        trajectory = [env.agent_pos.copy()]
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append(env.agent_pos.copy())
            done = terminated or truncated
            step += 1
        
        all_trajectories.append(np.array(trajectory))
        all_results.append({
            'episode': episode + 1,
            'steps': step,
            'success': info['success'],
            'target_pos': env.target_pos.copy(),
            'final_distance': info['distance'],
            'coverage': info['coverage']
        })
        
        status = "✅" if info['success'] else "❌"
        print(f"  {status} Steps: {step}, Distance: {info['distance']:.3f}, Coverage: {info['coverage']:.1%}")
        env.close()
    
    # 創建多軌跡比較圖
    create_multiple_trajectories_comparison(all_trajectories, all_results)

def create_multiple_trajectories_comparison(trajectories, results):
    """創建多軌跡比較圖"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 所有軌跡疊加
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
    
    for i, traj in enumerate(trajectories):
        color = colors[i]
        success = results[i]['success']
        linestyle = '-' if success else '--'
        alpha = 0.8 if success else 0.4
        
        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha, 
               linestyle=linestyle, linewidth=2, label=f'Episode {i+1}')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('Multiple Trajectories Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 步數比較
    ax = axes[0, 1]
    steps = [r['steps'] for r in results]
    successes = [r['success'] for r in results]
    
    colors = ['green' if s else 'red' for s in successes]
    bars = ax.bar(range(len(steps)), steps, color=colors, alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Steps Comparison')
    ax.grid(True, alpha=0.3)
    
    # 3. 成功率統計
    ax = axes[1, 0]
    success_count = sum([r['success'] for r in results])
    failure_count = len(results) - success_count
    
    ax.pie([success_count, failure_count], labels=['Success', 'Failure'], 
           colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
    ax.set_title('Success Rate')
    
    # 4. 性能指標
    ax = axes[1, 1]
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    if successful_results:
        successful_steps = [r['steps'] for r in successful_results]
        successful_distances = [r['final_distance'] for r in successful_results]
        
        ax.scatter(successful_steps, successful_distances, c='green', s=100, alpha=0.7, label='Successful')
    
    if failed_results:
        failed_steps = [r['steps'] for r in failed_results]
        failed_distances = [r['final_distance'] for r in failed_results]
        
        ax.scatter(failed_steps, failed_distances, c='red', s=100, alpha=0.7, label='Failed')
    
    ax.set_xlabel('Steps')
    ax.set_ylabel('Final Distance')
    ax.set_title('Steps vs Final Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存圖片
    plt.savefig('multiple_trajectories_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 多軌跡比較圖已保存為 'multiple_trajectories_comparison.png'")
    
    plt.close()

if __name__ == "__main__":
    auto_run_trajectory_analysis() 