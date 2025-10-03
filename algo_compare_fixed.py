# Fixed Algorithm Comparison Tool - Resolves Image Size Inconsistency and Fairness Issues
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from environment import BlindSearchEnv
from stable_baselines3 import PPO
import time
from abc import ABC, abstractmethod

# Unified figure configuration
FIGURE_SIZE = (16, 10)  # Unified figure size for all plots
DPI = 300
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Unified color scheme

class BaseSearchAlgorithm(ABC):
    """Base search algorithm abstract class"""
    
    def __init__(self, name: str, step_size: float = 0.02):
        self.name = name
        self.step_size = step_size  # Step distance for each move (for fair time comparison)
    
    @abstractmethod
    def search(self, env: BlindSearchEnv) -> dict:
        """Execute search and return results"""
        pass
    
    def calculate_path_length(self, trajectory):
        """Calculate actual path length"""
        if len(trajectory) < 2:
            return 0.0
        
        path_length = 0.0
        for i in range(1, len(trajectory)):
            distance = np.linalg.norm(trajectory[i] - trajectory[i-1])
            path_length += distance
        return path_length

class PPOAgent(BaseSearchAlgorithm):
    """PPO Agent"""
    
    def __init__(self):
        super().__init__("PPO Agent", step_size=0.02)
        self.model = self._load_model()
    
    def _load_model(self):
        model_path = "blind_search_ppo_final"
        if os.path.exists(model_path + '.zip'):
            custom_objects = {
                "learning_rate": 3e-4,
                "clip_range": 0.2,
                "lr_schedule": None
            }
            return PPO.load(model_path, custom_objects=custom_objects)
        else:
            raise FileNotFoundError("PPO model not found!")
    
    def search(self, env: BlindSearchEnv) -> dict:
        obs, _ = env.reset()
        steps = 0
        trajectory = [env.agent_pos.copy()]
        
        while steps < env.max_steps:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append(env.agent_pos.copy())
            steps += 1
            
            if terminated:
                break
        
        trajectory = np.array(trajectory)
        path_length = self.calculate_path_length(trajectory)
        # Calculate time based on path length (assuming constant speed)
        travel_time = path_length / self.step_size if self.step_size > 0 else steps
        
        return {
            'steps': steps,
            'success': info['success'],
            'final_distance': info['distance'],
            'coverage': info['coverage'],
            'trajectory': trajectory,
            'path_length': path_length,
            'travel_time': travel_time
        }

class LogarithmicSpiralSearch(BaseSearchAlgorithm):
    """Logarithmic Spiral Search"""
    
    def __init__(self, a: float = 0.1, b: float = 0.2):
        super().__init__("Logarithmic Spiral", step_size=0.02)
        self.a = a
        self.b = b
    
    def search(self, env: BlindSearchEnv) -> dict:
        obs, _ = env.reset()
        steps = 0
        trajectory = [env.agent_pos.copy()]
        theta = 0
        
        while steps < env.max_steps:
            r = self.a * np.exp(self.b * theta)
            target_x = 0.5 + r * np.cos(theta)
            target_y = 0.5 + r * np.sin(theta)
            
            target_x = np.clip(target_x, 0, 1)
            target_y = np.clip(target_y, 0, 1)
            
            current_pos = env.agent_pos
            direction = np.array([target_x, target_y]) - current_pos
            action = self._get_best_action(direction)
            
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append(env.agent_pos.copy())
            steps += 1
            theta += 0.1
            
            if terminated:
                break
        
        trajectory = np.array(trajectory)
        path_length = self.calculate_path_length(trajectory)
        travel_time = path_length / self.step_size if self.step_size > 0 else steps
        
        return {
            'steps': steps,
            'success': info['success'],
            'final_distance': info['distance'],
            'coverage': info['coverage'],
            'trajectory': trajectory,
            'path_length': path_length,
            'travel_time': travel_time
        }
    
    def _get_best_action(self, direction):
        directions = np.array([
            [1, 0], [1, 1], [0, 1], [-1, 1],
            [-1, 0], [-1, -1], [0, -1], [1, -1]
        ], dtype=np.float32)
        
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / norms
        
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        
        dots = np.dot(directions, direction)
        return np.argmax(dots)

class ArchimedeanSpiralSearch(BaseSearchAlgorithm):
    """Archimedean Spiral Search"""
    
    def __init__(self, a: float = 0.05):
        super().__init__("Archimedean Spiral", step_size=0.02)
        self.a = a
    
    def search(self, env: BlindSearchEnv) -> dict:
        obs, _ = env.reset()
        steps = 0
        trajectory = [env.agent_pos.copy()]
        theta = 0
        
        while steps < env.max_steps:
            r = self.a * theta
            target_x = 0.5 + r * np.cos(theta)
            target_y = 0.5 + r * np.sin(theta)
            
            target_x = np.clip(target_x, 0, 1)
            target_y = np.clip(target_y, 0, 1)
            
            current_pos = env.agent_pos
            direction = np.array([target_x, target_y]) - current_pos
            action = self._get_best_action(direction)
            
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append(env.agent_pos.copy())
            steps += 1
            theta += 0.1
            
            if terminated:
                break
        
        trajectory = np.array(trajectory)
        path_length = self.calculate_path_length(trajectory)
        travel_time = path_length / self.step_size if self.step_size > 0 else steps
        
        return {
            'steps': steps,
            'success': info['success'],
            'final_distance': info['distance'],
            'coverage': info['coverage'],
            'trajectory': trajectory,
            'path_length': path_length,
            'travel_time': travel_time
        }
    
    def _get_best_action(self, direction):
        directions = np.array([
            [1, 0], [1, 1], [0, 1], [-1, 1],
            [-1, 0], [-1, -1], [0, -1], [1, -1]
        ], dtype=np.float32)
        
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / norms
        
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        
        dots = np.dot(directions, direction)
        return np.argmax(dots)

class SinWaveSearch(BaseSearchAlgorithm):
    """Sine Wave Search"""
    
    def __init__(self, amplitude: float = 0.3, frequency: float = 2.0):
        super().__init__("Sin Wave Search", step_size=0.02)
        self.amplitude = amplitude
        self.frequency = frequency
    
    def search(self, env: BlindSearchEnv) -> dict:
        obs, _ = env.reset()
        steps = 0
        trajectory = [env.agent_pos.copy()]
        t = 0
        
        while steps < env.max_steps:
            progress = t / 100.0
            target_x = progress % 1.0
            target_y = 0.5 + self.amplitude * np.sin(self.frequency * 2 * np.pi * progress)
            
            target_x = np.clip(target_x, 0, 1)
            target_y = np.clip(target_y, 0, 1)
            
            current_pos = env.agent_pos
            direction = np.array([target_x, target_y]) - current_pos
            action = self._get_best_action(direction)
            
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append(env.agent_pos.copy())
            steps += 1
            t += 1
            
            if terminated:
                break
        
        trajectory = np.array(trajectory)
        path_length = self.calculate_path_length(trajectory)
        travel_time = path_length / self.step_size if self.step_size > 0 else steps
        
        return {
            'steps': steps,
            'success': info['success'],
            'final_distance': info['distance'],
            'coverage': info['coverage'],
            'trajectory': trajectory,
            'path_length': path_length,
            'travel_time': travel_time
        }
    
    def _get_best_action(self, direction):
        directions = np.array([
            [1, 0], [1, 1], [0, 1], [-1, 1],
            [-1, 0], [-1, -1], [0, -1], [1, -1]
        ], dtype=np.float32)
        
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / norms
        
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        
        dots = np.dot(directions, direction)
        return np.argmax(dots)

def visualize_algorithm_pattern(ax, algorithm):
    """Visualize algorithm's theoretical search pattern (mathematical formula pattern)"""
    
    if isinstance(algorithm, LogarithmicSpiralSearch):
        # Logarithmic spiral: r = a * e^(b*θ)
        theta = np.linspace(0, 4*np.pi, 200)
        r = algorithm.a * np.exp(algorithm.b * theta)
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta)
        
        valid_mask = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
        x = x[valid_mask]
        y = y[valid_mask]
        
        ax.plot(x, y, color=COLORS[1], linewidth=3, alpha=0.8)
        ax.set_title(f'r = {algorithm.a} × e^({algorithm.b}θ)', fontsize=14, fontweight='bold')
        
    elif isinstance(algorithm, ArchimedeanSpiralSearch):
        # Archimedean spiral: r = a * θ
        theta = np.linspace(0, 10*np.pi, 500)
        r = algorithm.a * theta
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta)
        
        valid_mask = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
        x = x[valid_mask]
        y = y[valid_mask]
        
        ax.plot(x, y, color=COLORS[2], linewidth=3, alpha=0.8)
        ax.set_title(f'r = {algorithm.a} × θ', fontsize=14, fontweight='bold')
        
    elif isinstance(algorithm, SinWaveSearch):
        # Sine wave: y = A*sin(f*t)
        t = np.linspace(0, 2, 200)
        x = t % 1.0
        y = 0.5 + algorithm.amplitude * np.sin(algorithm.frequency * 2 * np.pi * t)
        y = np.clip(y, 0, 1)
        
        ax.plot(x, y, color=COLORS[3], linewidth=3, alpha=0.8)
        ax.set_title(f'y = 0.5 + {algorithm.amplitude} × sin({algorithm.frequency} × 2π × t)', 
                    fontsize=14, fontweight='bold')
        
    else:  # PPO Agent
        # Display neural network learned policy (abstract representation)
        ax.text(0.5, 0.5, 'Neural Network\nLearned Policy', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS[0], alpha=0.3))
        ax.set_title('PPO Deep RL Policy', fontsize=14, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')

def create_clean_trajectory_visualization(algorithm, n_episodes=3):
    """Create clean trajectory visualization - each episode displayed independently + mathematical pattern"""
    
    print(f"Generating clean trajectory plot for {algorithm.name}...")
    
    episodes_data = []
    
    for episode in range(n_episodes):
        env = BlindSearchEnv(grid_size=50, sigma=0.05, max_steps=5000)
        result = algorithm.search(env)
        
        episodes_data.append({
            'trajectory': result['trajectory'],
            'success': result['success'],
            'target_pos': env.target_pos.copy(),
            'path_length': result['path_length'],
            'travel_time': result['travel_time'],
            'steps': result['steps']
        })
        
        env.close()
    
    # Create 2x2 layout: 3 episodes + 1 pattern
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    
    # Color scheme
    episode_colors = [COLORS[0], COLORS[1], COLORS[2]]  # Use different colors for each episode
    
    # First three subplots: each episode displayed independently
    for i, episode_data in enumerate(episodes_data):
        row, col = divmod(i, 2)  # Calculate subplot position
        ax = axes[row, col]
        
        trajectory = episode_data['trajectory']
        success = episode_data['success']
        target_pos = episode_data['target_pos']
        
        # Choose line style and transparency based on success
        linestyle = '-' if success else '--'
        alpha = 0.9 if success else 0.7
        linewidth = 3 if success else 2
        color = episode_colors[i]
        
        # Draw trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], 
               color=color, linestyle=linestyle, alpha=alpha, 
               linewidth=linewidth)
        
        # Start point (circle)
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'o', 
               color=color, markersize=12, markeredgecolor='white', 
               markeredgewidth=2, label='Start')
        
        # End point (square)
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 's', 
               color=color, markersize=12, markeredgecolor='white', 
               markeredgewidth=2, label='End')
        
        # Target area (circle)
        circle = plt.Circle(target_pos, 0.05, color='red', 
                           fill=False, linewidth=3, alpha=0.8)
        ax.add_patch(circle)
        
        # Target point (star)
        ax.plot(target_pos[0], target_pos[1], '*', 
               color='red', markersize=20, markeredgecolor='white', 
               markeredgewidth=1, label='Target')
        
        # Set subplot properties
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Title includes episode information
        status = "✓ Success" if success else "✗ Failed"
        ax.set_title(f'Episode {i+1} - {status}\nSteps: {episode_data["steps"]}, Path: {episode_data["path_length"]:.2f}', 
                    fontsize=12, fontweight='bold')
        
        # Only show x-axis label on bottom row
        if row == 1:
            ax.set_xlabel('X position', fontsize=10)
        
        # Only show y-axis label on leftmost column
        if col == 0:
            ax.set_ylabel('Y position', fontsize=10)
        
        # Add legend for the first episode
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    # Fourth subplot: mathematical pattern
    ax_pattern = axes[1, 1]
    visualize_algorithm_pattern(ax_pattern, algorithm)
    
    # Adjust overall title
    fig.suptitle(f'{algorithm.name} - Individual Episodes + Mathematical Pattern', 
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Leave space for main title
    
    # Save figure to fig directory
    import os
    os.makedirs('fig', exist_ok=True)
    filename = f'fig/clean_{algorithm.name.lower().replace(" ", "_")}_trajectory.png'
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    print(f"[OK] Clean trajectory plot saved as '{filename}'")
    plt.close()
    
    return episodes_data

def compare_algorithms_fair(n_trials: int = 100):
    """Fair comparison of algorithm performance - time calculation based on path length"""
    
    print(f"=== Fair Search Algorithm Performance Comparison (n_trials={n_trials}) ===\n")
    
    algorithms = [
        PPOAgent(),
        LogarithmicSpiralSearch(a=0.1, b=0.2),
        ArchimedeanSpiralSearch(a=0.05),
        SinWaveSearch(amplitude=0.3, frequency=2.0)
    ]
    
    results = {}
    
    for algorithm in algorithms:
        print(f"Testing {algorithm.name}...")
        
        algorithm_results = {
            'steps': [],
            'successes': [],
            'distances': [],
            'coverages': [],
            'path_lengths': [],
            'travel_times': []  # Fair time based on path length
        }
        
        for trial in range(n_trials):
            if (trial + 1) % 50 == 0:
                print(f"  Progress: {trial + 1}/{n_trials}")
            
            env = BlindSearchEnv(grid_size=50, sigma=0.05, max_steps=5000)
            result = algorithm.search(env)
            
            algorithm_results['steps'].append(result['steps'])
            algorithm_results['successes'].append(result['success'])
            algorithm_results['distances'].append(result['final_distance'])
            algorithm_results['coverages'].append(result['coverage'])
            algorithm_results['path_lengths'].append(result['path_length'])
            algorithm_results['travel_times'].append(result['travel_time'])
            
            env.close()
        
        results[algorithm.name] = algorithm_results
        
        # Calculate statistics
        success_rate = np.mean(algorithm_results['successes'])
        avg_steps = np.mean(algorithm_results['steps'])
        avg_distance = np.mean(algorithm_results['distances'])
        avg_coverage = np.mean(algorithm_results['coverages'])
        avg_path_length = np.mean(algorithm_results['path_lengths'])
        avg_travel_time = np.mean(algorithm_results['travel_times'])
        
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Average steps: {avg_steps:.1f}")
        print(f"  Average path length: {avg_path_length:.3f}")
        print(f"  Average travel time: {avg_travel_time:.1f}")
        print(f"  Average final distance: {avg_distance:.3f}")
        print(f"  Average coverage: {avg_coverage:.1%}\n")
    
    create_fair_comparison_plot(results)
    return results

def create_fair_comparison_plot(results):
    """Create fair comparison charts"""
    
    algorithms = list(results.keys())
    
    # Create uniform-sized charts
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    
    # 1. Success rate comparison
    ax = axes[0, 0]
    success_rates = [np.mean(results[algo]['successes']) for algo in algorithms]
    bars = ax.bar(range(len(algorithms)), success_rates, color=COLORS, alpha=0.8)
    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate Comparison', fontweight='bold')
    ax.set_ylim(0, 1)
    
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Fair time comparison (based on path length)
    ax = axes[0, 1]
    avg_travel_times = [np.mean(results[algo]['travel_times']) for algo in algorithms]
    bars = ax.bar(range(len(algorithms)), avg_travel_times, color=COLORS, alpha=0.8)
    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.set_ylabel('Travel Time (normalized)')
    ax.set_title('Fair Time Comparison\n(Path Length / Speed)', fontweight='bold')
    
    for i, (bar, time_val) in enumerate(zip(bars, avg_travel_times)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_travel_times)*0.01, 
                f'{time_val:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Path length comparison
    ax = axes[1, 0]
    avg_path_lengths = [np.mean(results[algo]['path_lengths']) for algo in algorithms]
    bars = ax.bar(range(len(algorithms)), avg_path_lengths, color=COLORS, alpha=0.8)
    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.set_ylabel('Average Path Length')
    ax.set_title('Path Length Comparison', fontweight='bold')
    
    for i, (bar, length) in enumerate(zip(bars, avg_path_lengths)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_path_lengths)*0.01, 
                f'{length:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Efficiency score (success rate / travel time)
    ax = axes[1, 1]
    efficiency_scores = []
    for algo in algorithms:
        success_rate = np.mean(results[algo]['successes'])
        avg_time = np.mean(results[algo]['travel_times'])
        # Avoid division by zero
        efficiency = success_rate / max(avg_time, 1) * 1000  # Multiply by 1000 for readability
        efficiency_scores.append(efficiency)
    
    bars = ax.bar(range(len(algorithms)), efficiency_scores, color=COLORS, alpha=0.8)
    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.set_ylabel('Efficiency Score\n(Success Rate / Travel Time × 1000)')
    ax.set_title('Algorithm Efficiency', fontweight='bold')
    
    for i, (bar, score) in enumerate(zip(bars, efficiency_scores)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(efficiency_scores)*0.01, 
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save to fig directory
    import os
    os.makedirs('fig', exist_ok=True)
    plt.savefig('fig/fair_algorithm_comparison.png', dpi=DPI, bbox_inches='tight')
    print("[OK] Fair comparison chart saved as 'fig/fair_algorithm_comparison.png'")
    plt.close()

def run_clean_analysis():
    """Run clean analysis - generate only trajectories and mathematical patterns"""
    
    print("=== Clean Algorithm Analysis ===")
    print("Generated content:")
    print("1. Trajectory visualization")
    print("2. Mathematical formula patterns")
    print("3. Fair performance comparison")
    print()
    
    algorithms = [
        PPOAgent(),
        LogarithmicSpiralSearch(a=0.1, b=0.2),
        ArchimedeanSpiralSearch(a=0.05),
        SinWaveSearch(amplitude=0.3, frequency=2.0)
    ]
    
    print("Step 1: Generating clean trajectory visualizations...")
    for algorithm in algorithms:
        create_clean_trajectory_visualization(algorithm)
    
    print("\nStep 2: Running fair performance comparison...")
    results = compare_algorithms_fair(n_trials=100)
    
    print(f"\n[DONE] Clean analysis complete!")
    print("Generated files (saved in fig/ directory):")
    print("  - fig/clean_ppo_agent_trajectory.png - PPO clean trajectory plot")
    print("  - fig/clean_logarithmic_spiral_trajectory.png - Logarithmic spiral clean trajectory plot")
    print("  - fig/clean_archimedean_spiral_trajectory.png - Archimedean spiral clean trajectory plot")
    print("  - fig/clean_sin_wave_search_trajectory.png - Sine wave clean trajectory plot")
    print("  - fig/fair_algorithm_comparison.png - Fair performance comparison chart")
    
    return results

if __name__ == "__main__":
    print("=== Fixed Search Algorithm Comparison Tool ===")
    print("Fixed content:")
    print("[OK] Unified image size")
    print("[OK] Simplified visualization (keep only trajectories and mathematical patterns)")
    print("[OK] Fair time calculation (path length / movement speed)")
    print()
    
    run_clean_analysis()