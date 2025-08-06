import numpy as np
import os
from opus_stabline3 import BlindSearchEnv
from stable_baselines3 import PPO

def test_model_performance(n_episodes=20):
    """Test model performance across multiple episodes"""
    
    print("=== Blind Search Model Performance Test ===")
    
    # Load model
    model_path = "blind_search_ppo_final"
    print(f"Loading model: {model_path}")
    
    if os.path.exists(model_path + '.zip'):
        custom_objects = {
            "learning_rate": 3e-4,
            "clip_range": 0.2,
            "lr_schedule": None
        }
        model = PPO.load(model_path, custom_objects=custom_objects)
        print("✅ Model loaded successfully!")
    else:
        print("❌ Model file not found!")
        return
    
    results = []
    
    print(f"\nRunning {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        # Create environment
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
        
        results.append({
            'episode': episode + 1,
            'steps': step,
            'success': info['success'],
            'final_distance': info['distance'],
            'coverage': info['coverage'],
            'target_pos': env.target_pos.copy()
        })
        
        # Print progress
        status = "✅" if info['success'] else "❌"
        print(f"Episode {episode+1:2d}: {status} Steps: {step:4d}, Distance: {info['distance']:.3f}, Coverage: {info['coverage']:.1%}")
        
        env.close()
    
    # Analyze results
    analyze_performance(results)
    
    return results

def analyze_performance(results):
    """Analyze performance results"""
    
    print(f"\n{'='*60}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'='*60}")
    
    successes = [r['success'] for r in results]
    steps = [r['steps'] for r in results]
    coverages = [r['coverage'] for r in results]
    distances = [r['final_distance'] for r in results]
    
    # Success statistics
    success_rate = np.mean(successes)
    success_count = sum(successes)
    
    print(f"Success Rate: {success_rate:.1%} ({success_count}/{len(results)})")
    
    # Steps statistics
    successful_steps = [s for s, success in zip(steps, successes) if success]
    failed_steps = [s for s, success in zip(steps, successes) if not success]
    
    if successful_steps:
        print(f"Successful episodes - Avg steps: {np.mean(successful_steps):.1f}, Min: {min(successful_steps)}, Max: {max(successful_steps)}")
    
    if failed_steps:
        print(f"Failed episodes - Avg steps: {np.mean(failed_steps):.1f}, Min: {min(failed_steps)}, Max: {max(failed_steps)}")
    
    # Coverage statistics
    successful_coverages = [c for c, success in zip(coverages, successes) if success]
    failed_coverages = [c for c, success in zip(coverages, successes) if not success]
    
    if successful_coverages:
        print(f"Successful episodes - Avg coverage: {np.mean(successful_coverages):.1%}")
    
    if failed_coverages:
        print(f"Failed episodes - Avg coverage: {np.mean(failed_coverages):.1%}")
    
    # Distance statistics
    successful_distances = [d for d, success in zip(distances, successes) if success]
    failed_distances = [d for d, success in zip(distances, successes) if not success]
    
    if successful_distances:
        print(f"Successful episodes - Avg final distance: {np.mean(successful_distances):.3f}")
    
    if failed_distances:
        print(f"Failed episodes - Avg final distance: {np.mean(failed_distances):.3f}")
    
    # Target position analysis
    successful_targets = [r['target_pos'] for r in results if r['success']]
    failed_targets = [r['target_pos'] for r in results if not r['success']]
    
    if successful_targets:
        successful_targets = np.array(successful_targets)
        print(f"\nSuccessful target positions:")
        print(f"  X range: [{successful_targets[:, 0].min():.3f}, {successful_targets[:, 0].max():.3f}]")
        print(f"  Y range: [{successful_targets[:, 1].min():.3f}, {successful_targets[:, 1].max():.3f}]")
    
    if failed_targets:
        failed_targets = np.array(failed_targets)
        print(f"\nFailed target positions:")
        print(f"  X range: [{failed_targets[:, 0].min():.3f}, {failed_targets[:, 0].max():.3f}]")
        print(f"  Y range: [{failed_targets[:, 1].min():.3f}, {failed_targets[:, 1].max():.3f}]")
    
    # Efficiency analysis
    if successful_steps:
        efficiency = 5000 / np.mean(successful_steps)  # How much faster than max steps
        print(f"\nEfficiency: Model is {efficiency:.1f}x faster than maximum allowed steps")
    
    # Overall assessment
    print(f"\n{'='*60}")
    if success_rate >= 0.8:
        print("🎉 EXCELLENT PERFORMANCE - Model shows strong generalization!")
    elif success_rate >= 0.6:
        print("✅ GOOD PERFORMANCE - Model performs well in most cases")
    elif success_rate >= 0.4:
        print("⚠️  MODERATE PERFORMANCE - Model needs improvement")
    else:
        print("❌ POOR PERFORMANCE - Model needs significant improvement")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_model_performance(n_episodes=20) 