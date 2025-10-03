# Blind Search with Reinforcement Learning

A beginner-friendly project demonstrating how Reinforcement Learning (RL) can learn intelligent search strategies without knowing where the target is located.

## What Does This Project Do?

Imagine you lost your keys in a dark room. You can't see them, so you need a strategy to search the room efficiently. This project:

1. **Simulates blind search scenarios** - An agent searches for a hidden target in a 2D grid without knowing its location
2. **Trains AI agents using Reinforcement Learning** - The AI learns effective search patterns through trial and error
3. **Compares different approaches** - Shows how RL agents perform against traditional search methods

## Quick Example

```
Grid (50x50):
┌─────────────────────┐
│ Agent starts here → ◉│
│                     │
│         🎯 ← Hidden │
│            target   │
│                     │
└─────────────────────┘

Goal: Find the target 🎯 as quickly as possible!
```

## How Does It Work?

### The Environment

- **Grid Size**: 50x50 cells
- **Agent**: Starts at bottom-left corner (0,0)
- **Target**: Randomly placed (unknown to agent)
- **Actions**: Move in 8 directions (↑ ↗ → ↘ ↓ ↙ ← ↖)
- **Success**: When agent gets within 0.05 units of target

### The Agent's Observations

The agent can see:
- Its current position
- Which areas it has already visited (memory map)
- How much time has passed
- Its last movement direction

**Note:** The agent CANNOT see where the target is!

---

## Reinforcement Learning vs Traditional Methods

### What is Reinforcement Learning?

Think of it like training a dog:
- Dog does something → Gets reward/punishment → Learns what works
- RL agent tries actions → Gets reward/punishment → Learns optimal strategy

### Traditional Methods (Rule-Based)

**Examples in this project:**
1. **Archimedean Spiral** - Moves in expanding circular pattern
2. **Logarithmic Spiral** - Grows exponentially outward
3. **Sine Wave** - Sweeps back and forth

**How they work:**
```python
# Traditional: Follow a mathematical formula
position = center + radius * cos(angle)
```

### Reinforcement Learning Method

**How it works:**
```python
# RL: Learn from experience
observation = env.get_state()
action = agent.decide(observation)  # Learned from 10M+ trials!
reward = env.step(action)
agent.learn(reward)  # Gets better over time
```

---

## Comparison: RL vs Traditional

### Traditional Methods

**Pros:**
- ✓ Simple to understand and implement
- ✓ Predictable behavior
- ✓ No training required
- ✓ Works immediately
- ✓ Good for known patterns

**Cons:**
- ✗ Fixed strategy (can't adapt)
- ✗ May be inefficient for certain target locations
- ✗ Doesn't learn from experience
- ✗ Same pattern every time

### Reinforcement Learning

**Pros:**
- ✓ Learns optimal strategy through experience
- ✓ Adapts to different situations
- ✓ Can discover creative solutions
- ✓ Improves with more training
- ✓ Remembers visited areas efficiently

**Cons:**
- ✗ Requires lots of training (millions of steps)
- ✗ Needs computational resources (GPU helpful)
- ✗ Results may vary between training runs
- ✗ Harder to understand why it works

---

## Project Structure

### Simple Version (All-in-One)
```
opus_stabline3.py          # Everything in one file
```

### Modular Version (Organized)
```
environment.py             # The search grid environment
callbacks.py               # Training progress tracking
training.py                # RL agent training
evaluation.py              # Performance testing
visualization.py           # Results visualization
main.py                    # Run everything
```

### Algorithm Comparison
```
algo_compare_fixed.py      # Compare RL vs Traditional methods
```

---

## Getting Started

### 1. Installation

```bash
# Clone or download this project
cd project2

# Install required packages
pip install -r requirements.txt

# Optional: For GPU support, install PyTorch with CUDA
# Visit https://pytorch.org/get-started/locally/ for instructions
```

**Dependencies:**
- Python 3.8 or higher recommended
- numpy - Numerical computing
- matplotlib - Visualization
- gymnasium - RL environment framework
- stable-baselines3 - RL algorithms (PPO, DQN, SAC)
- torch - Deep learning backend
- tensorboard - Training monitoring (optional)

### 2. Train an RL Agent

```python
# Option A: Use simple version
python opus_stabline3.py

# Option B: Use modular version
python main.py
```

### 3. Compare Algorithms

```python
# Compare RL vs Traditional methods
python algo_compare_fixed.py
```

This will:
- Train a PPO agent
- Test Archimedean Spiral, Logarithmic Spiral, and Sine Wave patterns
- Generate comparison charts in `fig/` folder

---

## Understanding the Results

### Success Metrics

1. **Success Rate** - How often the agent finds the target (higher is better)
2. **Average Steps** - How many moves it takes (lower is better)
3. **Coverage** - Percentage of area explored (higher can be better)
4. **Path Length** - Total distance traveled (lower is better)

### Typical Results

After training 10 million steps:
```
PPO Agent:              Success: ~60-80%, Steps: ~2000
Archimedean Spiral:     Success: ~40-50%, Steps: ~2500
Logarithmic Spiral:     Success: ~30-40%, Steps: ~2800
Sine Wave:              Success: ~20-30%, Steps: ~3000
```

---

## Key RL Concepts (Simplified)

### 1. State/Observation
What the agent "sees": position, visited areas, time

### 2. Action
What the agent can "do": move in 8 directions

### 3. Reward
Feedback for actions:
- **Big reward** (+1000): Found the target! 
- **Small reward** (+5): Explored new area
- **Small penalty** (-1): Each step taken

### 4. Policy
The agent's strategy (what action to take in each situation)

### 5. Training
The agent tries millions of searches, gradually learning what works

---

## Algorithms Used

### PPO (Proximal Policy Optimization)
- **Type**: On-policy RL algorithm
- **Best for**: Continuous learning with stable updates
- **Training time**: ~1-2 hours for 10M steps

### DQN (Deep Q-Network)
- **Type**: Off-policy RL algorithm  
- **Best for**: Discrete action spaces
- **Training time**: ~1-2 hours for 10M steps

### SAC (Soft Actor-Critic)
- **Type**: Off-policy RL algorithm
- **Best for**: Exploration-exploitation balance
- **Training time**: ~1-2 hours for 10M steps

---

## Visualization Examples

The project generates visualizations showing:

1. **Trajectory Plots** - The path the agent takes
2. **Coverage Heatmaps** - Which areas were visited most
3. **Success Distributions** - Performance across different targets
4. **Algorithm Comparisons** - Side-by-side performance

All saved in the `fig/` directory.

---

## Customization

### Change Environment Size
```python
env = BlindSearchEnv(
    grid_size=100,      # Larger search area
    sigma=0.03,         # Smaller target
    max_steps=10000     # More time allowed
)
```

### Adjust Training
```python
train_blind_search_agent(
    total_timesteps=5e6,    # Less training
    algorithm='PPO'         # Or 'DQN', 'SAC'
)
```

---

## Common Questions

**Q: How long does training take?**  
A: About 1-2 hours for 10 million steps (with GPU), longer on CPU.

**Q: Do I need a GPU?**  
A: No, but it's 5-10x faster with one.

**Q: Can I use the trained model?**  
A: Yes! Load with: `model = PPO.load("blind_search_ppo_final")`

**Q: Why doesn't the agent always win?**  
A: Blind search is hard! Even humans struggle finding things without any clues.

**Q: How is this useful in real life?**  
A: Applications include:
- Robot vacuum cleaners (room coverage)
- Search and rescue operations
- Resource exploration
- Network packet routing

---

## Performance Tips

1. **Use GPU** - 5-10x faster training
2. **Start small** - Test with 100k steps first
3. **Monitor progress** - Check TensorBoard logs
4. **Save checkpoints** - Don't lose training progress
5. **Compare fairly** - Use same number of steps for all methods

---

## File Outputs

After running, you'll find:

```
requirements.txt                        # Package dependencies
fig/                                    # Visualization images
blind_search_best_model/                # Best model checkpoint
blind_search_logs/                      # Evaluation logs
blind_search_tensorboard/               # Training metrics
blind_search_ppo_final.zip             # Trained model
training_results.pkl                    # Results data
```

---

## Next Steps

1. **Try different algorithms** - Uncomment `compare_algorithms()` in `main.py`
2. **Experiment with rewards** - Modify `_calculate_reward()` in `environment.py`
3. **Visualize learning** - Use TensorBoard: `tensorboard --logdir blind_search_tensorboard/`
4. **Test your own strategies** - Create new search patterns in `algo_compare_fixed.py`

---

## Learning Resources

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [RL Introduction (Spinning Up)](https://spinningup.openai.com/)

---

## Credits

Built with:
- **Stable-Baselines3** - RL algorithms
- **Gymnasium** - RL environment framework
- **PyTorch** - Deep learning backend
- **Matplotlib** - Visualization

---

## License

This project is for educational purposes. Feel free to modify and learn from it!

---

## Summary

This project demonstrates that:
1. **RL can learn complex strategies** without explicit programming
2. **Experience matters** - more training = better performance
3. **Adaptation is powerful** - RL adjusts to different situations
4. **Trade-offs exist** - RL is powerful but requires training time

The key insight: Instead of telling the computer HOW to search, we teach it to discover effective strategies on its own through trial and error!


