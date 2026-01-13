# Robotic Arm Pen Pickup - Reinforcement Learning

This project trains a robotic arm to pick up a pen using **Reinforcement Learning (PPO)** and **PyBullet** simulation.

## 📋 Overview

**Task**: The robot gripper starts **3cm away** from a randomly spawned pen. The goal is to learn a policy that successfully picks up the pen and lifts it above a threshold height (15cm).

**Method**: Proximal Policy Optimization (PPO) - a state-of-the-art RL algorithm for continuous control tasks.

## 🗂️ Project Structure

```
pen_pickup_rl/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── roarm.urdf                   # Robot URDF file
├── meshes/                      # Robot mesh files
│   ├── base_link.STL
│   ├── gripper_link.STL
│   ├── link1.STL
│   ├── link2.STL
│   └── link3.STL
├── pen_pickup_env.py            # Custom Gymnasium environment
├── train.py                     # Training script
├── test.py                      # Testing/visualization script
├── models/                      # Saved models (created during training)
│   ├── best_model.zip
│   ├── vec_normalize.pkl
│   └── ppo_pen_pickup_*.zip
├── logs/                        # Training logs (created during training)
└── videos/                      # Recorded videos (optional)
```

## 🚀 Setup Instructions

### 1. Prerequisites
- Python 3.8 or higher
- VSCode (recommended)
- Virtual environment (recommended)

### 2. Installation

```bash
# Navigate to project directory
cd pen_pickup_rl

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

Test the environment with random actions:

```bash
python pen_pickup_env.py
```

You should see a PyBullet window with the robot arm and a blue pen. The robot will perform random actions.

## 🎯 Training

### Quick Start (Recommended)

Train for 500K timesteps (~2-4 hours depending on your CPU):

```bash
python train.py
```

### Custom Training

```bash
# Train for specific number of timesteps
python train.py --timesteps 1000000

# Specify custom directories
python train.py --save-dir my_models --log-dir my_logs
```

### Training Options

- `--timesteps`: Total training timesteps (default: 500000)
- `--save-dir`: Directory to save models (default: models)
- `--log-dir`: Directory for logs (default: logs)

### Monitoring Training

View training progress in real-time with TensorBoard:

```bash
tensorboard --logdir logs/
```

Open browser and navigate to `http://localhost:6006`

### Training Process

The training script will:
1. Create 4 parallel environments for faster training
2. Save checkpoints every 10,000 steps
3. Evaluate the model every 5,000 steps
4. Save the best model based on evaluation performance
5. Display progress bar and statistics

**Expected Training Time**: 
- 500K steps: ~2-4 hours (CPU)
- 1M steps: ~4-8 hours (CPU)

## 🧪 Testing

### Test Trained Agent

After training, test your agent with visualization:

```bash
# Test best model (default)
python test.py

# Test specific model
python test.py --model models/ppo_pen_pickup_final.zip

# Test for more episodes
python test.py --episodes 20

# Test without rendering (faster)
python test.py --no-render
```

### Test Random Agent (Baseline)

Compare trained agent with random baseline:

```bash
python test.py --random --episodes 5
```

### What to Expect

**Random Agent**:
- Success rate: ~0-5%
- Chaotic, uncoordinated movements
- Rarely makes contact with pen

**Trained Agent** (after 500K steps):
- Success rate: ~60-80% (improves with more training)
- Smooth, coordinated movements
- Approaches pen systematically
- Closes gripper around pen
- Lifts pen smoothly

## 📊 Understanding the System

### Environment Details

**Observation Space** (19 dimensions):
- 4 joint positions
- 4 joint velocities  
- 3 gripper position (x, y, z)
- 3 pen position (x, y, z)
- 3 relative position (gripper to pen)
- 1 gripper opening state
- 1 pen height

**Action Space** (4 dimensions):
- 3 arm joint velocity commands (continuous, -1 to 1)
- 1 gripper control (continuous, -1 to 1 → maps to 0-1.5 rad)

**Reward Function**:
- Distance to pen (negative reward for being far)
- Pen height (positive reward for lifting)
- Contact with pen (bonus reward)
- Success bonus (large reward for lifting above threshold)
- Gripper closing near pen (encourages grasping)

### Why Reinforcement Learning?

For this task, RL (specifically PPO) is the **best choice** because:

1. **Complex Manipulation**: Picking up objects requires learning subtle contact dynamics
2. **No Explicit Solution**: Hard to hand-engineer a controller for all scenarios
3. **Continuous Control**: RL excels at continuous action spaces
4. **Trial and Error**: Robot learns through experience what works
5. **Generalization**: Can adapt to different pen positions

**Alternative Approaches** (and why they're less suitable):

- **Inverse Kinematics (IK)**: Only gives joint positions, doesn't handle grasping dynamics
- **Supervised Learning**: Would need thousands of expert demonstrations
- **Motion Planning**: Difficult to plan contact-rich manipulation
- **PID Control**: Requires manual tuning for each subtask

## 📈 Performance Metrics

During testing, you'll see:

- **Success Rate**: Percentage of successful pen pickups
- **Average Reward**: Higher = better performance
- **Episode Length**: Number of steps to complete (or timeout)
- **Distance to Pen**: How close gripper gets
- **Final Pen Height**: How high the pen is lifted

## 🔧 Troubleshooting

### Common Issues

**1. Import errors**
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt --upgrade
```

**2. PyBullet window not appearing**
- Check that you're running with `render_mode="human"`
- Some systems need X server for GUI

**3. Training is slow**
- Normal on CPU (RL is computationally intensive)
- Consider reducing `--timesteps` for quick tests
- Use fewer parallel environments (edit `train.py`, line 34)

**4. Model not found**
```bash
# Make sure you've trained first
python train.py

# Then test
python test.py
```

**5. Poor performance after training**
- Train for more timesteps (try 1M+)
- Check reward plots in TensorBoard
- May need hyperparameter tuning

## 🎓 Learning Resources

**Understanding RL**:
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

**Stable Baselines3 Documentation**:
- [Getting Started](https://stable-baselines3.readthedocs.io/)
- [PPO Guide](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

**PyBullet**:
- [PyBullet Quickstart](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/)

## 🔬 Experimentation

### Modify Reward Function

Edit `pen_pickup_env.py`, method `_calculate_reward()` to adjust:
- Reward weights
- Add new reward components
- Change success criteria

### Adjust Training Hyperparameters

Edit `train.py` to modify:
- Learning rate
- Network architecture
- Batch size
- Number of parallel environments

### Change Task Difficulty

In `pen_pickup_env.py`:
- Adjust `pickup_height_threshold` (line 68)
- Change `distance_from_pen` (line 67)
- Modify pen spawn range (lines 137-139)

## 📝 Next Steps

After mastering this task:

1. **Increase Difficulty**: Random pen orientations, multiple pens
2. **Add Vision**: Replace state with camera observations
3. **Sim-to-Real**: Transfer learned policy to real robot
4. **Multi-Step Tasks**: Pick up pen and place it somewhere
5. **Integrate Part 1**: Combine with active vision system

## 🤝 Contributing

This is a research/learning project. Feel free to:
- Experiment with different RL algorithms (SAC, TD3)
- Add curriculum learning
- Implement domain randomization
- Share improvements!

## 📄 License

Educational/Research use. See robot manufacturer license for URDF usage.

## ❓ Questions?

Common questions:

**Q: How long should I train?**
A: Start with 500K steps. If success rate < 70%, train for 1M+ steps.

**Q: Can I use GPU?**
A: PyTorch (used by Stable-Baselines3) will automatically use GPU if available. PyBullet physics runs on CPU.

**Q: Why does the robot sometimes fail?**
A: RL agents are probabilistic and may fail occasionally, especially in edge cases.

**Q: Can I use this for a real robot?**
A: This is simulation. Sim-to-real transfer requires domain randomization and real-world fine-tuning.

---

**Happy Training! 🤖🎓**
