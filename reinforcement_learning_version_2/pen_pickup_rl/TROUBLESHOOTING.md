# Troubleshooting Guide

## Common Issues and Solutions

### 1. Installation Issues

#### Problem: `pip install` fails
```
ERROR: Could not find a version that satisfies the requirement...
```

**Solution:**
```bash
# Update pip first
python -m pip install --upgrade pip

# Install packages one by one to identify the issue
pip install pybullet
pip install gymnasium
pip install stable-baselines3
```

#### Problem: PyTorch installation fails

**Solution:**
```bash
# Visit https://pytorch.org/ and get installation command for your system
# For CPU-only (faster installation):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

### 2. Environment Issues

#### Problem: "FileNotFoundError: roarm.urdf"

**Solution:**
```bash
# Make sure you're running from the pen_pickup_rl directory
cd pen_pickup_rl
python train.py

# Check if URDF file exists
ls roarm.urdf
```

#### Problem: Meshes not loading, weird robot appearance

**Solution:**
```bash
# Verify mesh files exist
ls meshes/

# Should see:
# base_link.STL
# gripper_link.STL
# link1.STL
# link2.STL
# link3.STL
```

#### Problem: PyBullet GUI not showing

**Solutions:**
1. Check render mode:
   ```python
   env = PenPickupEnv(render_mode="human")  # Should show GUI
   ```

2. On Linux, you may need X server:
   ```bash
   export DISPLAY=:0
   ```

3. On WSL2 (Windows Subsystem for Linux):
   ```bash
   # Install VcXsrv or similar X server on Windows
   export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
   ```

---

### 3. Training Issues

#### Problem: Training is very slow

**Normal Behavior:**
- RL training is computationally intensive
- 500K steps takes 2-4 hours on modern CPU
- GPU doesn't help much (physics simulation is on CPU)

**Solutions to speed up:**
1. Reduce timesteps for testing:
   ```bash
   python train.py --timesteps 50000
   ```

2. Use fewer parallel environments (edit train.py line 34):
   ```python
   n_envs=2  # Instead of 4
   ```

3. Reduce network size (edit train.py):
   ```python
   policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
   ```

#### Problem: "CUDA out of memory"

**Solution:**
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""

# Or reduce batch size in train.py
batch_size=32  # Instead of 64
```

#### Problem: Training crashes / freezes

**Solutions:**
1. Check RAM usage (need ~4-8GB free)
2. Reduce parallel environments
3. Check for system updates
4. Try running without TensorBoard logging

---

### 4. Testing Issues

#### Problem: "Model not found"

**Solution:**
```bash
# Make sure you've trained first
python train.py --timesteps 50000

# Check if model exists
ls models/best_model.zip

# If not, use checkpoint instead
python test.py --model models/ppo_pen_pickup_10000_steps.zip
```

#### Problem: Trained agent performs poorly

**Possible Causes:**
1. **Not enough training** - Train for longer (1M+ timesteps)
2. **Reward function issues** - Check TensorBoard for reward trends
3. **Hyperparameter issues** - Try default settings first

**Solution:**
```bash
# Check training progress in TensorBoard
tensorboard --logdir logs/

# Look for:
# - Increasing episode rewards
# - Increasing episode lengths (initially)
# - Decreasing episode lengths (after learning)
```

---

### 5. Simulation Issues

#### Problem: Robot behaves strangely / falls through floor

**Solution:**
1. Check URDF file is correct
2. Verify mesh files are in correct location
3. Try resetting simulation:
   ```python
   env.reset()
   ```

#### Problem: Pen flies away / physics glitches

**Solution:**
- This can happen with random actions
- Normal during early training
- Should improve as agent learns
- If persistent, check friction settings in environment

---

### 6. Performance Issues

#### Problem: Low success rate after training

**Expected Performance:**
- Random agent: 0-5%
- 100K steps: 20-40%
- 500K steps: 60-80%
- 1M steps: 80-90%

**If below expected:**
1. Train longer
2. Check reward trends in TensorBoard
3. Verify environment rewards are balanced
4. Try different hyperparameters (config.py)

#### Problem: Agent gets stuck in local minimum

**Solutions:**
1. Increase entropy coefficient:
   ```python
   ent_coef=0.02  # More exploration
   ```

2. Use different random seed:
   ```python
   python train.py  # Will use different seed
   ```

3. Adjust reward shaping in env

---

### 7. System-Specific Issues

#### Windows Issues

**Problem: Long paths**
```bash
# Use shorter directory names
cd C:\Users\YourName\rl
```

**Problem: Permission errors**
```bash
# Run as administrator or use different directory
```

#### Mac Issues

**Problem: PyBullet GUI rendering issues**
```bash
# Try different rendering backend
export PYOPENGL_PLATFORM=osmesa
```

#### Linux Issues

**Problem: Missing GL libraries**
```bash
# Install OpenGL libraries
sudo apt-get install python3-opengl
sudo apt-get install freeglut3-dev
```

---

### 8. Advanced Debugging

#### Enable verbose logging

```python
# In train.py, change:
verbose=1  # to
verbose=2  # More detailed output
```

#### Check environment manually

```python
from pen_pickup_env import PenPickupEnv

env = PenPickupEnv(render_mode="human")
obs, info = env.reset()

print("Observation:", obs)
print("Info:", info)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

# Take some steps
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    print(f"Reward: {reward:.2f}, Distance: {info['distance_to_pen']:.4f}")
```

#### Monitor system resources

```bash
# Check CPU/RAM usage during training
# On Linux:
htop

# On Windows:
# Use Task Manager

# On Mac:
# Use Activity Monitor
```

---

## Getting Help

If you're still stuck:

1. **Check error messages carefully** - Often contain the solution
2. **Run setup_check.py** - Verifies basic setup
3. **Try demo.py first** - Simpler than full training
4. **Search error message** - Someone likely had same issue
5. **Check Stable-Baselines3 docs** - https://stable-baselines3.readthedocs.io/

---

## Useful Commands Summary

```bash
# Verify setup
python setup_check.py

# Quick demo
python demo.py

# Test environment only
python pen_pickup_env.py

# Quick training test
python train.py --timesteps 10000

# Full training
python train.py

# Monitor training
tensorboard --logdir logs/

# Test trained model
python test.py

# Test with more episodes
python test.py --episodes 20

# Compare to random baseline
python test.py --random
```

---

## Still Having Issues?

1. Make sure you're in the correct directory
2. Virtual environment is activated
3. All dependencies are installed
4. Python version is 3.8+
5. Sufficient RAM available (4GB+ free)

Run the full diagnostic:
```bash
python setup_check.py
```
