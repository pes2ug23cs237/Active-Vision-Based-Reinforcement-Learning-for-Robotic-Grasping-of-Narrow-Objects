# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies (5 minutes)

```bash
# Make sure you're in the pen_pickup_rl directory
cd pen_pickup_rl

# Install required packages
pip install -r requirements.txt
```

### Step 2: Verify Setup (2 minutes)

```bash
# Run setup verification
python setup_check.py
```

You should see all checks passing ✓

### Step 3: Run Demo (Optional - 2 minutes)

```bash
# See the environment and task
python demo.py
```

Press Ctrl+C to stop. You'll see the robot with random actions (untrained).

---

## 📚 Main Workflows

### Training a Model

```bash
# Train for 500K steps (~2-4 hours)
python train.py

# Monitor progress
tensorboard --logdir logs/
```

### Testing Trained Model

```bash
# Test with visualization
python test.py

# Test more episodes
python test.py --episodes 20
```

### Compare with Random Baseline

```bash
python test.py --random --episodes 5
```

---

## 🎯 Expected Results

| Agent Type | Success Rate | Training Time |
|-----------|--------------|---------------|
| Random | 0-5% | - |
| Trained (500K steps) | 60-80% | 2-4 hours |
| Trained (1M steps) | 80-90% | 4-8 hours |

---

## 🔍 Key Files

- **pen_pickup_env.py** - Environment definition
- **train.py** - Training script
- **test.py** - Testing script
- **demo.py** - Quick visualization
- **roarm.urdf** - Robot model

---

## 💡 Pro Tips

1. **Use TensorBoard** to monitor training in real-time
2. **Start with short training runs** to verify everything works
3. **Save your best models** - they're in `models/best_model.zip`
4. **Experiment with hyperparameters** in `train.py` after getting baseline results

---

## ❓ Troubleshooting

**Problem: "Module not found"**
```bash
pip install -r requirements.txt --upgrade
```

**Problem: "Model not found"**
```bash
# Make sure you trained first
python train.py
```

**Problem: Training is slow**
- Normal! RL training is computationally intensive
- Reduce timesteps for quick tests: `python train.py --timesteps 50000`

---

## 📖 Full Documentation

See `README.md` for complete documentation, theory, and advanced usage.

---

**Ready to start? Run:**
```bash
python setup_check.py
```
