# Directory Structure

```
pen_pickup_rl/
│
├── README.md                    # Comprehensive documentation
├── QUICKSTART.md                # Quick start guide
├── TROUBLESHOOTING.md           # Troubleshooting guide
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── roarm.urdf                   # Robot URDF model
│
├── meshes/                      # Robot mesh files
│   ├── base_link.STL
│   ├── gripper_link.STL
│   ├── link1.STL
│   ├── link2.STL
│   └── link3.STL
│
├── pen_pickup_env.py            # Custom Gymnasium environment
├── train.py                     # Training script (PPO)
├── test.py                      # Testing/visualization script
├── demo.py                      # Quick demo script
├── setup_check.py               # Setup verification script
├── config.py                    # Configuration & hyperparameters
│
├── models/                      # Saved models (created during training)
│   ├── .gitkeep
│   ├── best_model.zip          # Best performing model
│   ├── vec_normalize.pkl       # Normalization statistics
│   └── ppo_pen_pickup_*.zip    # Checkpoints
│
├── logs/                        # Training logs (created during training)
│   ├── .gitkeep
│   └── PPO_*/                  # TensorBoard logs
│
└── videos/                      # Recorded videos (optional)
    └── .gitkeep
```

## File Descriptions

### Main Scripts

| File | Purpose |
|------|---------|
| `pen_pickup_env.py` | Custom PyBullet environment implementing the pen pickup task |
| `train.py` | Main training script using PPO algorithm |
| `test.py` | Test trained models with visualization |
| `demo.py` | Quick demo showing the task with random actions |
| `setup_check.py` | Verify installation and setup |
| `config.py` | Configuration presets and hyperparameters |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | Complete documentation, theory, and usage |
| `QUICKSTART.md` | Quick start guide (3-step setup) |
| `TROUBLESHOOTING.md` | Common issues and solutions |
| `DIRECTORY_STRUCTURE.md` | This file - directory structure reference |

### Robot Files

| File | Purpose |
|------|---------|
| `roarm.urdf` | Robot model description (URDF format) |
| `meshes/*.STL` | 3D meshes for robot visualization |

### Generated Directories

| Directory | Contents |
|-----------|----------|
| `models/` | Saved model checkpoints and best model |
| `logs/` | TensorBoard logs for monitoring training |
| `videos/` | Optional recorded videos of episodes |

## Workflow

```
1. Setup
   ├── pip install -r requirements.txt
   └── python setup_check.py

2. Exploration (Optional)
   └── python demo.py

3. Training
   ├── python train.py
   └── tensorboard --logdir logs/

4. Testing
   ├── python test.py
   └── python test.py --random (baseline comparison)

5. Iteration
   ├── Modify config.py
   ├── Adjust pen_pickup_env.py (rewards)
   └── Re-train with new settings
```

## File Sizes (Approximate)

- Python scripts: ~5-10 KB each
- URDF: ~7 KB
- Mesh files: ~5-280 KB each (total ~450 KB)
- Documentation: ~30-60 KB each
- Saved models: ~2-5 MB each
- Logs: Varies (can grow large with long training)

## Version Control

Files tracked by git:
- All .py files
- All .md files
- requirements.txt
- .gitignore
- roarm.urdf
- meshes/*.STL

Files ignored by git:
- models/*.zip
- models/*.pkl
- logs/*
- videos/*.mp4
- __pycache__/
- *.pyc
