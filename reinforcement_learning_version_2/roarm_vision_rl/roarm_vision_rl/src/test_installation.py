#!/usr/bin/env python3
"""
Quick test script to verify installation and environment setup
"""
import sys
import importlib

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name} - {str(e)}")
        return False

def main():
    print("=" * 70)
    print("RoArm Vision RL - Installation Test")
    print("=" * 70)
    print("\nChecking required packages...\n")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('gymnasium', 'Gymnasium'),
        ('pybullet', 'PyBullet'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm'),
        ('cv2', 'OpenCV')
    ]
    
    all_installed = True
    for module, name in required_packages:
        if not check_import(module, name):
            all_installed = False
    
    print("\n" + "=" * 70)
    
    if all_installed:
        print("✓ All required packages are installed!")
        print("\nChecking PyTorch configuration...")
        
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU device: {torch.cuda.get_device_name(0)}")
        else:
            print("  Note: Training will use CPU (slower)")
        
        print("\nTesting environment creation...")
        try:
            from roarm_vision_env import RoArmVisionEnv
            env = RoArmVisionEnv(render_mode=None, max_steps=10)
            obs, _ = env.reset()
            print(f"  ✓ Environment created successfully")
            print(f"  ✓ Image shape: {obs['image'].shape}")
            print(f"  ✓ Proprioception shape: {obs['proprioception'].shape}")
            
            # Test one step
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  ✓ Environment step executed")
            print(f"  ✓ Reward: {reward:.2f}")
            
            env.close()
            print("\n✓ Environment test passed!")
        except Exception as e:
            print(f"\n✗ Environment test failed: {str(e)}")
            all_installed = False
        
        print("\nTesting agent creation...")
        try:
            from vision_sac_agent import VisionSAC
            agent = VisionSAC(
                image_channels=3,
                proprioception_dim=8,
                action_dim=4,
                action_bounds=([-3.1416, -1.5708, -1.0, 0.0],
                              [3.1416, 1.5708, 3.1416, 1.5])
            )
            print(f"  ✓ Agent created successfully")
            print(f"  ✓ Device: {agent.device}")
        except Exception as e:
            print(f"\n✗ Agent test failed: {str(e)}")
            all_installed = False
        
        print("\n" + "=" * 70)
        if all_installed:
            print("✓ ALL TESTS PASSED!")
            print("\nYou can now start training:")
            print("  cd src")
            print("  python train_vision.py")
        else:
            print("✗ Some tests failed. Please check the errors above.")
    else:
        print("✗ Some required packages are missing.")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
    
    print("=" * 70)
    return 0 if all_installed else 1

if __name__ == "__main__":
    sys.exit(main())
