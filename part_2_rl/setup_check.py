"""
Setup Verification Script
Run this to check if everything is installed correctly
"""

import sys
import os


def check_imports():
    """Check if all required packages are installed"""
    
    print("Checking Python version...")
    print(f"  Python {sys.version}")
    
    if sys.version_info < (3, 8):
        print("  ❌ Python 3.8 or higher required!")
        return False
    else:
        print("  ✓ Python version OK")
    
    print("\nChecking required packages...")
    
    packages = [
        ('pybullet', 'PyBullet'),
        ('gymnasium', 'Gymnasium'),
        ('stable_baselines3', 'Stable-Baselines3'),
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
    ]
    
    all_ok = True
    
    for package, name in packages:
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {name} ({version})")
        except ImportError:
            print(f"  ❌ {name} not found!")
            all_ok = False
    
    return all_ok


def check_files():
    """Check if required files exist"""
    
    print("\nChecking required files...")
    
    files = [
        'roarm.urdf',
        'pen_pickup_env.py',
        'train.py',
        'test.py',
        'requirements.txt',
    ]
    
    all_ok = True
    
    for file in files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ❌ {file} not found!")
            all_ok = False
    
    print("\nChecking meshes directory...")
    if os.path.exists('meshes'):
        mesh_files = os.listdir('meshes')
        print(f"  ✓ meshes/ ({len(mesh_files)} files)")
        for mesh in mesh_files:
            print(f"    - {mesh}")
    else:
        print("  ❌ meshes/ directory not found!")
        all_ok = False
    
    return all_ok


def test_environment():
    """Test if environment can be created"""
    
    print("\nTesting environment creation...")
    
    try:
        from pen_pickup_env import PenPickupEnv
        
        # Create environment (no rendering)
        env = PenPickupEnv(render_mode=None)
        print("  ✓ Environment created successfully")
        
        # Test reset
        obs, info = env.reset()
        print(f"  ✓ Environment reset (observation shape: {obs.shape})")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  ✓ Environment step (reward: {reward:.2f})")
        
        env.close()
        print("  ✓ Environment closed")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("PEN PICKUP RL - SETUP VERIFICATION")
    print("=" * 60)
    print()
    
    # Run checks
    imports_ok = check_imports()
    files_ok = check_files()
    
    if imports_ok and files_ok:
        env_ok = test_environment()
    else:
        print("\n⚠️  Skipping environment test due to missing dependencies/files")
        env_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    if imports_ok and files_ok and env_ok:
        print("✓ All checks passed!")
        print("\nYou're ready to start!")
        print("\nNext steps:")
        print("  1. Run demo: python demo.py")
        print("  2. Start training: python train.py")
        print("  3. Test trained model: python test.py")
    else:
        print("❌ Some checks failed")
        print("\nTroubleshooting:")
        
        if not imports_ok:
            print("  - Install missing packages: pip install -r requirements.txt")
        
        if not files_ok:
            print("  - Make sure all files are in the correct directory")
            print("  - Check that you're in the pen_pickup_rl/ directory")
        
        if not env_ok:
            print("  - Check error messages above")
            print("  - Ensure PyBullet is installed correctly")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
