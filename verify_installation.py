#!/usr/bin/env python3
"""
Verification script to check if all required packages are properly installed.
Run this after installing dependencies to ensure everything is working.
"""

import sys

def check_package(package_name, import_name=None):
    """Check if a package can be imported successfully."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        if hasattr(module, '__version__'):
            version = module.__version__
        else:
            version = "unknown"
        print(f"✓ {package_name}: {version}")
        return True
    except ImportError as e:
        print(f"✗ {package_name}: FAILED - {e}")
        return False

def main():
    """Main verification function."""
    print("Verifying package installation...")
    print("=" * 50)
    
    packages_to_check = [
        ("tensorflow", "tensorflow"),
        ("numpy", "numpy"),
        ("onnx", "onnx"),
        ("tf2onnx", "tf2onnx"),
        ("requests", "requests"),
    ]
    
    all_good = True
    for package_name, import_name in packages_to_check:
        if not check_package(package_name, import_name):
            all_good = False
    
    print("=" * 50)
    
    if all_good:
        print("✓ All packages are properly installed!")
        
        # Test TensorFlow specifically
        print("\nTesting TensorFlow functionality...")
        try:
            import tensorflow as tf
            print(f"✓ TensorFlow {tf.__version__} is working properly")
            
            # Test basic TensorFlow operation
            x = tf.constant([[1, 2], [3, 4]])
            y = tf.constant([[1], [1]])
            result = tf.matmul(x, y)
            print("✓ TensorFlow operations are working")
            
        except Exception as e:
            print(f"✗ TensorFlow test failed: {e}")
            all_good = False
    
    if all_good:
        print("\n🎉 Installation verification completed successfully!")
        print("You can now run: uv run ./tfmodel2onnx_compatible.py")
    else:
        print("\n❌ Some packages are not working properly.")
        print("Try running: uv pip install tensorflow==2.12.1 --force-reinstall")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())