"""
Complete CUDA Verification Script
Diagnoses issues with CUDA/PyTorch/GPU setup
"""

import sys
import subprocess
import os

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.returncode
    except Exception as e:
        return str(e), 1

def check_nvidia_driver():
    """Check if NVIDIA driver is installed"""
    print("\n" + "="*60)
    print("1. Checking NVIDIA Driver")
    print("="*60)
    
    output, code = run_command("nvidia-smi")
    
    if code == 0:
        print("✅ NVIDIA Driver installed")
        # Parse driver version
        for line in output.split('\n'):
            if 'Driver Version' in line:
                print(f"   {line.strip()}")
                break
        return True
    else:
        print("❌ nvidia-smi not found!")
        print("\n   SOLUTION:")
        print("   1. Install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx")
        print("   2. Reboot your computer")
        print("   3. Run this script again")
        return False

def check_cuda_toolkit():
    """Check CUDA Toolkit installation"""
    print("\n" + "="*60)
    print("2. Checking CUDA Toolkit")
    print("="*60)
    
    output, code = run_command("nvcc --version")
    
    if code == 0:
        print("✅ CUDA Toolkit installed")
        # Parse CUDA version
        for line in output.split('\n'):
            if 'release' in line.lower():
                print(f"   {line.strip()}")
                break
        return True
    else:
        print("⚠️  nvcc not found (CUDA Toolkit not in PATH)")
        print("\n   This is OK if you installed PyTorch with CUDA included.")
        print("   PyTorch bundles its own CUDA runtime.")
        return None  # Not critical

def check_python_packages():
    """Check if required Python packages are installed"""
    print("\n" + "="*60)
    print("3. Checking Python Packages")
    print("="*60)
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'ultralytics': 'Ultralytics',
        'mediapipe': 'MediaPipe',
        'streamlit': 'Streamlit'
    }
    
    all_installed = True
    
    for package, name in packages.items():
        try:
            if package == 'cv2':
                import cv2
                version = cv2.__version__
            else:
                mod = __import__(package)
                version = mod.__version__ if hasattr(mod, '__version__') else 'installed'
            
            print(f"✅ {name:15} - {version}")
        except ImportError:
            print(f"❌ {name:15} - NOT INSTALLED")
            all_installed = False
    
    if not all_installed:
        print("\n   SOLUTION:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_pytorch_cuda():
    """Check if PyTorch can see CUDA"""
    print("\n" + "="*60)
    print("4. Checking PyTorch CUDA Integration")
    print("="*60)
    
    try:
        import torch
        
        print(f"PyTorch Version: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✅ CUDA Available: True")
            print(f"   CUDA Version (PyTorch): {torch.version.cuda}")
            print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"   Number of GPUs: {torch.cuda.device_count()}")
            
            # GPU details
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\n   GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"   - Compute Capability: {props.major}.{props.minor}")
                print(f"   - Total Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"   - Multi-Processors: {props.multi_processor_count}")
            
            return True
        else:
            print(f"❌ CUDA Available: False")
            print("\n   POSSIBLE CAUSES:")
            print("   1. PyTorch installed without CUDA support (CPU-only version)")
            print("   2. NVIDIA driver not installed")
            print("   3. Incompatible CUDA versions")
            
            # Check if CPU-only PyTorch
            if '+cpu' in torch.__version__:
                print("\n   DIAGNOSIS: PyTorch CPU-only version detected!")
                print("\n   SOLUTION:")
                print("   pip uninstall torch torchvision torchaudio")
                print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            
            return False
            
    except ImportError:
        print("❌ PyTorch not installed!")
        print("\n   SOLUTION:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False

def test_cuda_operations():
    """Test actual CUDA operations"""
    print("\n" + "="*60)
    print("5. Testing CUDA Operations")
    print("="*60)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⏭️  Skipped (CUDA not available)")
            return False
        
        # Test tensor creation on GPU
        print("Testing tensor operations on GPU...")
        
        # Create tensor on GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        
        # Matrix multiplication
        import time
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"✅ GPU tensor operations successful")
        print(f"   Matrix multiplication (1000x1000): {gpu_time*1000:.2f}ms")
        
        # Compare with CPU
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        start = time.time()
        z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start
        
        print(f"   CPU time: {cpu_time*1000:.2f}ms")
        print(f"   Speedup: {cpu_time/gpu_time:.1f}x faster on GPU")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA operations failed: {e}")
        return False

def test_yolo_inference():
    """Test YOLO model inference on GPU"""
    print("\n" + "="*60)
    print("6. Testing YOLO Inference on GPU")
    print("="*60)
    
    try:
        import torch
        from ultralytics import YOLO
        import numpy as np
        
        if not torch.cuda.is_available():
            print("⏭️  Skipped (CUDA not available)")
            return False
        
        print("Loading YOLO model...")
        model = YOLO('yolov8n.pt')  # Will auto-download if needed
        
        print("Running inference on GPU...")
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run inference
        results = model(dummy_image, device=0, verbose=False)
        
        print(f"✅ YOLO GPU inference successful")
        print(f"   Inference time: {results[0].speed['inference']:.1f}ms")
        print(f"   Model running on: {next(model.model.parameters()).device}")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_tensorrt_support():
    """Check if TensorRT is available"""
    print("\n" + "="*60)
    print("7. Checking TensorRT Support")
    print("="*60)
    
    try:
        import tensorrt as trt
        print(f"✅ TensorRT installed: {trt.__version__}")
        return True
    except ImportError:
        print("⚠️  TensorRT not installed")
        print("\n   This is OPTIONAL but recommended for best performance.")
        print("   TensorRT provides 2-3x speedup on NVIDIA GPUs.")
        print("\n   Install: pip install tensorrt")
        print("   Or let Ultralytics handle it automatically during export.")
        return None

def main():
    """Run all verification checks"""
    
    print("="*60)
    print("CUDA & GPU Setup Verification")
    print("RTX 2050 with CUDA 13.1")
    print("="*60)
    
    results = {}
    
    # Run checks
    results['nvidia_driver'] = check_nvidia_driver()
    results['cuda_toolkit'] = check_cuda_toolkit()
    results['python_packages'] = check_python_packages()
    results['pytorch_cuda'] = check_pytorch_cuda()
    
    if results['pytorch_cuda']:
        results['cuda_operations'] = test_cuda_operations()
        results['yolo_inference'] = test_yolo_inference()
    else:
        results['cuda_operations'] = False
        results['yolo_inference'] = False
    
    results['tensorrt'] = check_tensorrt_support()
    
    # Summary
    print("\n\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    critical_checks = {
        'NVIDIA Driver': results['nvidia_driver'],
        'Python Packages': results['python_packages'],
        'PyTorch CUDA': results['pytorch_cuda'],
    }
    
    optional_checks = {
        'CUDA Toolkit': results['cuda_toolkit'],
        'CUDA Operations': results['cuda_operations'],
        'YOLO Inference': results['yolo_inference'],
        'TensorRT': results['tensorrt']
    }
    
    print("\nCritical Components:")
    all_critical_ok = True
    for name, status in critical_checks.items():
        symbol = "✅" if status else "❌"
        print(f"  {symbol} {name}")
        if not status:
            all_critical_ok = False
    
    print("\nOptional Components:")
    for name, status in optional_checks.items():
        if status is True:
            symbol = "✅"
        elif status is False:
            symbol = "❌"
        else:
            symbol = "⚠️ "
        print(f"  {symbol} {name}")
    
    # Final verdict
    print("\n" + "="*60)
    if all_critical_ok and results['pytorch_cuda']:
        print("✅ SYSTEM READY!")
        print("\nYour RTX 2050 is properly configured.")
        print("You can now:")
        print("  1. Run: streamlit run app.py")
        print("  2. (Optional) Convert to TensorRT: python convert_to_tensorrt.py")
        print("\nExpected performance:")
        print("  - 4-6 cameras with .pt models")
        print("  - 8-10 cameras with .engine models (TensorRT)")
    elif all_critical_ok:
        print("⚠️  SYSTEM PARTIALLY READY")
        print("\nCPU-only mode will work but be slower.")
        print("Fix PyTorch CUDA installation for GPU acceleration.")
    else:
        print("❌ SETUP INCOMPLETE")
        print("\nPlease fix the issues above before proceeding.")
        print("Scroll up to see specific solutions for each problem.")
    
    print("="*60)

if __name__ == "__main__":
    main()
