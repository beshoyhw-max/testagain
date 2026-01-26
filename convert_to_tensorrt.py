"""
Convert YOLO .pt models to TensorRT .engine format
for 2-3x faster inference on NVIDIA GPUs (RTX 2050)
"""

from ultralytics import YOLO
import torch
import os

def check_gpu():
    """Verify CUDA is available"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available! TensorRT requires GPU.")
        print("\n   TROUBLESHOOTING:")
        print("   1. Run: python verify_cuda_setup.py")
        print("   2. Check: nvidia-smi")
        print("   3. Verify PyTorch CUDA installation:")
        print("      python -c \"import torch; print(torch.cuda.is_available())\"")
        print("\n   If CUDA still not working:")
        print("   - Reinstall PyTorch with CUDA:")
        print("     pip uninstall torch torchvision")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False
    
    print("✅ CUDA Available")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Additional checks
    if torch.cuda.get_device_properties(0).total_memory < 3 * 1024**3:
        print("   ⚠️  Warning: Less than 3GB VRAM detected")
        print("      Consider using smaller models or lower resolution")
    
    return True

def convert_model(model_path, imgsz=1280, half=True, workspace=4):
    """
    Convert YOLO .pt model to TensorRT .engine
    
    Args:
        model_path: Path to .pt model (e.g., 'yolo26n.pt')
        imgsz: Input size for inference (must match what you use in detection)
        half: Use FP16 precision (faster, less memory)
        workspace: GPU workspace in GB (4GB recommended for RTX 2050)
    """
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Converting: {model_path}")
    print(f"{'='*60}")
    print(f"Settings:")
    print(f"  - Input size: {imgsz}x{imgsz}")
    print(f"  - Precision: {'FP16' if half else 'FP32'}")
    print(f"  - Workspace: {workspace}GB")
    print(f"{'='*60}\n")
    
    try:
        # Load YOLO model
        print("Loading model...")
        model = YOLO(model_path)
        
        # Export to TensorRT
        print("Exporting to TensorRT (this may take 5-10 minutes)...")
        print("⏳ Building optimized engine for your GPU...")
        
        success = model.export(
            format='engine',      # TensorRT format
            device=0,             # GPU 0
            half=half,            # FP16 precision
            imgsz=imgsz,          # Input size
            workspace=workspace,  # Workspace size in GB
            verbose=True
        )
        
        # Get output path
        engine_path = model_path.replace('.pt', '.engine')
        
        if os.path.exists(engine_path):
            size_mb = os.path.getsize(engine_path) / 1024**2
            print(f"\n✅ SUCCESS!")
            print(f"   Original: {model_path}")
            print(f"   TensorRT: {engine_path} ({size_mb:.1f} MB)")
            print(f"\n🚀 Expected speedup: 2-3x faster inference")
            return True
        else:
            print(f"❌ Export failed - .engine file not created")
            return False
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_tensorrt_model(engine_path, test_image=None):
    """Test the TensorRT model"""
    print(f"\n{'='*60}")
    print(f"Verifying TensorRT model...")
    print(f"{'='*60}")
    
    try:
        model = YOLO(engine_path)
        
        if test_image:
            print(f"Running test inference on: {test_image}")
            results = model(test_image)
            print(f"✅ TensorRT model works!")
            print(f"   Inference time: {results[0].speed['inference']:.1f}ms")
        else:
            print(f"✅ TensorRT model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorRT model verification failed: {e}")
        return False

def main():
    """Convert all YOLO models to TensorRT"""
    
    print("="*60)
    print("YOLO to TensorRT Converter")
    print("Optimized for RTX 2050 with CUDA 13.1")
    print("="*60)
    
    # Check GPU
    if not check_gpu():
        return
    
    # Models to convert
    models = [
        {
            'path': 'yolo26n.pt',
            'name': 'Detection Model',
            'imgsz': 1280  # Match your detector.py imgsz setting
        },
        {
            'path': 'yolo26n-pose.pt',
            'name': 'Pose Model',
            'imgsz': 640  # Pose models typically use smaller input
        }
    ]
    
    converted = []
    failed = []
    
    for model_info in models:
        model_path = model_info['path']
        
        print(f"\n\n{'#'*60}")
        print(f"# {model_info['name']}")
        print(f"{'#'*60}")
        
        if not os.path.exists(model_path):
            print(f"⏭️  Skipping {model_path} (not found)")
            print(f"   The model will be auto-downloaded on first run")
            continue
        
        # Convert
        success = convert_model(
            model_path=model_path,
            imgsz=model_info['imgsz'],
            half=True,  # FP16 for RTX 2050
            workspace=4  # 4GB workspace
        )
        
        if success:
            converted.append(model_path)
        else:
            failed.append(model_path)
    
    # Summary
    print(f"\n\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Converted: {len(converted)}")
    for model in converted:
        print(f"   - {model} → {model.replace('.pt', '.engine')}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for model in failed:
            print(f"   - {model}")
    
    if converted:
        print(f"\n{'='*60}")
        print("NEXT STEPS")
        print(f"{'='*60}")
        print("1. Update your code to use .engine files:")
        print("   In detector.py and camera_manager.py:")
        print("   - Change: model_path='yolo26n.pt'")
        print("   - To:     model_path='yolo26n.engine'")
        print()
        print("   - Change: pose_model_path='yolo26n-pose.pt'")
        print("   - To:     pose_model_path='yolo26n-pose.engine'")
        print()
        print("2. Restart your application")
        print()
        print("3. Enjoy 2-3x faster performance! 🚀")
        print()
        print("⚠️  NOTE: .engine files are GPU-specific!")
        print("   They only work on the RTX 2050 they were built on.")
        print("   Keep your .pt files for portability.")
    
if __name__ == "__main__":
    main()
