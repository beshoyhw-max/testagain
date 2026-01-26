from ultralytics import YOLO
import torch
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='TensorRT Export for 4GB GPU')
    parser.add_argument('model', type=str, help='Path to .pt model')
    parser.add_argument('--imgsz', type=int, default=1280, help='Image size (default: 1280)')
    parser.add_argument('--workspace', type=int, default=2, help='Workspace GB (default: 2 for 4GB GPU)')
    parser.add_argument('--half', action='store_true', help='Use FP16 to save memory')
    
    args = parser.parse_args()
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print("="*70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {gpu_memory_gb:.2f} GB")
    print("="*70)
    
    # Auto-adjust workspace for small GPUs
    if gpu_memory_gb < 5 and args.workspace > 2:
        print(f"⚠️  Auto-adjusting workspace from {args.workspace}GB to 2GB for 4GB GPU")
        args.workspace = 2
    
    print(f"\nExport settings:")
    print(f"  Model: {args.model}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Workspace: {args.workspace}GB")
    print(f"  Precision: {'FP16' if args.half else 'FP32'}")
    print()
    
    # Clear cache
    torch.cuda.empty_cache()
    
    # Load and export
    model = YOLO(args.model)
    
    try:
        engine_path = model.export(
            format='engine',
            imgsz=args.imgsz,
            batch=1,
            half=args.half,
            workspace=args.workspace,
            simplify=True,
            dynamic=False,
            verbose=True
        )
        
        print(f"\n✅ Export successful: {engine_path}")
        
        if os.path.exists(engine_path):
            size_mb = os.path.getsize(engine_path) / 1e6
            print(f"   Size: {size_mb:.1f} MB")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ GPU Out of Memory!")
            print(f"\nTry these options:")
            print(f"  • python {__file__} {args.model} --workspace 1")
            print(f"  • python {__file__} {args.model} --imgsz 640 --workspace 2")
            print(f"  • python {__file__} {args.model} --half --workspace 2")
        raise

if __name__ == "__main__":
    main()