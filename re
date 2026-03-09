TensorRT: starting export with TensorRT 10.14.1.48.post1...
[03/09/2026-15:30:00] [TRT] [W] Unable to determine GPU memory usage: In nvinfer1::getGpuMemStatsInBytes at C:\_src\common\extended\resources.cpp:1175
[03/09/2026-15:30:00] [TRT] [E] createInferBuilder: Error Code 6: API Usage Error (CUDA initialization failure with error: 35. Please check your CUDA installation: http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html In `anonymous-namespace'::ensureCudaInitialized::<lambda_63fe7abdc75658d02cd46c4d8b39f14c>::operator () at C:\_src\optimizer\api\builder.cpp:1363)
[03/09/2026-15:30:00] [TRT] [E] [checkMacros.cpp::nvinfer1::catchCudaError::229] Error Code 1: Cuda Runtime (In nvinfer1::catchCudaError at C:\_src\common\dispatch\checkMacros.cpp:229)
ERROR TensorRT: export failure 1.1s: pybind11::init(): factory function returned nullptr
  ✗ Export failed: pybind11::init(): factory function returned nullptr
    The app will use .pt files (slower but still works)

  → Exporting YOLO Pose Model...
    Source: yolo26s-pose.pt
    Target: yolo26s-pose.engine
    This may take 2-3 minutes...
