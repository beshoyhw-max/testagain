# Installation Guide

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX 2050 or better (4GB+ VRAM)
- **RAM**: 16GB minimum
- **CPU**: Intel i5 or better (i9-13900H recommended)
- **Storage**: 10GB free space

### Software
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Python**: 3.9 - 3.11 (Python 3.12 not yet fully supported by all packages)
- **CUDA**: 11.8 or 12.1 (for GPU acceleration)

---

## Step-by-Step Installation

### 1. Install Python (if not already installed)

**Windows:**
```bash
# Download from python.org
# Or use winget
winget install Python.Python.3.11
```

**Linux:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

**macOS:**
```bash
brew install python@3.11
```

### 2. Install CUDA (for GPU acceleration)

**Check if CUDA is installed:**
```bash
nvidia-smi
```

**If not installed:**
- Download from: https://developer.nvidia.com/cuda-downloads
- Install CUDA Toolkit 11.8 or 12.1
- Verify: `nvcc --version`

### 3. Clone/Download Project

```bash
# If using git
git clone <your-repo-url>
cd phone-detection-system

# Or download and extract ZIP
```

### 4. Create Virtual Environment (Recommended)

```bash
# Create venv
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### 5. Install Dependencies

**Option A: CPU Only (No GPU)**
```bash
pip install -r requirements.txt
```

**Option B: GPU (CUDA 11.8) - RECOMMENDED for RTX 2050**
```bash
# Install PyTorch with CUDA first
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Then install other requirements
pip install -r requirements.txt
```

**Option C: GPU (CUDA 12.1)**
```bash
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 6. Download YOLO Models

```bash
# Create models directory
mkdir -p models

# Download detection model (auto-downloads on first run)
# Or manually:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O yolo26n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt -O yolo26n-pose.pt
```

**Note:** The system will auto-download models on first run if not present.

### 7. Download MediaPipe Face Model

```bash
# Download face landmarker model
mkdir -p models
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -O models/face_landmarker.task
```

### 8. Setup Face Recognition (Optional)

```bash
# Run setup script
chmod +x setup_face_recognition.sh
./setup_face_recognition.sh

# Or manually:
mkdir -p models
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx -O models/face_detection_yunet_2023mar.onnx
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx -O models/face_recognition_sface_2021dec.onnx
```

### 9. Verify Installation

```bash
# Check Python packages
pip list | grep -E "torch|opencv|ultralytics|streamlit|mediapipe"

# Expected output:
# mediapipe            0.10.x
# opencv-contrib-python 4.8.x
# opencv-python        4.8.x
# streamlit            1.28.x
# torch                2.1.x+cu118
# torchvision          0.16.x+cu118
# ultralytics          8.x.x
```

### 10. Test Installation

```python
# test_install.py
import cv2
import torch
import mediapipe as mp
from ultralytics import YOLO
import streamlit as st

print("✓ OpenCV:", cv2.__version__)
print("✓ PyTorch:", torch.__version__)
print("✓ CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✓ CUDA Device:", torch.cuda.get_device_name(0))
print("✓ MediaPipe:", mp.__version__)
print("✓ Streamlit:", st.__version__)

# Test YOLO
try:
    model = YOLO('yolov8n.pt')
    print("✓ YOLO model loaded successfully")
except Exception as e:
    print("✗ YOLO error:", e)

print("\n✅ All dependencies installed correctly!")
```

Run test:
```bash
python test_install.py
```

---

## Troubleshooting

### Issue: "CUDA not available"

**Solution:**
```bash
# Verify NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### Issue: "ModuleNotFoundError: No module named 'cv2'"

**Solution:**
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python==4.8.1.78 opencv-contrib-python==4.8.1.78
```

### Issue: MediaPipe protobuf error

**Solution:**
```bash
pip uninstall protobuf
pip install protobuf==4.25.3
```

### Issue: Streamlit won't start

**Solution:**
```bash
# Check port 8501 is free
# Windows:
netstat -ano | findstr :8501

# Linux/Mac:
lsof -i :8501

# Use different port
streamlit run app.py --server.port 8502
```

### Issue: Models download fails

**Solution:**
```bash
# Set proxy if behind firewall
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port

# Or download manually from URLs in setup script
```

---

## Directory Structure After Installation

```
phone-detection-system/
├── venv/                          # Virtual environment
├── models/                        # AI models
│   ├── face_landmarker.task
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
├── detections/                    # Evidence screenshots (auto-created)
├── registered_faces/              # Face registration (auto-created)
├── app.py                         # Main Streamlit app
├── detector.py                    # Detection logic
├── camera_manager.py              # Camera handling
├── sleep_detector.py              # Sleep detection
├── face_recognizer.py             # Face recognition
├── requirements.txt               # Dependencies
├── cameras.json                   # Camera config (auto-created)
├── face_database.pkl             # Face DB (auto-created)
├── yolo26n.pt                    # Detection model (auto-downloaded)
└── yolo26n-pose.pt               # Pose model (auto-downloaded)
```

---

## First Run

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Run the application
streamlit run app.py

# Open browser to http://localhost:8501
```

---

## Performance Optimization (Optional)

### Convert to TensorRT (2-3x faster)

```python
from ultralytics import YOLO

# Convert detection model
model = YOLO('yolo26n.pt')
model.export(format='engine', device=0, half=True)

# Convert pose model
pose = YOLO('yolo26n-pose.pt')
pose.export(format='engine', device=0, half=True)

# Update code to use .engine files
```

### Reduce Memory Usage

In `camera_manager.py`:
```python
# Lower resolution
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)  # from 1280
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540) # from 720
```

In `detector.py`:
```python
# Lower inference size
imgsz=640  # from 1280
```

---

## Next Steps

1. ✅ Run `streamlit run app.py`
2. ✅ Configure cameras in "Configuration" tab
3. ✅ (Optional) Register faces in "Face Recognition" tab
4. ✅ Start monitoring in "Live Dashboard"

For detailed usage instructions, see `README_FACE_RECOGNITION.md`
