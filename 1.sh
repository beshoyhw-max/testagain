#!/bin/bash

echo "======================================"
echo "Face Recognition Setup Script"
echo "======================================"
echo ""

# Create models directory
echo "Creating models directory..."
mkdir -p models

# Download Face Detection Model
echo ""
echo "Downloading Face Detection Model (YuNet)..."
wget -O models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

if [ $? -eq 0 ]; then
    echo "✓ Face detection model downloaded successfully"
else
    echo "✗ Failed to download face detection model"
    exit 1
fi

# Download Face Recognition Model
echo ""
echo "Downloading Face Recognition Model (SFace)..."
wget -O models/face_recognition_sface_2021dec.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

if [ $? -eq 0 ]; then
    echo "✓ Face recognition model downloaded successfully"
else
    echo "✗ Failed to download face recognition model"
    exit 1
fi

# Create registered faces directory
echo ""
echo "Creating registered_faces directory..."
mkdir -p registered_faces

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Place face photos in registered_faces/ folder"
echo "   Structure: registered_faces/Person_Name/photo.jpg"
echo ""
echo "2. Run the application: streamlit run app.py"
echo ""
echo "3. Go to 'Face Recognition' tab to:"
echo "   - Register individual faces"
echo "   - Bulk import from folder"
echo "   - Manage registered people"
echo ""