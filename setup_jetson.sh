#!/bin/bash
# Setup script for MediaPipe 3D Pose Tracking on Jetson AGX Orin
# JetPack 5.1.2 compatible

set -e

echo "=== MediaPipe 3D Pose Setup for Jetson AGX Orin ==="
echo "JetPack 5.1.2 Environment Setup"
echo "=================================================="

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "Warning: This script is designed for NVIDIA Jetson devices"
    echo "Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    cmake \
    build-essential \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libusb-1.0-0-dev \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6

# Create virtual environment
echo "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install wheel and setuptools
pip install wheel setuptools

# Install PyTorch for Jetson (JetPack 5.1.2)
echo "Installing PyTorch for Jetson..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install NumPy (compatible version for Jetson)
echo "Installing NumPy..."
pip install "numpy>=1.21.0,<1.25.0"

# Install OpenCV for Python
echo "Installing OpenCV..."
pip install opencv-python==4.8.0.76

# Install MediaPipe
echo "Installing MediaPipe..."
pip install mediapipe==0.10.7

# Install Open3D (Jetson compatible version)
echo "Installing Open3D..."
pip install open3d==0.17.0

# Install RealSense SDK Python bindings
echo "Installing RealSense Python bindings..."
pip install pyrealsense2==2.54.1.5217

# Install additional dependencies
echo "Installing additional dependencies..."
pip install scipy matplotlib

# Install other requirements
echo "Installing project requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Create udev rules for RealSense camera
echo "Setting up RealSense udev rules..."
sudo tee /etc/udev/rules.d/99-realsense-libusb.rules > /dev/null <<EOF
# RealSense device rules for libusb
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad1", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad2", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad3", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad4", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad5", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0af6", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0afe", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0aff", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b00", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b01", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b03", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b07", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b3a", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b48", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b49", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b4b", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b4d", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b52", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b5b", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b5c", MODE="0666", GROUP="plugdev"
EOF

# Reload udev rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# Add user to video and dialout groups
echo "Adding user to video and dialout groups..."
sudo usermod -a -G video $USER
sudo usermod -a -G dialout $USER

# Test installations
echo "Testing installations..."

echo "Testing NumPy..."
python3 -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

echo "Testing OpenCV..."
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

echo "Testing MediaPipe..."
python3 -c "import mediapipe; print(f'MediaPipe version: {mediapipe.__version__}')"

echo "Testing Open3D..."
python3 -c "import open3d; print(f'Open3D version: {open3d.__version__}')"

echo "Testing RealSense..."
python3 -c "import pyrealsense2; print(f'RealSense version: {pyrealsense2.__version__}')"

echo ""
echo "=== Setup Complete ==="
echo "1. Please reboot the system or log out/in to apply group changes"
echo "2. Connect your RealSense D455 camera"
echo "3. Activate the virtual environment: source venv/bin/activate"
echo "4. Run the 3D pose tracker: python3 my_mediapipe_3d.py"
echo ""
echo "Troubleshooting:"
echo "- If RealSense camera is not detected, check USB connections"
echo "- If permission errors occur, ensure user is in video group"
echo "- For display issues, set DISPLAY environment variable"
echo "====================="
