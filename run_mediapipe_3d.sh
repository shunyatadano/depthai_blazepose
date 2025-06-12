#!/bin/bash
# Run script for MediaPipe 3D with Open3D fix for Jetson
# This script sets environment variables to fix libgomp TLS issues

echo "Setting up environment for MediaPipe 3D on Jetson..."

# Fix for libgomp TLS allocation issue on Jetson
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1

# Alternative environment variables that might help
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Set display for GUI applications
export DISPLAY=${DISPLAY:-:0}

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Activating virtual environment..."
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "Warning: Virtual environment not found. Creating one..."
        python3 -m venv venv
        source venv/bin/activate
    fi
fi

echo "Environment setup complete."
echo "Running MediaPipe 3D pose tracker..."
python3 my_mediapipe_3d.py "$@"
