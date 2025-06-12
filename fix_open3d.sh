#!/bin/bash
# Jetson Open3D Fix Script
# This script attempts to fix Open3D import issues on Jetson devices

echo "Jetson Open3D Compatibility Fix"
echo "================================"

# Method 1: Preload libgomp
echo "Method 1: Setting LD_PRELOAD for libgomp..."
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD

# Method 2: Set OpenMP environment variables
echo "Method 2: Setting OpenMP environment variables..."
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Method 3: Try different libgomp versions
echo "Method 3: Checking available libgomp versions..."
LIBGOMP_PATHS=(
    "/lib/aarch64-linux-gnu/libgomp.so.1"
    "/usr/lib/aarch64-linux-gnu/libgomp.so.1" 
    "/usr/local/lib/libgomp.so.1"
    "/opt/nvidia/vpi2/lib64/libgomp.so.1"
)

for path in "${LIBGOMP_PATHS[@]}"; do
    if [ -f "$path" ]; then
        echo "Found libgomp at: $path"
    fi
done

# Method 4: Install alternative Open3D version
echo "Method 4: Checking if alternative Open3D installation is needed..."

# Test Open3D import
echo "Testing Open3D import..."
python3 -c "
try:
    import open3d as o3d
    print('✓ Open3D imported successfully')
    print(f'Open3D version: {o3d.__version__}')
except ImportError as e:
    print(f'✗ Open3D import failed: {e}')
    print('Suggesting alternative solutions...')
"

echo ""
echo "Suggested solutions:"
echo "1. Use the lite version: python3 my_mediapipe_3d_lite.py"
echo "2. Use the run script: ./run_mediapipe_3d.sh"
echo "3. Install CPU-only Open3D: pip install open3d-cpu"
echo "4. Use conda environment with specific Open3D build"
echo ""
echo "If none work, the lite version provides all functionality except real-time 3D visualization."
