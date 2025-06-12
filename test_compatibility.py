#!/usr/bin/env python3
"""
Comprehensive test script for MediaPipe 3D pose detection on Jetson
Tests different visualization methods and identifies the best working solution
"""

import sys
import subprocess

def test_basic_imports():
    """Test basic module imports"""
    print("Testing basic imports...")
    
    modules = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'), 
        ('mediapipe', 'MediaPipe'),
        ('pyrealsense2', 'RealSense'),
        ('matplotlib', 'Matplotlib')
    ]
    
    all_success = True
    for module, name in modules:
        try:
            exec(f"import {module}")
            print(f"✓ {name} imported successfully")
        except ImportError as e:
            print(f"✗ {name} import failed: {e}")
            all_success = False
    
    return all_success

def test_open3d_import():
    """Test Open3D import with different methods"""
    print("\nTesting Open3D import...")
    
    # Method 1: Direct import
    try:
        import open3d as o3d
        print(f"✓ Open3D imported successfully (version: {o3d.__version__})")
        return True
    except ImportError as e:
        print(f"✗ Direct Open3D import failed: {e}")
    
    # Method 2: With environment variable
    import os
    original_preload = os.environ.get('LD_PRELOAD', '')
    try:
        os.environ['LD_PRELOAD'] = '/lib/aarch64-linux-gnu/libgomp.so.1'
        import open3d as o3d
        print(f"✓ Open3D imported with LD_PRELOAD (version: {o3d.__version__})")
        return True
    except ImportError as e:
        print(f"✗ Open3D import with LD_PRELOAD failed: {e}")
    finally:
        os.environ['LD_PRELOAD'] = original_preload
    
    return False

def test_camera_access():
    """Test camera access"""
    print("\nTesting camera access...")
    
    import cv2
    
    # Test different camera IDs
    for camera_id in range(6):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera {camera_id} is working ({frame.shape})")
                cap.release()
                return camera_id
            else:
                print(f"✗ Camera {camera_id} opened but cannot read frames")
        else:
            print(f"✗ Camera {camera_id} cannot be opened")
        cap.release()
    
    return None

def test_realsense():
    """Test RealSense camera"""
    print("\nTesting RealSense camera...")
    
    try:
        import pyrealsense2 as rs
        
        # Check for devices
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("✗ No RealSense devices found")
            return False
        
        print(f"✓ Found {len(devices)} RealSense device(s)")
        for i, device in enumerate(devices):
            name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
            print(f"  Device {i}: {name} (Serial: {serial})")
        
        # Test basic pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        try:
            profile = pipeline.start(config)
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            
            if color_frame:
                print("✓ RealSense pipeline test successful")
                pipeline.stop()
                return True
            else:
                print("✗ RealSense pipeline started but no frames received")
                pipeline.stop()
                return False
                
        except Exception as e:
            print(f"✗ RealSense pipeline test failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ RealSense test failed: {e}")
        return False

def run_demo_tests():
    """Run available demo scripts"""
    print("\nTesting demo scripts...")
    
    demos = [
        ('demo_simple.py', 'Basic MediaPipe demo'),
        ('test_realsense.py', 'RealSense test'),
        ('my_mediapipe_3d_lite.py', '3D pose tracking (lite)'),
    ]
    
    working_demos = []
    
    for script, description in demos:
        print(f"\nTesting {description}...")
        try:
            # Just test import, don't run the full demo
            result = subprocess.run([
                sys.executable, '-c', 
                f"exec(open('{script}').read().split('if __name__')[0])"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print(f"✓ {script} imports successfully")
                working_demos.append((script, description))
            else:
                print(f"✗ {script} import failed: {result.stderr[:200]}")
                
        except subprocess.TimeoutExpired:
            print(f"⚠ {script} test timed out (might still work)")
        except Exception as e:
            print(f"✗ {script} test error: {e}")
    
    return working_demos

def main():
    """Main test function"""
    print("MediaPipe 3D Pose Detection - Jetson Compatibility Test")
    print("=" * 60)
    
    # Test basic imports
    basic_imports_ok = test_basic_imports()
    
    # Test Open3D
    open3d_ok = test_open3d_import()
    
    # Test camera access
    working_camera = test_camera_access()
    
    # Test RealSense
    realsense_ok = test_realsense()
    
    # Test demo scripts
    working_demos = run_demo_tests()
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPATIBILITY TEST SUMMARY")
    print("=" * 60)
    
    print(f"Basic imports: {'✓ PASS' if basic_imports_ok else '✗ FAIL'}")
    print(f"Open3D import: {'✓ PASS' if open3d_ok else '✗ FAIL'}")
    print(f"Camera access: {'✓ PASS' if working_camera is not None else '✗ FAIL'}")
    if working_camera is not None:
        print(f"  Working camera ID: {working_camera}")
    print(f"RealSense: {'✓ PASS' if realsense_ok else '✗ FAIL'}")
    
    print(f"\nWorking demos: {len(working_demos)}")
    for script, description in working_demos:
        print(f"  ✓ {script} - {description}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 40)
    
    if not basic_imports_ok:
        print("❗ Install missing basic dependencies first:")
        print("   pip install opencv-python numpy mediapipe pyrealsense2 matplotlib")
    
    if not open3d_ok:
        print("❗ Open3D not working. Use these alternatives:")
        print("   1. Run: python3 my_mediapipe_3d_lite.py")
        print("   2. Or try: ./run_mediapipe_3d.sh")
        print("   3. Or run: ./fix_open3d.sh")
    else:
        print("✓ Open3D working. You can use the full version:")
        print("   python3 my_mediapipe_3d.py")
    
    if working_camera is None:
        print("❗ No working camera found. Check connections.")
    
    if not realsense_ok:
        print("❗ RealSense not working. Check:")
        print("   - USB 3.0 connection")
        print("   - Device permissions")
        print("   - Driver installation")
    
    print("\nFor the most compatible experience, use:")
    print("  python3 my_mediapipe_3d_lite.py")


if __name__ == "__main__":
    main()
