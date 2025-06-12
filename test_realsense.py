#!/usr/bin/env python3
"""
RealSense D455 Test Script
Tests RealSense camera functionality and displays basic depth information
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import time
from typing import Optional


class RealSenseTest:
    """Test RealSense D455 camera functionality"""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        """Initialize RealSense camera"""
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        # Start pipeline
        try:
            self.profile = self.pipeline.start(self.config)
            print("✓ RealSense camera initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize RealSense camera: {e}")
            raise
        
        # Get device information
        device = self.profile.get_device()
        print(f"Device: {device.get_info(rs.camera_info.name)}")
        print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")
        print(f"Firmware: {device.get_info(rs.camera_info.firmware_version)}")
        
        # Get depth sensor and scale
        self.depth_sensor = device.first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        print(f"Depth scale: {self.depth_scale}")
        
        # Get camera intrinsics
        self.color_intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.depth_intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        
        print(f"Color intrinsics: {self.color_intrinsics.width}x{self.color_intrinsics.height}")
        print(f"Color focal length: fx={self.color_intrinsics.fx:.1f}, fy={self.color_intrinsics.fy:.1f}")
        print(f"Color principal point: cx={self.color_intrinsics.ppx:.1f}, cy={self.color_intrinsics.ppy:.1f}")
        
        # Create alignment object (align depth to color)
        self.align = rs.align(rs.stream.color)
        
        # Create colorizer for depth visualization
        self.colorizer = rs.colorizer()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
    
    def get_3d_point(self, x: int, y: int, depth_frame) -> Optional[tuple]:
        """Get 3D world coordinates from 2D pixel coordinates"""
        try:
            depth_value = depth_frame.get_distance(x, y)
            if depth_value > 0:
                point_3d = rs.rs2_deproject_pixel_to_point(
                    self.color_intrinsics, [x, y], depth_value
                )
                return (point_3d[0], point_3d[1], point_3d[2])
        except:
            pass
        return None
    
    def draw_crosshair_and_info(self, image, depth_frame):
        """Draw crosshair at center and display depth information"""
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Draw crosshair
        cv2.line(image, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
        cv2.line(image, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)
        cv2.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)
        
        # Get depth at center
        point_3d = self.get_3d_point(center_x, center_y, depth_frame)
        if point_3d:
            distance = np.sqrt(point_3d[0]**2 + point_3d[1]**2 + point_3d[2]**2)
            cv2.putText(image, f"Center Distance: {distance:.2f}m", (10, h - 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image, f"3D Point: ({point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f})", 
                       (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw depth at corners
        corners = [(50, 50), (w-50, 50), (w-50, h-50), (50, h-50)]
        for i, (x, y) in enumerate(corners):
            point_3d = self.get_3d_point(x, y, depth_frame)
            if point_3d:
                distance = np.sqrt(point_3d[0]**2 + point_3d[1]**2 + point_3d[2]**2)
                cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
                cv2.putText(image, f"{distance:.2f}m", (x+10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    def run_test(self):
        """Run the camera test"""
        print("\nStarting RealSense test...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'd' - Toggle depth view")
        print("  'c' - Capture frame")
        print("  'i' - Toggle info display")
        
        show_depth = False
        show_info = True
        capture_count = 0
        
        try:
            while True:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                
                # Align depth to color
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Create depth colormap
                depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
                
                # Choose display image
                if show_depth:
                    display_image = depth_colormap.copy()
                    window_title = "RealSense Test - Depth View"
                else:
                    display_image = color_image.copy()
                    window_title = "RealSense Test - Color View"
                
                # Draw information overlay
                if show_info:
                    self.draw_crosshair_and_info(display_image, depth_frame)
                
                # Calculate and display FPS
                self.frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                if elapsed_time > 0:
                    fps = self.frame_count / elapsed_time
                    cv2.putText(display_image, f"FPS: {fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display mode indicator
                h, w = display_image.shape[:2]
                mode_text = "DEPTH" if show_depth else "COLOR"
                cv2.putText(display_image, mode_text, (w - 100, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Display instructions
                cv2.putText(display_image, "q:quit d:depth c:capture i:info", (10, h - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow(window_title, display_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    show_depth = not show_depth
                    print(f"Depth view: {'ON' if show_depth else 'OFF'}")
                elif key == ord('i'):
                    show_info = not show_info
                    print(f"Info overlay: {'ON' if show_info else 'OFF'}")
                elif key == ord('c'):
                    # Capture current frame
                    capture_count += 1
                    color_filename = f"capture_color_{capture_count:03d}.jpg"
                    depth_filename = f"capture_depth_{capture_count:03d}.png"
                    
                    cv2.imwrite(color_filename, color_image)
                    cv2.imwrite(depth_filename, depth_image)
                    print(f"Captured: {color_filename}, {depth_filename}")
                    
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.pipeline.stop()
        cv2.destroyAllWindows()


def check_realsense_devices():
    """Check for connected RealSense devices"""
    print("Checking for RealSense devices...")
    
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("✗ No RealSense devices found")
        return False
    
    print(f"✓ Found {len(devices)} RealSense device(s):")
    for i, device in enumerate(devices):
        print(f"  Device {i}:")
        print(f"    Name: {device.get_info(rs.camera_info.name)}")
        print(f"    Serial: {device.get_info(rs.camera_info.serial_number)}")
        print(f"    Firmware: {device.get_info(rs.camera_info.firmware_version)}")
        
        # List available sensors
        sensors = device.query_sensors()
        print(f"    Sensors: {len(sensors)}")
        for j, sensor in enumerate(sensors):
            print(f"      Sensor {j}: {sensor.get_info(rs.camera_info.name)}")
    
    return True


def main():
    """Main function"""
    print("RealSense D455 Test Script")
    print("=" * 40)
    
    # Check for RealSense devices
    if not check_realsense_devices():
        print("\nPlease ensure:")
        print("1. RealSense D455 is connected via USB 3.0")
        print("2. RealSense SDK is properly installed")
        print("3. User has permission to access USB devices")
        return
    
    print("=" * 40)
    
    try:
        # Initialize and run test
        test = RealSenseTest()
        test.run_test()
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("\nTroubleshooting:")
        print("- Check USB connection (use USB 3.0 port)")
        print("- Ensure no other applications are using the camera")
        print("- Try running with sudo if permission issues occur")


if __name__ == "__main__":
    main()
