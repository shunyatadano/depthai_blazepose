#!/usr/bin/env python3
"""
MediaPipe 3D Pose Estimation with RealSense D455 (No Open3D)
for Jetson AGX Orin (JetPack 5.1.2)

This script demonstrates real-time 3D pose estimation using MediaPipe
with RealSense D455 camera, without Open3D for visualization.
This version is designed to work around Open3D compatibility issues on Jetson.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import time
from typing import Optional, Tuple, List
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class RealSense3DPoseTrackerLite:
    """3D Pose Tracker using RealSense D455 and MediaPipe (without Open3D)"""
    
    def __init__(self, 
                 width: int = 640, 
                 height: int = 480, 
                 fps: int = 30,
                 enable_depth: bool = True):
        """
        Initialize the 3D pose tracker
        
        Args:
            width: Image width
            height: Image height 
            fps: Frame rate
            enable_depth: Whether to use depth information
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configure pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Use lighter model for better performance
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if enable_depth:
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            
        # Start pipeline
        self.profile = self.pipeline.start(self.config)
        
        # Get camera intrinsics
        if enable_depth:
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = self.depth_sensor.get_depth_scale()
            
            # Get depth and color intrinsics
            self.color_intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.depth_intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            
            # Create alignment object
            self.align = rs.align(rs.stream.color)
            
        # Initialize pose connections for visualization
        self.pose_connections = [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),  # nose to left ear
            (0, 4), (4, 5), (5, 6), (6, 8),  # nose to right ear
            (9, 10),  # mouth
            
            # Arms
            (11, 12),  # shoulders
            (11, 13), (13, 15),  # left arm
            (12, 14), (14, 16),  # right arm
            (15, 17), (15, 19), (15, 21),  # left hand
            (16, 18), (16, 20), (16, 22),  # right hand
            
            # Torso
            (11, 23), (12, 24),  # shoulder to hip
            (23, 24),  # hips
            
            # Legs
            (23, 25), (25, 27),  # left leg
            (24, 26), (26, 28),  # right leg
            (27, 29), (27, 31),  # left foot
            (28, 30), (28, 32),  # right foot
        ]
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        # 3D coordinates history for analysis
        self.pose_history = []
        
    def get_3d_coordinates(self, landmarks, depth_frame) -> List[Tuple[float, float, float]]:
        """
        Convert 2D landmarks to 3D coordinates using depth information
        
        Args:
            landmarks: MediaPipe pose landmarks
            depth_frame: RealSense depth frame
            
        Returns:
            List of 3D coordinates (x, y, z) in meters
        """
        coordinates_3d = []
        
        for landmark in landmarks.landmark:
            # Convert normalized coordinates to pixel coordinates
            x_pixel = int(landmark.x * self.width)
            y_pixel = int(landmark.y * self.height)
            
            # Ensure coordinates are within frame bounds
            x_pixel = max(0, min(x_pixel, self.width - 1))
            y_pixel = max(0, min(y_pixel, self.height - 1))
            
            if self.enable_depth:
                # Get depth value at the landmark position
                depth_value = depth_frame.get_distance(x_pixel, y_pixel)
                
                if depth_value > 0:
                    # Convert pixel coordinates to 3D world coordinates
                    point_3d = rs.rs2_deproject_pixel_to_point(
                        self.color_intrinsics, [x_pixel, y_pixel], depth_value
                    )
                    coordinates_3d.append((point_3d[0], point_3d[1], point_3d[2]))
                else:
                    # Use MediaPipe's world coordinates if depth is invalid
                    coordinates_3d.append((landmark.x - 0.5, -(landmark.y - 0.5), landmark.z))
            else:
                # Use MediaPipe's world coordinates
                coordinates_3d.append((landmark.x - 0.5, -(landmark.y - 0.5), landmark.z))
                
        return coordinates_3d
    
    def draw_3d_info_on_2d(self, image, coordinates_3d):
        """
        Draw 3D coordinate information on 2D image
        
        Args:
            image: 2D image to draw on
            coordinates_3d: List of 3D coordinates
        """
        h, w = image.shape[:2]
        
        if len(coordinates_3d) == 0:
            return
            
        # Calculate center of mass in 3D
        center_x = np.mean([coord[0] for coord in coordinates_3d])
        center_y = np.mean([coord[1] for coord in coordinates_3d])
        center_z = np.mean([coord[2] for coord in coordinates_3d])
        
        # Display 3D center of mass
        cv2.putText(image, f"3D Center: ({center_x:.2f}, {center_y:.2f}, {center_z:.2f})", 
                   (10, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Display key landmarks in 3D
        if len(coordinates_3d) > 0:  # Nose
            nose_3d = coordinates_3d[0]
            cv2.putText(image, f"Nose 3D: ({nose_3d[0]:.2f}, {nose_3d[1]:.2f}, {nose_3d[2]:.2f})", 
                       (10, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if len(coordinates_3d) > 11:  # Left shoulder
            shoulder_3d = coordinates_3d[11]
            cv2.putText(image, f"L.Shoulder 3D: ({shoulder_3d[0]:.2f}, {shoulder_3d[1]:.2f}, {shoulder_3d[2]:.2f})", 
                       (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if len(coordinates_3d) > 12:  # Right shoulder
            shoulder_3d = coordinates_3d[12]
            cv2.putText(image, f"R.Shoulder 3D: ({shoulder_3d[0]:.2f}, {shoulder_3d[1]:.2f}, {shoulder_3d[2]:.2f})", 
                       (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Calculate and display distances
        if len(coordinates_3d) > 12:
            left_shoulder = np.array(coordinates_3d[11])
            right_shoulder = np.array(coordinates_3d[12])
            shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
            cv2.putText(image, f"Shoulder Width: {shoulder_distance:.2f}m", 
                       (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def plot_3d_skeleton(self, coordinates_3d, save_path: str = None):
        """
        Create a 3D plot of the skeleton using matplotlib
        
        Args:
            coordinates_3d: List of 3D coordinates
            save_path: Path to save the plot (optional)
        """
        if len(coordinates_3d) == 0:
            return
            
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract coordinates
        xs = [coord[0] for coord in coordinates_3d]
        ys = [coord[1] for coord in coordinates_3d]
        zs = [coord[2] for coord in coordinates_3d]
        
        # Plot points
        ax.scatter(xs, ys, zs, c='red', marker='o', s=50)
        
        # Draw connections
        for connection in self.pose_connections:
            if connection[0] < len(coordinates_3d) and connection[1] < len(coordinates_3d):
                p1 = coordinates_3d[connection[0]]
                p2 = coordinates_3d[connection[1]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-', linewidth=2)
        
        # Set labels and title
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('3D Pose Skeleton')
        
        # Set equal aspect ratio
        max_range = np.array([xs, ys, zs]).max()
        min_range = np.array([xs, ys, zs]).min()
        ax.set_xlim([min_range, max_range])
        ax.set_ylim([min_range, max_range])
        ax.set_zlim([min_range, max_range])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"3D plot saved to {save_path}")
        else:
            plt.show(block=False)
            plt.pause(0.001)
    
    def save_pose_data(self, coordinates_3d: List[Tuple[float, float, float]], timestamp: float, filename: str = "pose_data.json"):
        """
        Save pose data to JSON file
        
        Args:
            coordinates_3d: List of 3D coordinates
            timestamp: Frame timestamp
            filename: Output filename
        """
        pose_data = {
            "timestamp": timestamp,
            "landmarks": [{"x": coord[0], "y": coord[1], "z": coord[2]} for coord in coordinates_3d]
        }
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {"poses": []}
            
        data["poses"].append(pose_data)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def run(self, save_data: bool = False, show_2d: bool = True, show_3d_plot: bool = False):
        """
        Main tracking loop
        
        Args:
            save_data: Whether to save pose data to file
            show_2d: Whether to show 2D visualization window
            show_3d_plot: Whether to show 3D matplotlib plot
        """
        print("Starting 3D pose tracking (Lite version without Open3D)...")
        print("Press 'q' to quit, 's' to save current pose, 'p' to show 3D plot")
        
        plt.ion() if show_3d_plot else None  # Enable interactive plotting
        
        try:
            while True:
                # Get frames from RealSense
                frames = self.pipeline.wait_for_frames()
                
                if self.enable_depth:
                    # Align depth frame to color frame
                    aligned_frames = self.align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                else:
                    color_frame = frames.get_color_frame()
                    depth_frame = None
                
                if not color_frame:
                    continue
                
                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                
                # Convert BGR to RGB for MediaPipe
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                rgb_image.flags.writeable = False
                
                # Process pose detection
                results = self.pose.process(rgb_image)
                
                # Convert back to BGR for OpenCV
                rgb_image.flags.writeable = True
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                
                if results.pose_landmarks:
                    # Get 3D coordinates
                    coordinates_3d = self.get_3d_coordinates(results.pose_landmarks, depth_frame)
                    
                    # Add to history
                    self.pose_history.append({
                        'timestamp': time.time(),
                        'coordinates': coordinates_3d
                    })
                    
                    # Keep only recent history (last 100 frames)
                    if len(self.pose_history) > 100:
                        self.pose_history = self.pose_history[-100:]
                    
                    # Save pose data if requested
                    if save_data:
                        timestamp = time.time()
                        self.save_pose_data(coordinates_3d, timestamp)
                    
                    # Draw 2D pose on image
                    if show_2d:
                        self.mp_drawing.draw_landmarks(
                            bgr_image,
                            results.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                        )
                        
                        # Draw 3D information on 2D image
                        self.draw_3d_info_on_2d(bgr_image, coordinates_3d)
                
                # Calculate and display FPS
                self.frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                if elapsed_time > 0:
                    fps = self.frame_count / elapsed_time
                    cv2.putText(bgr_image, f"FPS: {fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show 2D image
                if show_2d:
                    cv2.imshow('MediaPipe Pose (Lite)', bgr_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and results.pose_landmarks:
                    # Save current pose
                    coordinates_3d = self.get_3d_coordinates(results.pose_landmarks, depth_frame)
                    timestamp = time.time()
                    self.save_pose_data(coordinates_3d, timestamp, f"pose_{int(timestamp)}.json")
                    print(f"Saved pose data to pose_{int(timestamp)}.json")
                elif key == ord('p') and results.pose_landmarks:
                    # Show 3D plot
                    coordinates_3d = self.get_3d_coordinates(results.pose_landmarks, depth_frame)
                    self.plot_3d_skeleton(coordinates_3d, f"3d_pose_{int(time.time())}.png")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.pipeline.stop()
        cv2.destroyAllWindows()
        plt.close('all')


def main():
    """Main function"""
    # Configuration
    config = {
        'width': 640,
        'height': 480,
        'fps': 30,
        'enable_depth': True,
        'save_data': False,
        'show_2d': True,
        'show_3d_plot': False
    }
    
    print("MediaPipe 3D Pose Tracker (Lite) for Jetson AGX Orin")
    print("=" * 60)
    print(f"Resolution: {config['width']}x{config['height']}")
    print(f"FPS: {config['fps']}")
    print(f"Depth enabled: {config['enable_depth']}")
    print("Note: This version uses matplotlib instead of Open3D for 3D visualization")
    print("=" * 60)
    
    try:
        # Initialize tracker
        tracker = RealSense3DPoseTrackerLite(
            width=config['width'],
            height=config['height'],
            fps=config['fps'],
            enable_depth=config['enable_depth']
        )
        
        # Run tracking
        tracker.run(
            save_data=config['save_data'],
            show_2d=config['show_2d'],
            show_3d_plot=config['show_3d_plot']
        )
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure RealSense D455 is connected and MediaPipe is installed")


if __name__ == "__main__":
    main()
