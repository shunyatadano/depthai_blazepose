#!/usr/bin/env python3
"""
MediaPipe 3D Pose Estimation with RealSense D455
for Jetson AGX Orin (JetPack 5.1.2)

This script demonstrates real-time 3D pose estimation using MediaPipe
with RealSense D455 camera, including depth information for accurate
3D coordinate calculation.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import time
import open3d as o3d
from typing import Optional, Tuple, List
import json


class RealSense3DPoseTracker:
    """3D Pose Tracker using RealSense D455 and MediaPipe"""
    
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
            model_complexity=2,  # 0, 1, or 2 (higher = more accurate but slower)
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
            
        # Initialize Open3D visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("3D Pose Visualization", width=800, height=600)
        
        # Create coordinate frame
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.vis.add_geometry(self.coord_frame)
        
        # Initialize pose connections for 3D visualization
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
        
        # Colors for different body parts
        self.colors = {
            'face': [1.0, 0.8, 0.8],      # Light pink
            'left_arm': [0.8, 0.2, 0.2],  # Red
            'right_arm': [0.2, 0.8, 0.2], # Green
            'torso': [0.2, 0.2, 0.8],     # Blue
            'left_leg': [0.8, 0.2, 0.8],  # Magenta
            'right_leg': [0.8, 0.8, 0.2], # Yellow
        }
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
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
    
    def create_3d_skeleton(self, coordinates_3d: List[Tuple[float, float, float]]) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.LineSet]:
        """
        Create 3D skeleton visualization from 3D coordinates
        
        Args:
            coordinates_3d: List of 3D coordinates
            
        Returns:
            Tuple of (point cloud, line set) for visualization
        """
        # Create point cloud for landmarks
        point_cloud = o3d.geometry.PointCloud()
        points = np.array(coordinates_3d)
        point_cloud.points = o3d.utility.Vector3dVector(points)
        
        # Color the points
        colors = []
        for i in range(len(coordinates_3d)):
            if i <= 10:  # Face
                colors.append(self.colors['face'])
            elif i in [11, 13, 15, 17, 19, 21]:  # Left arm
                colors.append(self.colors['left_arm'])
            elif i in [12, 14, 16, 18, 20, 22]:  # Right arm
                colors.append(self.colors['right_arm'])
            elif i in [23, 24]:  # Torso
                colors.append(self.colors['torso'])
            elif i in [23, 25, 27, 29, 31]:  # Left leg
                colors.append(self.colors['left_leg'])
            elif i in [24, 26, 28, 30, 32]:  # Right leg
                colors.append(self.colors['right_leg'])
            else:
                colors.append([0.5, 0.5, 0.5])  # Default gray
                
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        # Create line set for skeleton connections
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        
        # Add lines for pose connections
        lines = []
        line_colors = []
        for connection in self.pose_connections:
            if connection[0] < len(coordinates_3d) and connection[1] < len(coordinates_3d):
                lines.append(connection)
                # Choose color based on body part
                if connection[0] <= 10 or connection[1] <= 10:
                    line_colors.append(self.colors['face'])
                elif (connection[0] in [11, 13, 15, 17, 19, 21] and 
                      connection[1] in [11, 13, 15, 17, 19, 21]):
                    line_colors.append(self.colors['left_arm'])
                elif (connection[0] in [12, 14, 16, 18, 20, 22] and 
                      connection[1] in [12, 14, 16, 18, 20, 22]):
                    line_colors.append(self.colors['right_arm'])
                else:
                    line_colors.append([0.7, 0.7, 0.7])
                    
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        
        return point_cloud, line_set
    
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
    
    def run(self, save_data: bool = False, show_2d: bool = True):
        """
        Main tracking loop
        
        Args:
            save_data: Whether to save pose data to file
            show_2d: Whether to show 2D visualization window
        """
        print("Starting 3D pose tracking...")
        print("Press 'q' to quit, 's' to save current pose, 'r' to reset view")
        
        # Initialize 3D visualization objects
        current_point_cloud = None
        current_line_set = None
        
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
                    
                    # Update 3D visualization
                    new_point_cloud, new_line_set = self.create_3d_skeleton(coordinates_3d)
                    
                    # Remove previous geometry
                    if current_point_cloud is not None:
                        self.vis.remove_geometry(current_point_cloud, reset_bounding_box=False)
                    if current_line_set is not None:
                        self.vis.remove_geometry(current_line_set, reset_bounding_box=False)
                    
                    # Add new geometry
                    self.vis.add_geometry(new_point_cloud, reset_bounding_box=False)
                    self.vis.add_geometry(new_line_set, reset_bounding_box=False)
                    
                    current_point_cloud = new_point_cloud
                    current_line_set = new_line_set
                    
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
                    cv2.imshow('MediaPipe Pose', bgr_image)
                
                # Update 3D visualization
                self.vis.poll_events()
                self.vis.update_renderer()
                
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
                elif key == ord('r'):
                    # Reset 3D view
                    self.vis.reset_view_point(True)
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.pipeline.stop()
        cv2.destroyAllWindows()
        self.vis.destroy_window()


def main():
    """Main function"""
    # Configuration
    config = {
        'width': 640,
        'height': 480,
        'fps': 30,
        'enable_depth': True,
        'save_data': False,
        'show_2d': True
    }
    
    print("MediaPipe 3D Pose Tracker for Jetson AGX Orin")
    print("=" * 50)
    print(f"Resolution: {config['width']}x{config['height']}")
    print(f"FPS: {config['fps']}")
    print(f"Depth enabled: {config['enable_depth']}")
    print("=" * 50)
    
    try:
        # Initialize tracker
        tracker = RealSense3DPoseTracker(
            width=config['width'],
            height=config['height'],
            fps=config['fps'],
            enable_depth=config['enable_depth']
        )
        
        # Run tracking
        tracker.run(
            save_data=config['save_data'],
            show_2d=config['show_2d']
        )
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure RealSense D455 is connected and MediaPipe is installed")


if __name__ == "__main__":
    main()