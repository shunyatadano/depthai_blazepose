#!/usr/bin/env python3
"""
3D Skeletal Keypoint Estimation from Single Image using MediaPipe BlazePose
This script processes a single image and performs pose estimation with visualization.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Import MediaPipe
import mediapipe as mp

# MediaPipe solutions
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def main():
    print("MediaPipe version:", mp.__version__)
    
    # Image path configuration
    image_path = "/home/eevee/depthai_blazepose/data/cropped/phone/clark-center-2019-02-28_1/000197.jpg"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    print(f"Image file found: {image_path}")
    
    # Load and display image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title("Input Image")
    plt.axis('off')
    
    # Save input image plot
    plot_dir = Path("/home/eevee/depthai_blazepose/data/plot")
    plot_dir.mkdir(exist_ok=True)
    plt.savefig(plot_dir / "input_image.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Image size: {img.shape[1]} x {img.shape[0]}")
    
    # Initialize MediaPipe Pose
    try:
        pose = mp_pose.Pose(
            static_image_mode=True,  # For single image
            model_complexity=2,      # High accuracy model
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("✓ MediaPipe Pose initialization completed")
        
        # Read image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Execute pose detection
        print("Executing pose detection...")
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            print("✓ Human body detection successful")
            print(f"Number of detected landmarks: {len(results.pose_landmarks.landmark)}")
            
            # Save results
            pose_landmarks = results.pose_landmarks
            pose_world_landmarks = results.pose_world_landmarks
            
        else:
            print("Warning: No human body detected")
            pose_landmarks = None
            pose_world_landmarks = None
            
    except Exception as e:
        print(f"Error: MediaPipe Pose initialization failed: {e}")
        pose_landmarks = None
        pose_world_landmarks = None
        return
    
    # 3D Skeleton Visualization with side-by-side comparison (Cell 7)
    if pose_world_landmarks is not None:
        print("Drawing 3D skeleton with 2D comparison...")
        
        # Extract 3D coordinates
        landmarks_3d = []
        for landmark in pose_world_landmarks.landmark:
            landmarks_3d.append([landmark.x, landmark.y, landmark.z])
        
        landmarks_3d = np.array(landmarks_3d)
        
        # Define pose connections for 3D visualization
        pose_connections = [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10),
            # Arms
            (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
            # Body
            (11, 23), (12, 24), (23, 24),
            # Legs
            (23, 25), (25, 27), (27, 29), (27, 31),
            (24, 26), (26, 28), (28, 30), (28, 32)
        ]
        
        # Create side-by-side comparison plot
        fig = plt.figure(figsize=(20, 8))
        
        # Left subplot: 2D image with pose annotation
        ax1 = fig.add_subplot(121)
        annotated_image = image.copy()
        ax1.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('2D Pose Detection - Input Image', fontsize=14)
        ax1.axis('off')
        
        # Right subplot: 3D skeleton
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Plot landmarks
        ax2.scatter(landmarks_3d[:, 0], landmarks_3d[:, 2], -landmarks_3d[:, 1], 
                  c='red', s=50, alpha=0.8, label='Landmarks')
        
        # Plot connections
        for connection in pose_connections:
            if connection[0] < len(landmarks_3d) and connection[1] < len(landmarks_3d):
                point1 = landmarks_3d[connection[0]]
                point2 = landmarks_3d[connection[1]]
                ax2.plot([point1[0], point2[0]], 
                       [point1[2], point2[2]], 
                       [-point1[1], -point2[1]], 'b-', alpha=0.7)
        
        # Set labels and title for 3D plot
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y') 
        ax2.set_zlabel('Z')
        ax2.set_title('3D Pose Estimation - World Coordinates', fontsize=14)
        ax2.legend()
        
        # Set equal aspect ratio for 3D plot
        max_range = np.array([landmarks_3d[:, 0].max()-landmarks_3d[:, 0].min(),
                             landmarks_3d[:, 1].max()-landmarks_3d[:, 1].min(),
                             landmarks_3d[:, 2].max()-landmarks_3d[:, 2].min()]).max() / 2.0
        mid_x = (landmarks_3d[:, 0].max()+landmarks_3d[:, 0].min()) * 0.5
        mid_y = (landmarks_3d[:, 1].max()+landmarks_3d[:, 1].min()) * 0.5
        mid_z = (landmarks_3d[:, 2].max()+landmarks_3d[:, 2].min()) * 0.5
        
        ax2.set_xlim(mid_x - max_range, mid_x + max_range)
        ax2.set_ylim(mid_y - max_range, mid_y + max_range)
        ax2.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        # Save the comparison plot
        plt.savefig(plot_dir / "2d_3d_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        
    else:
        print("3D coordinate data not available")
    
    # Detailed Keypoint Information Display (Cell 8)
    if pose_landmarks is not None:
        print("\n=== Detailed Keypoint Information ===")
        
        # MediaPipe keypoint names mapping
        keypoint_names = [
            "Nose", "Left Eye (Inner)", "Left Eye", "Left Eye (Outer)", 
            "Right Eye (Inner)", "Right Eye", "Right Eye (Outer)",
            "Left Ear", "Right Ear", "Mouth (Left)", "Mouth (Right)", 
            "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
            "Left Wrist", "Right Wrist", "Left Pinky", "Right Pinky", 
            "Left Index", "Right Index", "Left Thumb", "Right Thumb",
            "Left Hip", "Right Hip", "Left Knee", "Right Knee",
            "Left Ankle", "Right Ankle", "Left Heel", "Right Heel", 
            "Left Foot Index", "Right Foot Index"
        ]
        
        print("\n2D Landmark Coordinates (x, y, z, visibility):")
        for i, (name, landmark) in enumerate(zip(keypoint_names, pose_landmarks.landmark)):
            print(f"{i:2d}. {name:<18}: x={landmark.x:.4f}, y={landmark.y:.4f}, z={landmark.z:.4f}, v={landmark.visibility:.3f}")
        
        # Display 3D World coordinates if available
        if pose_world_landmarks is not None:
            print("\n3D World Coordinates (key joints):")
            key_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # Key joints
            for i in key_indices:
                if i < len(pose_world_landmarks.landmark):
                    world_landmark = pose_world_landmarks.landmark[i]
                    print(f"{keypoint_names[i]:<18}: x={world_landmark.x:8.4f}, y={world_landmark.y:8.4f}, z={world_landmark.z:8.4f}")
    else:
        print("Keypoint information not available")
    
    # Joint Angle Calculations (Cell 9)
    if pose_landmarks is not None:
        print("\n=== Joint Angle Calculation Examples ===")
        
        def calculate_angle(p1, p2, p3):
            """Calculate angle between three points (in degrees)"""
            v1 = np.array([p1.x, p1.y, p1.z]) - np.array([p2.x, p2.y, p2.z])
            v2 = np.array([p3.x, p3.y, p3.z]) - np.array([p2.x, p2.y, p2.z])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
            angle = np.arccos(cos_angle)
            return np.degrees(angle)
        
        try:
            landmarks = pose_landmarks.landmark
            
            # Left elbow angle (shoulder-elbow-wrist)
            left_shoulder = landmarks[11]  # Left shoulder
            left_elbow = landmarks[13]     # Left elbow  
            left_wrist = landmarks[15]     # Left wrist
            
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            print(f"Left Elbow Angle: {left_elbow_angle:.1f}°")
            
            # Right elbow angle (shoulder-elbow-wrist)
            right_shoulder = landmarks[12]  # Right shoulder
            right_elbow = landmarks[14]     # Right elbow
            right_wrist = landmarks[16]     # Right wrist
            
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            print(f"Right Elbow Angle: {right_elbow_angle:.1f}°")
            
            # Left knee angle (hip-knee-ankle)
            left_hip = landmarks[23]     # Left hip
            left_knee = landmarks[25]    # Left knee
            left_ankle = landmarks[27]   # Left ankle
            
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            print(f"Left Knee Angle: {left_knee_angle:.1f}°")
            
            # Right knee angle (hip-knee-ankle)
            right_hip = landmarks[24]     # Right hip
            right_knee = landmarks[26]    # Right knee
            right_ankle = landmarks[28]   # Right ankle
            
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            print(f"Right Knee Angle: {right_knee_angle:.1f}°")
            
            # Visualize angles on a bar chart
            angles = [left_elbow_angle, right_elbow_angle, left_knee_angle, right_knee_angle]
            labels = ['Left Elbow', 'Right Elbow', 'Left Knee', 'Right Knee']
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(labels, angles, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
            plt.title('Joint Angles Analysis')
            plt.ylabel('Angle (degrees)')
            plt.ylim(0, 180)
            
            # Add value labels on bars
            for bar, angle in zip(bars, angles):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                        f'{angle:.1f}°', ha='center', va='bottom')
            
            plt.grid(axis='y', alpha=0.3)
            
            # Save joint angles plot
            plt.savefig(plot_dir / "joint_angles.png", dpi=150, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error during angle calculation: {e}")
    else:
        print("Skipping joint angle calculations")
    
    # Pose Confidence and Visibility Analysis (Cell 11)
    if pose_landmarks is not None:
        print("\n=== Pose Confidence and Visibility Analysis ===")
        
        # Extract visibility scores
        visibility_scores = [landmark.visibility for landmark in pose_landmarks.landmark]
        
        # Calculate statistics
        avg_visibility = np.mean(visibility_scores)
        min_visibility = np.min(visibility_scores)
        max_visibility = np.max(visibility_scores)
        
        print(f"Average Visibility Score: {avg_visibility:.3f}")
        print(f"Minimum Visibility Score: {min_visibility:.3f}")
        print(f"Maximum Visibility Score: {max_visibility:.3f}")
        
        # Create visibility histogram
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Visibility histogram
        plt.subplot(1, 2, 1)
        plt.hist(visibility_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Landmark Visibility Score Distribution')
        plt.xlabel('Visibility Score')
        plt.ylabel('Number of Landmarks')
        plt.grid(alpha=0.3)
        
        # Subplot 2: Visibility per landmark
        plt.subplot(1, 2, 2)
        landmark_indices = range(len(visibility_scores))
        colors = ['red' if v < 0.5 else 'orange' if v < 0.7 else 'green' for v in visibility_scores]
        plt.bar(landmark_indices, visibility_scores, color=colors, alpha=0.7)
        plt.title('Visibility Score per Landmark')
        plt.xlabel('Landmark Index')
        plt.ylabel('Visibility Score')
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Low visibility threshold')
        plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Medium visibility threshold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save visibility analysis plot
        plt.savefig(plot_dir / "visibility_analysis.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        # Show low visibility landmarks
        low_visibility_landmarks = [(i, score) for i, score in enumerate(visibility_scores) if score < 0.5]
        if low_visibility_landmarks:
            print(f"\nLandmarks with low visibility (< 0.5):")
            keypoint_names = [
                "Nose", "Left Eye (Inner)", "Left Eye", "Left Eye (Outer)", 
                "Right Eye (Inner)", "Right Eye", "Right Eye (Outer)",
                "Left Ear", "Right Ear", "Mouth (Left)", "Mouth (Right)", 
                "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
                "Left Wrist", "Right Wrist", "Left Pinky", "Right Pinky", 
                "Left Index", "Right Index", "Left Thumb", "Right Thumb",
                "Left Hip", "Right Hip", "Left Knee", "Right Knee",
                "Left Ankle", "Right Ankle", "Left Heel", "Right Heel", 
                "Left Foot Index", "Right Foot Index"
            ]
            for idx, score in low_visibility_landmarks:
                print(f"  {idx:2d}. {keypoint_names[idx]:<18}: {score:.3f}")
        else:
            print("\nAll landmarks have good visibility (≥ 0.5)")
            
    else:
        print("Pose confidence analysis not available")
    
    # Processing statistics and cleanup (Cell 12)
    print("\n=== Processing Statistics ===")
    
    try:
        if pose_landmarks is not None:
            print(f"✓ Pose detection: Success")
            print(f"✓ Number of detected landmarks: {len(pose_landmarks.landmark)}")
            
            if pose_world_landmarks is not None:
                print(f"✓ 3D World coordinates: Available")
            else:
                print(f"⚠ 3D World coordinates: Not available")
                
            # Calculate average visibility
            avg_visibility = np.mean([landmark.visibility for landmark in pose_landmarks.landmark])
            print(f"✓ Average landmark visibility: {avg_visibility:.3f}")
            
        else:
            print(f"✗ Pose detection: Failed")
            
    except Exception as e:
        print(f"Error during statistics generation: {e}")
    
    # Resource cleanup
    print("\nCleaning up resources...")
    try:
        if 'pose' in locals():
            pose.close()
        print("✓ Cleanup completed")
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    print("\n=== Processing Complete ===")
    print("MediaPipe BlazePose single image inference completed successfully!")
    print(f"Plots saved to: {plot_dir}")

if __name__ == "__main__":
    main()