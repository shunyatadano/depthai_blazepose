#!/usr/bin/env python3
"""
3D Skeletal Keypoint Estimation from Multiple Images using MediaPipe BlazePose
This script processes all images in data/cropped directory and performs pose estimation with visualization.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import pandas as pd
from tqdm import tqdm

# Import MediaPipe
import mediapipe as mp

# MediaPipe solutions
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set matplotlib to non-interactive backend
plt.ioff()

def normalize_landmarks(landmarks_3d):
    """Normalize 3D landmarks to [-1, 1] range"""
    # Find the center point (hip center)
    left_hip = landmarks_3d[23]
    right_hip = landmarks_3d[24]
    hip_center = (left_hip + right_hip) / 2
    
    # Center the landmarks
    centered_landmarks = landmarks_3d - hip_center
    
    # Calculate the scale factor based on the maximum distance from center
    max_distance = np.max(np.linalg.norm(centered_landmarks, axis=1))
    
    # Normalize to [-1, 1] range
    if max_distance > 0:
        normalized_landmarks = centered_landmarks / max_distance
    else:
        normalized_landmarks = centered_landmarks
    
    return normalized_landmarks

def process_single_image(image_path, pose, output_dir):
    """Process a single image and return pose data"""
    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Execute pose detection
        results = pose.process(image_rgb)
        
        if not results.pose_landmarks or not results.pose_world_landmarks:
            print(f"Warning: No pose detected in {image_path}")
            return None
        
        # Extract 3D world coordinates
        landmarks_3d = []
        visibility_scores = []
        for landmark in results.pose_world_landmarks.landmark:
            landmarks_3d.append([landmark.x, landmark.y, landmark.z])
        
        for landmark in results.pose_landmarks.landmark:
            visibility_scores.append(landmark.visibility)
        
        landmarks_3d = np.array(landmarks_3d)
        visibility_scores = np.array(visibility_scores)
        
        # Normalize landmarks
        normalized_landmarks = normalize_landmarks(landmarks_3d)
        
        # Create 2D/3D comparison plot
        create_comparison_plot(image, normalized_landmarks, output_dir, image_path.stem)
        
        return {
            'landmarks': normalized_landmarks,
            'visibility': visibility_scores,
            'success': True
        }
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def create_comparison_plot(image, landmarks_3d, output_dir, filename):
    """Create and save 2D/3D comparison plot"""
    try:
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
        
        # Left subplot: 2D image
        ax1 = fig.add_subplot(121)
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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
        ax2.set_title('3D Pose Estimation - World Coordinates (Normalized)', fontsize=14)
        ax2.legend()
        
        # Set equal aspect ratio for 3D plot
        max_range = 1.0  # Since landmarks are normalized to [-1, 1]
        ax2.set_xlim(-max_range, max_range)
        ax2.set_ylim(-max_range, max_range)
        ax2.set_zlim(-max_range, max_range)
        
        plt.tight_layout()
        
        # Save the comparison plot
        output_path = output_dir / f"{filename}_2d_3d_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
    except Exception as e:
        print(f"Error creating plot for {filename}: {e}")
        plt.close()

def main():
    print("MediaPipe version:", mp.__version__)
    
    # Base directories
    base_dir = Path("/home/eevee/depthai_blazepose")
    cropped_dir = base_dir / "data" / "cropped"
    plot_base_dir = base_dir / "data" / "plot"
    
    # Create main plot directory
    plot_base_dir.mkdir(exist_ok=True)
    
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
    except Exception as e:
        print(f"Error: MediaPipe Pose initialization failed: {e}")
        return
    
    # Prepare CSV data storage
    csv_data = []
    
    # Process both classes
    classes = ['phone', 'walk']
    
    for class_name in classes:
        class_dir = cropped_dir / class_name
        if not class_dir.exists():
            print(f"Warning: Directory {class_dir} does not exist")
            continue
        
        print(f"\nProcessing class: {class_name}")
        
        # Create output directory for this class
        plot_class_dir = plot_base_dir / class_name
        plot_class_dir.mkdir(exist_ok=True)
        
        # Find all scene directories
        scene_dirs = [d for d in class_dir.iterdir() if d.is_dir()]
        
        for scene_dir in tqdm(scene_dirs, desc=f"Processing {class_name} scenes"):
            scene_name = scene_dir.name
            
            # Create output directory for this scene
            scene_output_dir = plot_class_dir / scene_name
            scene_output_dir.mkdir(exist_ok=True)
            
            # Find all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(scene_dir.glob(ext))
            
            print(f"  Processing scene {scene_name}: {len(image_files)} images")
            
            for image_file in tqdm(image_files, desc=f"  {scene_name}", leave=False):
                # Process image
                result = process_single_image(image_file, pose, scene_output_dir)
                
                if result and result['success']:
                    # Prepare CSV row
                    csv_row = {
                        'scene': scene_name,
                        'image': image_file.stem,
                        'class': class_name
                    }
                    
                    # Add normalized landmarks and visibility
                    landmarks = result['landmarks']
                    visibility = result['visibility']
                    
                    for i in range(33):  # MediaPipe has 33 landmarks
                        if i < len(landmarks):
                            csv_row[f'x{i+1}'] = landmarks[i][0]
                            csv_row[f'y{i+1}'] = landmarks[i][1]
                            csv_row[f'z{i+1}'] = landmarks[i][2]
                            csv_row[f'v{i+1}'] = visibility[i] if i < len(visibility) else 0.0
                        else:
                            csv_row[f'x{i+1}'] = 0.0
                            csv_row[f'y{i+1}'] = 0.0
                            csv_row[f'z{i+1}'] = 0.0
                            csv_row[f'v{i+1}'] = 0.0
                    
                    csv_data.append(csv_row)
    
    # Save CSV data
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = plot_base_dir / "pose_landmarks_3d_normalized.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ CSV data saved to: {csv_path}")
        print(f"✓ Total processed images: {len(csv_data)}")
    else:
        print("\n⚠ No data to save to CSV")
    
    # Resource cleanup
    print("\nCleaning up resources...")
    try:
        pose.close()
        print("✓ Cleanup completed")
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    print("\n=== Processing Complete ===")
    print("MediaPipe BlazePose batch processing completed successfully!")
    print(f"Plots saved to: {plot_base_dir}")

if __name__ == "__main__":
    main()