#!/usr/bin/env python3
"""
Simple demo script for MediaPipe 3D Pose Detection
Tests basic functionality without RealSense camera (using webcam)
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from typing import List, Tuple


class SimplePoseDemo:
    """Simple pose detection demo using webcam"""
    
    def __init__(self, camera_id: int = 0):
        """Initialize the demo with webcam"""
        self.camera_id = camera_id
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configure pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Use lighter model for demo
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
    def get_landmark_coordinates(self, landmarks) -> List[Tuple[float, float, float]]:
        """Extract 3D coordinates from MediaPipe landmarks"""
        coordinates = []
        for landmark in landmarks.landmark:
            coordinates.append((landmark.x, landmark.y, landmark.z))
        return coordinates
    
    def draw_pose_info(self, image, landmarks):
        """Draw additional pose information on the image"""
        h, w = image.shape[:2]
        
        # Calculate center of mass
        if landmarks:
            coords = self.get_landmark_coordinates(landmarks)
            center_x = np.mean([coord[0] for coord in coords]) * w
            center_y = np.mean([coord[1] for coord in coords]) * h
            
            # Draw center of mass
            cv2.circle(image, (int(center_x), int(center_y)), 10, (255, 255, 0), -1)
            cv2.putText(image, "Center of Mass", (int(center_x) + 15, int(center_y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Draw bounding box
            x_coords = [coord[0] * w for coord in coords]
            y_coords = [coord[1] * h for coord in coords]
            
            min_x, max_x = int(min(x_coords)), int(max(x_coords))
            min_y, max_y = int(min(y_coords)), int(max(y_coords))
            
            cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 255), 2)
            
            # Display some landmark coordinates
            nose = coords[0]  # Nose landmark
            cv2.putText(image, f"Nose: ({nose[0]:.2f}, {nose[1]:.2f}, {nose[2]:.2f})", 
                       (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if len(coords) > 11:  # Left shoulder
                left_shoulder = coords[11]
                cv2.putText(image, f"L.Shoulder: ({left_shoulder[0]:.2f}, {left_shoulder[1]:.2f}, {left_shoulder[2]:.2f})", 
                           (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if len(coords) > 12:  # Right shoulder
                right_shoulder = coords[12]
                cv2.putText(image, f"R.Shoulder: ({right_shoulder[0]:.2f}, {right_shoulder[1]:.2f}, {right_shoulder[2]:.2f})", 
                           (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Run the demo"""
        print("Starting MediaPipe Pose Demo...")
        print("Press 'q' to quit, 'i' to toggle info display")
        
        show_info = True
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                
                # Process pose detection
                results = self.pose.process(rgb_frame)
                
                # Convert back to BGR
                rgb_frame.flags.writeable = True
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                
                # Draw pose landmarks
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        bgr_frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # Draw additional info if enabled
                    if show_info:
                        self.draw_pose_info(bgr_frame, results.pose_landmarks)
                
                # Calculate and display FPS
                self.frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                if elapsed_time > 0:
                    fps = self.frame_count / elapsed_time
                    cv2.putText(bgr_frame, f"FPS: {fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display instructions
                h, w = bgr_frame.shape[:2]
                cv2.putText(bgr_frame, "Press 'q' to quit, 'i' for info", (10, h - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('MediaPipe Pose Demo', bgr_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('i'):
                    show_info = not show_info
                    print(f"Info display: {'ON' if show_info else 'OFF'}")
                    
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()


def test_imports():
    """Test if all required modules can be imported"""
    print("Testing module imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print(f"✓ MediaPipe version: {mp.__version__}")
    except ImportError as e:
        print(f"✗ MediaPipe import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    return True


def main():
    """Main function"""
    print("MediaPipe Pose Detection Demo")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("Some required modules are missing. Please install them first.")
        return
    
    print("All modules imported successfully!")
    print("=" * 40)
    
    # Try different camera IDs
    camera_ids_to_try = [0, 1, 2, 3, 4, 5]
    demo = None
    
    for camera_id in camera_ids_to_try:
        try:
            print(f"Trying camera ID: {camera_id}")
            demo = SimplePoseDemo(camera_id=camera_id)
            print(f"✓ Successfully initialized camera {camera_id}")
            break
        except ValueError as e:
            print(f"✗ Camera {camera_id} failed: {e}")
            continue
    
    if demo is None:
        print("Failed to initialize any camera. Please check connections.")
        return
    
    try:
        demo.run()
    except Exception as e:
        print(f"Unexpected error during demo: {e}")
        if demo:
            demo.cleanup()


if __name__ == "__main__":
    main()
