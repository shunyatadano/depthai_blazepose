import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
from models_class.jetson_pose_classifier import PoseClassifier

#!/usr/bin/env python3

class MediaPipe3DSkeleton:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start the pipeline
        self.profile = self.pipeline.start(self.config)
        
        # Initialize pose classifier
        self.pose_classifier = PoseClassifier()
        
        # For displaying the classification results
        self.classification_result = None
        self.result_probs = {}
        self.result_color = (0, 255, 0)  # Green by default

    def process_frame(self):
        # Get frameset
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            return None, None, None
        
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        
        # Process the color image with MediaPipe Pose
        results = self.pose.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            return color_image, None, None
        
        # Get normalized 3D keypoints from MediaPipe with visibility
        keypoints_3d = []
        
        if results.pose_world_landmarks:
            # Get MediaPipe's 3D pose estimation with visibility
            for i, landmark in enumerate(results.pose_world_landmarks.landmark):
                # Add visibility from pose_landmarks
                visibility = results.pose_landmarks.landmark[i].visibility if results.pose_landmarks else 0.0
                keypoints_3d.append([landmark.x, landmark.y, landmark.z, visibility])
        
        # Draw the pose landmarks on color image
        self.mp_drawing.draw_landmarks(
            color_image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Classify pose if keypoints are available
        pose_class = None
        if keypoints_3d and len(keypoints_3d) == 33:
            keypoints_array = np.array(keypoints_3d)
            pose_class, probs = self.pose_classifier.predict(keypoints_array, return_proba=True)
            self.classification_result = pose_class
            self.result_probs = probs
            
            # Set color based on class (緑：歩行、赤：歩きスマホ)
            if pose_class == "walk":
                self.result_color = (0, 255, 0)  # Green for walking
            else:
                self.result_color = (0, 0, 255)  # Red for phone usage
        
        return color_image, np.array(keypoints_3d) if keypoints_3d else None, pose_class

    def run(self):
        try:
            while True:
                color_image, keypoints_3d, pose_class = self.process_frame()
                
                if color_image is None:
                    continue
                
                # Display classification results on the image
                if pose_class is not None:
                    # Get probabilities for each class
                    walk_prob = self.result_probs.get('walk', 0) * 100
                    phone_prob = self.result_probs.get('phone', 0) * 100
                    
                    # Add text to image
                    cv2.putText(color_image, f"Class: {pose_class}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.result_color, 2)
                    cv2.putText(color_image, f"Walk: {walk_prob:.1f}%", 
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(color_image, f"Phone: {phone_prob:.1f}%", 
                                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                if keypoints_3d is not None:
                    # Example: print position of nose keypoint (optional)
                    # print(f"Nose position (x,y,z): {keypoints_3d[0][:3]}")
                    pass
                
                # Display color image
                cv2.imshow('MediaPipe 3D Pose', color_image)
                
                if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
                    break
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    tracker = MediaPipe3DSkeleton()
    tracker.run()
