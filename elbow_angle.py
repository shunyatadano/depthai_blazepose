#!/usr/bin/env python3

from BlazeposeRenderer import BlazeposeRenderer
from mediapipe_utils import KEYPOINT_DICT
import argparse
import numpy as np
import math

# Constants for arm keypoints
LEFT_SHOULDER = KEYPOINT_DICT['left_shoulder']
RIGHT_SHOULDER = KEYPOINT_DICT['right_shoulder']
LEFT_ELBOW = KEYPOINT_DICT['left_elbow']
RIGHT_ELBOW = KEYPOINT_DICT['right_elbow']
LEFT_WRIST = KEYPOINT_DICT['left_wrist']
RIGHT_WRIST = KEYPOINT_DICT['right_wrist']
# Additional keypoints for pose classification
NOSE = KEYPOINT_DICT['nose']
LEFT_HIP = KEYPOINT_DICT['left_hip']
RIGHT_HIP = KEYPOINT_DICT['right_hip']

def calculate_angle_3d(point1, point2, point3):
    """Calculate angle between three 3D points"""
    # Convert to numpy arrays
    p1 = np.array(point1)
    p2 = np.array(point2)  # vertex point
    p3 = np.array(point3)
    
    # Calculate vectors
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Calculate angle
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def classify_pose_3d(landmarks_3d):
    """Classify pose based on 3D landmarks"""
    try:
        # Get key points
        nose = landmarks_3d[NOSE]
        left_shoulder = landmarks_3d[LEFT_SHOULDER]
        right_shoulder = landmarks_3d[RIGHT_SHOULDER]
        left_elbow = landmarks_3d[LEFT_ELBOW]
        right_elbow = landmarks_3d[RIGHT_ELBOW]
        left_wrist = landmarks_3d[LEFT_WRIST]
        right_wrist = landmarks_3d[RIGHT_WRIST]
        left_hip = landmarks_3d[LEFT_HIP]
        right_hip = landmarks_3d[RIGHT_HIP]
        
        # Calculate shoulder center and hip center
        shoulder_center = [(left_shoulder[i] + right_shoulder[i]) / 2 for i in range(3)]
        hip_center = [(left_hip[i] + right_hip[i]) / 2 for i in range(3)]
        
        # Calculate arm angles (shoulder-elbow-wrist)
        left_arm_angle = calculate_angle_3d(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle_3d(right_shoulder, right_elbow, right_wrist)
        
        # Calculate shoulder-elbow angles (relative to torso)
        left_shoulder_angle = calculate_angle_3d(shoulder_center, left_shoulder, left_elbow)
        right_shoulder_angle = calculate_angle_3d(shoulder_center, right_shoulder, right_elbow)
        
        # Calculate wrist heights relative to shoulders
        left_wrist_height = left_wrist[1] - left_shoulder[1]  # Y coordinate (negative means above)
        right_wrist_height = right_wrist[1] - right_shoulder[1]
        
        # Calculate torso angle (forward/backward lean)
        torso_vector = [shoulder_center[i] - hip_center[i] for i in range(3)]
        torso_angle = math.atan2(torso_vector[2], torso_vector[1])  # Z/Y ratio for forward lean
        torso_lean = math.degrees(torso_angle)
        
        print(f"--- Pose Analysis ---")
        print(f"Left arm angle: {left_arm_angle:.1f}°")
        print(f"Right arm angle: {right_arm_angle:.1f}°")
        print(f"Left shoulder angle: {left_shoulder_angle:.1f}°")
        print(f"Right shoulder angle: {right_shoulder_angle:.1f}°")
        print(f"Torso lean: {torso_lean:.1f}°")
        
        # Pose classification logic
        pose_detected = []
        
        # T-pose detection
        if (80 <= left_shoulder_angle <= 100 and 80 <= right_shoulder_angle <= 100 and
            160 <= left_arm_angle <= 200 and 160 <= right_arm_angle <= 200):
            pose_detected.append("T-POSE")
        
        # Banzai (hands up) detection
        elif (left_wrist_height < -50 and right_wrist_height < -50 and  # hands above shoulders
              left_shoulder_angle < 45 and right_shoulder_angle < 45):  # arms raised up
            pose_detected.append("BANZAI")
        
        # Forward lean detection
        if abs(torso_lean) > 30:  # significant forward/backward lean
            if torso_lean > 0:
                pose_detected.append("FORWARD_LEAN")
            else:
                pose_detected.append("BACKWARD_LEAN")
        
        # Default pose
        if not pose_detected:
            pose_detected.append("NEUTRAL")
        
        return pose_detected
        
    except Exception as e:
        print(f"Error in pose classification: {e}")
        return ["UNKNOWN"]

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--edge', action="store_true",
                    help="Use Edge mode (postprocessing runs on the device)")
parser_tracker = parser.add_argument_group("Tracker arguments")                 
parser_tracker.add_argument('-i', '--input', type=str, default="rgb", 
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default=%(default)s)")
parser_tracker.add_argument("--pd_m", type=str,
                    help="Path to an .blob file for pose detection model")
parser_tracker.add_argument("--lm_m", type=str,
                    help="Landmark model ('full' or 'lite' or 'heavy') or path to an .blob file")
parser_tracker.add_argument('-xyz', '--xyz', action="store_true", 
                    help="Get (x,y,z) coords of reference body keypoint in camera coord system (only for compatible devices)")
parser_tracker.add_argument('-c', '--crop', action="store_true", 
                    help="Center crop frames to a square shape before feeding pose detection model")
parser_tracker.add_argument('--no_smoothing', action="store_true", 
                    help="Disable smoothing filter")
parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument('--internal_frame_height', type=int, default=640,                                                                                    
                    help="Internal color camera frame height in pixels (default=%(default)i)")                    
parser_tracker.add_argument('-s', '--stats', action="store_true", 
                    help="Print some statistics at exit")
parser_tracker.add_argument('-t', '--trace', action="store_true", 
                    help="Print some debug messages")
parser_tracker.add_argument('--force_detection', action="store_true", 
                    help="Force person detection on every frame (never use landmarks from previous frame to determine ROI)")

parser_renderer = parser.add_argument_group("Renderer arguments")
parser_renderer.add_argument('-3', '--show_3d', choices=[None, "image", "world", "mixed"], default=None,
                    help="Display skeleton in 3d in a separate window. See README for description.")
parser_renderer.add_argument("-o","--output",
                    help="Path to output video file")
 

args = parser.parse_args()

from BlazeposeDepthaiEdge import BlazeposeDepthai
# if args.edge:
#     from BlazeposeDepthaiEdge import BlazeposeDepthai
# else:
#     from BlazeposeDepthai import BlazeposeDepthai

tracker = BlazeposeDepthai(input_src=args.input, 
            pd_model=args.pd_m,
            lm_model=args.lm_m,
            smoothing=not args.no_smoothing,   
            xyz=args.xyz,            
            crop=args.crop,
            internal_fps=args.internal_fps,
            internal_frame_height=args.internal_frame_height,
            force_detection=args.force_detection,
            stats=True,
            trace=args.trace)   

renderer = BlazeposeRenderer(
                tracker, 
                show_3d=args.show_3d, 
                output=args.output)

cnt = 0
while True:
    # Run blazepose on next frame
    frame, body = tracker.next_frame()
    if frame is None: break
    
    # Print arm keypoints coordinates if body is detected
    if body and cnt % 10 == 0:
        cnt += 1
        print(f"\n--- Body detected at frame ---")

        # Check if world landmarks are available (world coordinate system)
        if hasattr(body, 'landmarks_world') and body.landmarks_world is not None:
                # print("--- 3D World Coordinates (normalized units) ---")
                # print(f"Left shoulder:  {body.landmarks_world[LEFT_SHOULDER]}")
                # print(f"Left elbow:     {body.landmarks_world[LEFT_ELBOW]}")
                # print(f"Left wrist:     {body.landmarks_world[LEFT_WRIST]}")
                # print(f"Right shoulder: {body.landmarks_world[RIGHT_SHOULDER]}")
                # print(f"Right elbow:    {body.landmarks_world[RIGHT_ELBOW]}")
                # print(f"Right wrist:    {body.landmarks_world[RIGHT_WRIST]}")
                # Additional debug info for available attributes
                print(f"Available body attributes: {[attr for attr in dir(body) if not attr.startswith('_')]}")

                # Alternative pose classification using world landmarks
                if not (hasattr(body, 'xyz') and body.xyz is not None):
                    poses = classify_pose_3d(body.landmarks_world)
                    print(f"🌍 DETECTED POSE(S) (World): {', '.join(poses)}")
    
        
        #     # Check if landmarks are available
        #     if hasattr(body, 'landmarks') and len(body.landmarks) > max(LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST):
        #         print("--- 2D Normalized Coordinates ---")
        #         print(f"landmarks_world shape: {body.landmarks.shape}")
        #         print(f"Left shoulder:  {body.landmarks[LEFT_SHOULDER]}")
        #         print(f"Left shoulder:  {body.landmarks[LEFT_SHOULDER]}")
        #         print(f"Left elbow:     {body.landmarks[LEFT_ELBOW]}")
        #         print(f"Left wrist:     {body.landmarks[LEFT_WRIST]}")
        #         print(f"Right shoulder: {body.landmarks[RIGHT_SHOULDER]}")
        #         print(f"Right elbow:    {body.landmarks[RIGHT_ELBOW]}")
        #         print(f"Right wrist:    {body.landmarks[RIGHT_WRIST]}")
        
    # Draw 2d skeleton
    frame = renderer.draw(frame, body)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break
renderer.exit()
tracker.exit()