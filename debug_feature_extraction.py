#!/usr/bin/env python3
"""
特徴量抽出のデバッグスクリプト
学習データと現在の実装の特徴量形式を比較
"""

import numpy as np
import mediapipe as mp
import cv2
import sys

# 姿勢分類器をインポート
sys.path.append('/home/tamlab/ws_whill/src/depthai_blazepose/models_class')
from jetson_pose_classifier import PoseClassifier

# MediaPipeセットアップ
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

def extract_features_current(results):
    """現在の実装（座標順）"""
    if not results.pose_world_landmarks:
        return None

    landmarks_3d = []
    for landmark in results.pose_world_landmarks.landmark:
        landmarks_3d.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

    landmarks_array = np.array(landmarks_3d)
    x_coords = landmarks_array[:, 0]
    y_coords = landmarks_array[:, 1]
    z_coords = landmarks_array[:, 2]
    visibility = landmarks_array[:, 3]

    # 座標順: [x1,x2,...,x33, y1,y2,...,y33, z1,z2,...,z33, v1,v2,...,v33]
    return np.concatenate([x_coords, y_coords, z_coords, visibility])

def extract_features_correct(results):
    """正しい実装（キーポイント順）"""
    if not results.pose_world_landmarks:
        return None

    landmarks_3d = []
    for landmark in results.pose_world_landmarks.landmark:
        # キーポイント順: [x1,y1,z1,v1, x2,y2,z2,v2, ...]
        landmarks_3d.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

    return np.array(landmarks_3d)

def compare_feature_formats():
    """特徴量形式の比較"""
    print("=== 特徴量形式の比較 ===")

    # ダミーのpose_world_landmarksを作成
    class DummyLandmark:
        def __init__(self, x, y, z, v):
            self.x, self.y, self.z, self.visibility = x, y, z, v

    class DummyResults:
        def __init__(self):
            self.pose_world_landmarks = type('obj', (object,), {
                'landmark': [DummyLandmark(i*0.01, i*0.02, i*0.03, 0.9) for i in range(33)]
            })()

    dummy_results = DummyResults()

    # 現在の実装
    features_current = extract_features_current(dummy_results)
    # 正しい実装
    features_correct = extract_features_correct(dummy_results)

    print(f"現在の実装（座標順）: {features_current[:12]}")  # 最初の12要素
    print(f"正しい実装（キーポイント順）: {features_correct[:12]}")  # 最初の12要素

    print(f"\n差分の確認:")
    print(f"同じ値か: {np.allclose(features_current, features_correct)}")

    # 分類器でテスト
    classifier = PoseClassifier('/home/tamlab/ws_whill/src/depthai_blazepose/models_class')

    pred_current, prob_current = classifier.predict(features_current, return_proba=True)
    pred_correct, prob_correct = classifier.predict(features_correct, return_proba=True)

    print(f"\n=== 分類結果の比較 ===")
    print(f"現在の実装: {pred_current} ({prob_current})")
    print(f"正しい実装: {pred_correct} ({prob_correct})")

if __name__ == "__main__":
    compare_feature_formats()