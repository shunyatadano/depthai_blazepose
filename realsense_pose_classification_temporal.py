#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSense + 時系列姿勢分類 (MVP版)
既存のrealsense_pose_classification.pyに時系列フィルタリングを追加
"""

import numpy as np
import cv2
import mediapipe as mp
import pyrealsense2 as rs
import sys
import time
from pathlib import Path

# 既存の分類器とフィルタをインポート
sys.path.append('/home/tamlab/ws_whill/src/depthai_blazepose/models_class')
from jetson_pose_classifier import PoseClassifier
from state_machine_pose_filter import StateMachinePoseFilter

# MediaPipeの初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# カメラ設定（既存と同じ）
width = 1280
height = 720
fx = 643.2799682617188
fy = 641.89794921875
cx = 645.2747802734375
cy = 361.512939453125
person_threshold = 0.6

class TemporalRealSensePoseClassifier:
    def __init__(self):
        # MediaPipe設定
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )

        # 姿勢分類器の初期化
        model_dir = '/home/tamlab/ws_whill/src/depthai_blazepose/models_class'
        self.classifier = PoseClassifier(model_dir)

        # **NEW: 状態遷移フィルタの初期化**
        self.temporal_filter = StateMachinePoseFilter(
            activation_frames=4,        # STABLE状態への遷移フレーム数
            deactivation_frames=5,      # UNSTABLE状態への復帰フレーム数
            confidence_threshold=0.6,   # 0.6以上の信頼度のみ使用
            smoothing_factor=0.1        # 半径スムージング係数
        )

        # RealSenseの設定
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)

        # アライメント処理
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # パフォーマンス計測
        self.frame_count = 0
        self.classification_count = 0
        self.stable_classification_count = 0
        self.fps_counter = 0
        self.fps_timer = time.time()

        print("📱 歩きスマホ検出システムを初期化しました")
        print(f"📊 分類クラス: {self.classifier.class_names}")
        print(f"🔄 状態遷移フィルタ: phone検出={self.temporal_filter.activation_frames}フレーム, 復帰=3フレーム")
        print(f"🎯 コストマップ半径: phone=0.7m, その他=0.4m")

    def get_landmark_depth(self, depth_image, x, y):
        """深度値取得（既存と同じ）"""
        try:
            if 0 <= x < width and 0 <= y < height:
                x_int, y_int = int(x), int(y)
                depth_values = []
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        nx, ny = x_int + dx, y_int + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            depth_val = depth_image[ny, nx]
                            if depth_val > 0:
                                depth_values.append(depth_val)
                if depth_values:
                    return np.median(depth_values)
                else:
                    return 0.0
            return 0.0
        except (IndexError, TypeError):
            return 0.0

    def extract_landmarks_features(self, results):
        """特徴量抽出（既存と同じ）"""
        if not results.pose_world_landmarks:
            return None
        landmarks_3d = []
        for landmark in results.pose_world_landmarks.landmark:
            landmarks_3d.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        landmarks_array = np.array(landmarks_3d)
        features = []
        for i in range(33):
            features.extend([landmarks_array[i, 0], landmarks_array[i, 1],
                           landmarks_array[i, 2], landmarks_array[i, 3]])
        return np.array(features)

    def calculate_body_center(self, results):
        """胴体中心計算（既存と同じ）"""
        landmarks = results.pose_landmarks.landmark
        left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * width,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * height])
        right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * width,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * height])
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height])
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height])
        center = (left_hip + right_hip + left_shoulder + right_shoulder) / 4
        return center

    def run(self):
        """メインループ（時系列フィルタリング統合版）"""
        try:
            profile = self.pipeline.start(self.config)
            print("📹 RealSenseカメラを開始しました")
            print("📱 歩きスマホ検出実行中... (qキーで終了)")

            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                self.frame_count += 1
                self.fps_counter += 1

                # MediaPipeで姿勢推定
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_image)

                # **NEW: デフォルト時系列結果**
                temporal_result = {
                    'pose': 'unknown',
                    'confidence': 0.0,
                    'is_stable': False,
                    'recommended_radius': 0.4
                }

                if results.pose_landmarks:
                    # 骨格描画
                    mp_drawing.draw_landmarks(
                        color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    visibility = np.mean([landmark.visibility for landmark in results.pose_landmarks.landmark])

                    if visibility > person_threshold:
                        center = self.calculate_body_center(results)
                        center_x, center_y = int(center[0]), int(center[1])
                        cv2.circle(color_image, (center_x, center_y), 8, (255, 0, 255), -1)

                        # **NEW: 時系列フィルタリング統合分類**
                        if self.frame_count % 2 == 0:  # 2フレームに1回（負荷軽減）
                            features = self.extract_landmarks_features(results)
                            if features is not None:
                                try:
                                    # 元の分類器で予測
                                    prediction, probabilities = self.classifier.predict(features, return_proba=True)
                                    confidence = probabilities[prediction]
                                    self.classification_count += 1

                                    # **NEW: 状態遷移フィルタで平滑化**
                                    temporal_result = self.temporal_filter.update(
                                        prediction, confidence, probabilities)

                                    # **DEBUG: 元の分類結果を保存**
                                    temporal_result['raw_prediction'] = prediction
                                    temporal_result['raw_confidence'] = confidence

                                    if temporal_result['is_stable']:
                                        self.stable_classification_count += 1

                                except Exception as e:
                                    print(f"分類エラー: {e}")

                        # **NEW: 時系列結果の描画**
                        self._draw_temporal_results(color_image, temporal_result, center_x, center_y, depth_image)

                    else:
                        cv2.putText(color_image, "Low Detection Confidence", (20, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(color_image, "No Person Detected", (20, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # FPS計算
                current_time = time.time()
                if current_time - self.fps_timer >= 1.0:
                    fps = self.fps_counter / (current_time - self.fps_timer)
                    self.fps_counter = 0
                    self.fps_timer = current_time
                    cv2.putText(color_image, f"FPS: {fps:.1f}", (width-150, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # **NEW: 時系列統計の表示**
                self._draw_statistics(color_image)

                cv2.imshow('Temporal Pose Classification', color_image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except Exception as e:
            print(f"❌ エラー: {e}")
        finally:
            self.pipeline.stop()
            self.pose.close()
            cv2.destroyAllWindows()
            self._print_final_stats()

    def _draw_temporal_results(self, image, temporal_result, center_x, center_y, depth_image):
        """時系列結果の描画"""
        pose = temporal_result['pose']
        confidence = temporal_result['confidence']
        is_stable = temporal_result['is_stable']
        radius = temporal_result['recommended_radius']

        # 歩きスマホ検出に応じた色分け
        if pose == 'phone':
            color = (0, 0, 255)  # 赤色（危険）
            status_text = "PHONE DETECTED (危険)"
        else:
            color = (0, 255, 0)  # 緑色（安全）
            status_text = "NORMAL"

        # メイン結果
        cv2.putText(image, f"Status: {status_text}", (20, 50),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.putText(image, f"Confidence: {confidence:.3f}", (20, 90),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # **DEBUG: 元の分類結果表示**
        if 'raw_prediction' in temporal_result:
            raw_pred = temporal_result['raw_prediction']
            raw_conf = temporal_result['raw_confidence']
            cv2.putText(image, f"Raw: {raw_pred.upper()} ({raw_conf:.3f})", (20, 130),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # **NEW: コストマップ半径情報**
        cv2.putText(image, f"Costmap Radius: {radius:.1f}m", (20, 160),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 深度情報
        person_depth = self.get_landmark_depth(depth_image, center_x, center_y)
        cv2.putText(image, f"Depth: {person_depth/1000:.2f}m", (20, 190),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # **NEW: 歩きスマホ警告インジケータ**
        warning_color = (0, 0, 255) if pose == 'phone' else (0, 255, 0)
        cv2.circle(image, (width-50, 50), 15, warning_color, -1)

    def _draw_statistics(self, image):
        """統計情報の描画"""
        stability_rate = (self.stable_classification_count / max(self.classification_count, 1)) * 100

        cv2.putText(image, f"Total: {self.frame_count}", (width-200, height-80),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Classifications: {self.classification_count}", (width-200, height-60),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Stable: {self.stable_classification_count} ({stability_rate:.1f}%)", (width-200, height-40),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # フィルタ統計
        filter_stats = self.temporal_filter.get_statistics()
        if filter_stats:
            cv2.putText(image, f"State: {filter_stats.get('current_state', 'unknown')}", (width-200, height-20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _print_final_stats(self):
        """最終統計の表示"""
        stability_rate = (self.stable_classification_count / max(self.classification_count, 1)) * 100
        print(f"\n📊 実行結果:")
        print(f"   - 総フレーム数: {self.frame_count}")
        print(f"   - 分類実行回数: {self.classification_count}")
        print(f"   - 安定分類回数: {self.stable_classification_count}")
        print(f"   - 安定性: {stability_rate:.1f}%")
        print("🏁 時系列姿勢分類終了")

def main():
    print("🚀 時系列姿勢分類プロトタイプ起動中...")
    try:
        classifier = TemporalRealSensePoseClassifier()
        classifier.run()
    except KeyboardInterrupt:
        print("\n⏹️ ユーザーによる中断")
    except Exception as e:
        print(f"❌ 初期化エラー: {e}")

if __name__ == "__main__":
    main()