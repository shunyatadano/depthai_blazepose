#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSense + 姿勢分類 プロトタイプ
mpandrs.pyベースの処理でリアルタイム姿勢分類を実行
"""

import numpy as np
import cv2
import mediapipe as mp
import pyrealsense2 as rs
import sys
import time
from pathlib import Path

# 姿勢分類器をインポート
sys.path.append('/home/tamlab/ws_whill/src/depthai_blazepose/models_class')
from jetson_pose_classifier import PoseClassifier

# MediaPipeの初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# カメラ設定（mpandrs.pyと同じ）
width = 1280
height = 720
fx = 643.2799682617188
fy = 641.89794921875
cx = 645.2747802734375
cy = 361.512939453125
person_threshold = 0.6

class RealSensePoseClassifier:
    def __init__(self):
        # MediaPipe設定（mpandrs.pyと同じ）
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )

        # 姿勢分類器の初期化
        model_dir = '/home/tamlab/ws_whill/src/depthai_blazepose/models_class'
        self.classifier = PoseClassifier(model_dir)

        # RealSenseの設定
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)

        # アライメント処理（深度をRGBに合わせる）
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # パフォーマンス計測
        self.frame_count = 0
        self.classification_count = 0
        self.fps_counter = 0
        self.fps_timer = time.time()

        print("🎥 RealSense姿勢分類器を初期化しました")
        print(f"📊 分類クラス: {self.classifier.class_names}")
        print(f"🎯 モデル精度: {self.classifier.config['performance']['test_accuracy']:.3f}")

    def get_landmark_depth(self, depth_image, x, y):
        """指定された座標の深度値を取得（mpandrs.pyと同じ安定化処理）"""
        try:
            if 0 <= x < width and 0 <= y < height:
                x_int, y_int = int(x), int(y)
                depth_values = []

                # 周辺3x3ピクセルの深度値を取得してノイズを削減
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
        """MediaPipeの結果から132次元特徴量を抽出（学習データと同じ形式）"""
        if not results.pose_world_landmarks:
            return None

        # MediaPipeの正規化3D座標を使用（学習時と同じ）
        landmarks_3d = []
        for landmark in results.pose_world_landmarks.landmark:
            landmarks_3d.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

        landmarks_array = np.array(landmarks_3d)  # (33, 4)

        # 132次元特徴量ベクトルに変換（キーポイント順: x1,y1,z1,v1, x2,y2,z2,v2, ...）
        features = []
        for i in range(33):
            features.extend([landmarks_array[i, 0], landmarks_array[i, 1],
                           landmarks_array[i, 2], landmarks_array[i, 3]])

        return np.array(features)

    def calculate_body_center(self, results):
        """胴体の中心を計算（mpandrs.pyと同じ）"""
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
        """メインループ"""
        try:
            # RealSenseパイプライン開始
            profile = self.pipeline.start(self.config)
            print("📹 RealSenseカメラを開始しました")
            print("🔄 姿勢分類実行中... (qキーで終了)")

            while True:
                # フレーム取得
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)

                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # NumPy配列に変換
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                self.frame_count += 1
                self.fps_counter += 1

                # MediaPipeで姿勢推定
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_image)

                # 結果の描画と分類
                if results.pose_landmarks:
                    # 骨格を描画
                    mp_drawing.draw_landmarks(
                        color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # 検出スコアの確認（mpandrs.pyと同じ）
                    visibility = np.mean([landmark.visibility for landmark in results.pose_landmarks.landmark])

                    if visibility > person_threshold:
                        # 胴体中心を計算・描画
                        center = self.calculate_body_center(results)
                        center_x, center_y = int(center[0]), int(center[1])
                        cv2.circle(color_image, (center_x, center_y), 8, (255, 0, 255), -1)

                        # 3フレームに1回分類実行（性能とのバランス）
                        if self.frame_count % 3 == 0:
                            features = self.extract_landmarks_features(results)
                            if features is not None:
                                try:
                                    prediction, probabilities = self.classifier.predict(features, return_proba=True)
                                    self.classification_count += 1

                                    # 結果を画面に描画
                                    confidence = probabilities[prediction]
                                    color = (0, 255, 0) if prediction == 'walk' else (0, 0, 255)

                                    # メイン結果
                                    cv2.putText(color_image, f"Pose: {prediction.upper()}", (20, 50),
                                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                                    cv2.putText(color_image, f"Confidence: {confidence:.3f}", (20, 90),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                                    # 詳細確率
                                    cv2.putText(color_image, f"Walk: {probabilities['walk']:.3f}", (20, 130),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                    cv2.putText(color_image, f"Phone: {probabilities['phone']:.3f}", (20, 160),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                                    # 深度情報（参考）
                                    person_depth = self.get_landmark_depth(depth_image, center_x, center_y)
                                    cv2.putText(color_image, f"Depth: {person_depth/1000:.2f}m", (20, 190),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                                except Exception as e:
                                    cv2.putText(color_image, f"Classification Error: {str(e)[:30]}", (20, 50),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        # 検出信頼度が低い場合
                        cv2.putText(color_image, "Low Detection Confidence", (20, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.putText(color_image, f"Visibility: {visibility:.3f}", (20, 80),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    # 人物が検出されない場合
                    cv2.putText(color_image, "No Person Detected", (20, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # FPS計算と表示
                current_time = time.time()
                if current_time - self.fps_timer >= 1.0:
                    fps = self.fps_counter / (current_time - self.fps_timer)
                    self.fps_counter = 0
                    self.fps_timer = current_time

                    cv2.putText(color_image, f"FPS: {fps:.1f}", (width-150, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 統計情報
                cv2.putText(color_image, f"Frames: {self.frame_count}", (width-200, height-60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(color_image, f"Classifications: {self.classification_count}", (width-200, height-40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(color_image, f"Model: RF (Acc: {self.classifier.config['performance']['test_accuracy']:.3f})",
                          (width-250, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # 画面表示
                cv2.imshow('RealSense Pose Classification', color_image)

                # 終了判定
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except Exception as e:
            print(f"❌ エラー: {e}")

        finally:
            # クリーンアップ
            self.pipeline.stop()
            self.pose.close()
            cv2.destroyAllWindows()

            print(f"\n📊 実行結果:")
            print(f"   - 総フレーム数: {self.frame_count}")
            print(f"   - 分類実行回数: {self.classification_count}")
            print(f"   - 分類実行率: {self.classification_count/self.frame_count*100:.1f}%")
            print("🏁 プロトタイプ終了")

def main():
    print("🚀 RealSense姿勢分類プロトタイプ起動中...")

    try:
        classifier = RealSensePoseClassifier()
        classifier.run()
    except KeyboardInterrupt:
        print("\n⏹️ ユーザーによる中断")
    except Exception as e:
        print(f"❌ 初期化エラー: {e}")
        print("💡 RealSenseカメラが接続されているか確認してください")

if __name__ == "__main__":
    main()