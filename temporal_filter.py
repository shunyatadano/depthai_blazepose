#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小限の時系列フィルタリングクラス
コストマップ変調のための安定した姿勢分類を提供
"""

import numpy as np
from collections import deque
import time

class TemporalPoseFilter:
    """MVP版: シンプルな滑動窓+信頼度フィルタリング"""

    def __init__(self, window_size=5, confidence_threshold=0.7, min_stable_frames=3):
        """
        Args:
            window_size: 滑動窓のサイズ
            confidence_threshold: 最小信頼度閾値
            min_stable_frames: 状態変化に必要な最小フレーム数
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.min_stable_frames = min_stable_frames

        # 履歴データ
        self.prediction_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        self.timestamp_history = deque(maxlen=window_size)

        # 現在の安定状態
        self.current_stable_pose = "unknown"
        self.stable_frame_count = 0
        self.last_update_time = 0

        # コストマップ変調用パラメータ
        self.radius_map = {
            "walk": 0.5,    # 歩行中: 標準半径
            "phone": 0.3,   # 電話中: 小さい半径（静止傾向）
            "unknown": 0.4  # 不明: 中間値
        }

    def update(self, prediction, confidence, probabilities=None):
        """
        新しい予測結果でフィルタを更新

        Args:
            prediction: 予測結果 ("walk", "phone", etc.)
            confidence: 信頼度 [0.0-1.0]
            probabilities: 各クラスの確率辞書 (optional)

        Returns:
            dict: {
                'pose': 安定化された姿勢,
                'confidence': 平滑化された信頼度,
                'is_stable': 安定性フラグ,
                'recommended_radius': コストマップ推奨半径
            }
        """
        current_time = time.time()

        # 低信頼度の予測を除外
        if confidence < self.confidence_threshold:
            prediction = "unknown"
            confidence = 0.0

        # 履歴に追加
        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)
        self.timestamp_history.append(current_time)

        # 履歴が不十分な場合はunknownを返す
        if len(self.prediction_history) < self.min_stable_frames:
            return self._build_result("unknown", confidence, False)

        # 滑動窓内での多数決
        window_predictions = list(self.prediction_history)
        most_common = max(set(window_predictions), key=window_predictions.count)

        # 信頼度の平均
        avg_confidence = np.mean(list(self.confidence_history))

        # 安定性判定
        is_stable = self._update_stability(most_common)

        # 最終的な姿勢決定
        final_pose = self.current_stable_pose if is_stable else "unknown"

        self.last_update_time = current_time

        return self._build_result(final_pose, avg_confidence, is_stable)

    def _update_stability(self, predicted_pose):
        """安定性を更新"""
        if predicted_pose == self.current_stable_pose:
            self.stable_frame_count += 1
        else:
            # 新しい姿勢が連続で検出された場合のみ変更
            recent_predictions = list(self.prediction_history)[-self.min_stable_frames:]
            if all(p == predicted_pose for p in recent_predictions):
                self.current_stable_pose = predicted_pose
                self.stable_frame_count = self.min_stable_frames
            else:
                self.stable_frame_count = 0

        return self.stable_frame_count >= self.min_stable_frames

    def _build_result(self, pose, confidence, is_stable):
        """結果辞書を構築"""
        return {
            'pose': pose,
            'confidence': float(confidence),
            'is_stable': is_stable,
            'recommended_radius': self.radius_map.get(pose, 0.4)
        }

    def reset(self):
        """フィルタをリセット"""
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.timestamp_history.clear()
        self.current_stable_pose = "unknown"
        self.stable_frame_count = 0

    def get_statistics(self):
        """デバッグ用統計情報"""
        if not self.prediction_history:
            return {}

        return {
            'window_size': len(self.prediction_history),
            'current_stable_pose': self.current_stable_pose,
            'stable_frame_count': self.stable_frame_count,
            'avg_confidence': np.mean(list(self.confidence_history)),
            'recent_predictions': list(self.prediction_history)
        }

# テスト用関数
def test_temporal_filter():
    """基本的なテスト"""
    print("🧪 TemporalPoseFilter テスト開始")

    filter = TemporalPoseFilter(window_size=5, min_stable_frames=3)

    # テストシーケンス: ノイジーな入力 → 安定した出力
    test_data = [
        ("walk", 0.8),
        ("phone", 0.6),  # 低信頼度
        ("walk", 0.9),
        ("walk", 0.8),
        ("walk", 0.9),   # ここで安定化されるはず
        ("phone", 0.9),
        ("phone", 0.8),
        ("phone", 0.9),  # 新しい安定状態
    ]

    for i, (pred, conf) in enumerate(test_data):
        result = filter.update(pred, conf)
        print(f"Step {i+1}: Input({pred}, {conf:.1f}) → {result['pose']} "
              f"(stable: {result['is_stable']}, radius: {result['recommended_radius']:.1f}m)")

    print("✅ テスト完了")
    return True

if __name__ == "__main__":
    test_temporal_filter()