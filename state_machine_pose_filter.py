#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
状態遷移ベースの姿勢分類フィルタ (MVP版)
ジッタを抑制し、安定した姿勢分類結果を提供
"""

from enum import Enum
import time

class PoseState(Enum):
    """歩きスマホ検出の状態定義"""
    NORMAL = "normal"
    PHONE_DETECTED = "phone_detected"

class StateMachinePoseFilter:
    """
    状態遷移モデルベースの姿勢分類フィルタ
    ヒステリシスを使用してジッタを抑制
    """

    def __init__(self, activation_frames=4, deactivation_frames=5,
                 confidence_threshold=0.6, smoothing_factor=0.1):
        """
        Args:
            activation_frames (int): STABLE状態への遷移に必要な連続フレーム数
            deactivation_frames (int): UNSTABLE状態への復帰に必要な連続フレーム数
            confidence_threshold (float): 信頼度の最低閾値
            smoothing_factor (float): 半径スムージングの係数 (0-1)
        """
        # 状態遷移パラメータ
        self.activation_frames = activation_frames
        self.deactivation_frames = deactivation_frames
        self.confidence_threshold = confidence_threshold
        self.smoothing_factor = smoothing_factor

        # 状態管理
        self.current_state = PoseState.NORMAL
        self.consecutive_count = 0
        self.last_detected_pose = 'unknown'

        # コストマップ半径設定（歩きスマホ検出特化）
        self.radius_targets = {
            'phone': 0.7,
            'other': 0.4
        }
        self.current_radius = self.radius_targets['other']

        print(f"🔄 StateMachinePoseFilter初期化完了")
        print(f"   - 活性化フレーム: {activation_frames}")
        print(f"   - 非活性化フレーム: {deactivation_frames}")
        print(f"   - 信頼度閾値: {confidence_threshold}")

    def update(self, prediction, confidence, probabilities=None):
        """
        姿勢分類結果を更新し、フィルタリング後の結果を返す

        Args:
            prediction (str): 予測クラス ('walk', 'phone')
            confidence (float): 予測の信頼度
            probabilities (dict): クラス別確率 (オプション)

        Returns:
            dict: {
                'pose': str,              # フィルタ済み姿勢
                'is_stable': bool,        # 安定状態かどうか
                'recommended_radius': float # 推奨コストマップ半径
            }
        """
        # 信頼度チェック
        if confidence < self.confidence_threshold:
            prediction = 'unknown'

        # 状態遷移ロジック
        self._update_state(prediction)

        # 半径スムージング
        self._update_radius()

        # フィルタ結果の決定
        filtered_pose = self._get_filtered_pose()
        is_stable = self.current_state != PoseState.UNSTABLE

        return {
            'pose': filtered_pose,
            'is_stable': is_stable,
            'recommended_radius': round(self.current_radius, 2),
            'confidence': confidence  # 元の信頼度を保持
        }

    def _update_state(self, prediction):
        """歩きスマホ検出の状態遷移更新"""
        if self.current_state == PoseState.NORMAL:
            # NORMAL状態: phone検出を監視
            if prediction == 'phone':
                self.consecutive_count += 1
                if self.consecutive_count >= self.activation_frames:
                    # phone検出状態に遷移
                    self.current_state = PoseState.PHONE_DETECTED
                    self.consecutive_count = 0
            else:
                # phone以外はカウントリセット
                self.consecutive_count = 0

        else:  # PHONE_DETECTED状態
            # PHONE_DETECTED状態: phone以外の検出を監視
            if prediction != 'phone':
                self.consecutive_count += 1
                if self.consecutive_count >= 3:  # phone以外3フレームで復帰
                    # NORMAL状態に復帰
                    self.current_state = PoseState.NORMAL
                    self.consecutive_count = 0
            else:
                # phone検出継続
                self.consecutive_count = 0

    def _update_radius(self):
        """コストマップ半径のスムージング更新"""
        filtered_pose = self._get_filtered_pose()
        target_radius = self.radius_targets[filtered_pose] if filtered_pose == 'phone' else self.radius_targets['other']

        # 線形補間によるスムージング
        self.current_radius += self.smoothing_factor * (target_radius - self.current_radius)

    def _get_filtered_pose(self):
        """現在の状態に基づくフィルタ済み姿勢を取得"""
        if self.current_state == PoseState.PHONE_DETECTED:
            return 'phone'
        else:
            return 'normal'

    def get_statistics(self):
        """統計情報を取得"""
        return {
            'current_state': self.current_state.value,
            'consecutive_count': self.consecutive_count,
            'current_radius': round(self.current_radius, 2),
            'last_detected': self.last_detected_pose
        }