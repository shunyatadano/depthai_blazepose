#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正した姿勢分類の性能確認テスト
RealSenseカメラなしでも実行可能
"""

import numpy as np
import sys
import time
from pathlib import Path

sys.path.append('/home/tamlab/ws_whill/src/depthai_blazepose/models_class')
from jetson_pose_classifier import PoseClassifier

class PoseClassificationPerformanceTest:
    def __init__(self):
        """テストクラス初期化"""
        self.model_dir = '/home/tamlab/ws_whill/src/depthai_blazepose/models_class'
        self.classifier = PoseClassifier(self.model_dir)

        print("🤖 姿勢分類性能テスト初期化")
        print(f"📊 クラス: {self.classifier.class_names}")
        print(f"🎯 モデル精度: {self.classifier.config['performance']['test_accuracy']:.3f}")

    def create_walk_like_pose(self):
        """歩行らしい姿勢データを作成"""
        # 歩行時の特徴を模擬
        landmarks = np.zeros((33, 4))

        # 主要ランドマーク（MediaPipeの順序）
        # 顔（0-10）
        for i in range(11):
            landmarks[i] = [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.5, -0.3),
                           np.random.uniform(-0.1, 0.1), np.random.uniform(0.8, 1.0)]

        # 肩（11, 12）
        landmarks[11] = [-0.2, -0.2, 0.0, 0.9]  # 左肩
        landmarks[12] = [0.2, -0.2, 0.0, 0.9]   # 右肩

        # 肘（13, 14）
        landmarks[13] = [-0.3, 0.0, 0.1, 0.8]   # 左肘
        landmarks[14] = [0.3, 0.0, 0.1, 0.8]    # 右肘

        # 手首（15, 16）- 歩行時は体の横で振る
        landmarks[15] = [-0.35, 0.2, 0.2, 0.7]  # 左手首
        landmarks[16] = [0.35, 0.2, 0.2, 0.7]   # 右手首

        # 指（17-22）- 歩行時はリラックス
        for i in range(17, 23):
            side = -1 if i < 20 else 1
            landmarks[i] = [side * 0.4, 0.25, 0.25, 0.5]

        # 腰（23, 24）
        landmarks[23] = [-0.15, 0.4, 0.0, 0.9]  # 左腰
        landmarks[24] = [0.15, 0.4, 0.0, 0.9]   # 右腰

        # 膝（25, 26）
        landmarks[25] = [-0.2, 0.7, 0.0, 0.8]   # 左膝
        landmarks[26] = [0.2, 0.7, 0.0, 0.8]    # 右膝

        # 足首（27, 28）
        landmarks[27] = [-0.2, 1.0, 0.0, 0.7]   # 左足首
        landmarks[28] = [0.2, 1.0, 0.0, 0.7]    # 右足首

        # 足（29-32）
        for i in range(29, 33):
            side = -1 if i in [29, 31] else 1
            landmarks[i] = [side * 0.2, 1.1, 0.0, 0.6]

        return landmarks

    def create_phone_like_pose(self):
        """スマホを使っている姿勢データを作成"""
        # スマホ使用時の特徴を模擬（片手でスマホ、もう片手は下）
        landmarks = np.zeros((33, 4))

        # 顔（0-10）- やや下向き
        for i in range(11):
            landmarks[i] = [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.3, -0.1),
                           np.random.uniform(-0.1, 0.1), np.random.uniform(0.8, 1.0)]

        # 肩（11, 12）
        landmarks[11] = [-0.2, -0.2, 0.0, 0.9]  # 左肩
        landmarks[12] = [0.2, -0.2, 0.0, 0.9]   # 右肩

        # 肘（13, 14）- 右肘は曲がってスマホ持ち
        landmarks[13] = [-0.25, 0.1, 0.1, 0.7]  # 左肘（下）
        landmarks[14] = [0.4, -0.1, 0.2, 0.9]   # 右肘（上、曲がっている）

        # 手首（15, 16）- 右手首は顔の前（スマホ持ち）
        landmarks[15] = [-0.25, 0.3, 0.1, 0.6]  # 左手首（下）
        landmarks[16] = [0.1, -0.2, 0.3, 0.9]   # 右手首（顔の前）

        # 指（17-22）- 右手の指は集中（スマホタップ）
        # 左手の指（17-19）
        for i in range(17, 20):
            landmarks[i] = [-0.3, 0.35, 0.15, 0.4]

        # 右手の指（20-22）- スマホ操作中
        for i in range(20, 23):
            landmarks[i] = [0.15, -0.15, 0.35, 0.7]

        # 腰以下は歩行と同様
        landmarks[23] = [-0.15, 0.4, 0.0, 0.9]  # 左腰
        landmarks[24] = [0.15, 0.4, 0.0, 0.9]   # 右腰

        landmarks[25] = [-0.2, 0.7, 0.0, 0.8]   # 左膝
        landmarks[26] = [0.2, 0.7, 0.0, 0.8]    # 右膝

        landmarks[27] = [-0.2, 1.0, 0.0, 0.7]   # 左足首
        landmarks[28] = [0.2, 1.0, 0.0, 0.7]    # 右足首

        for i in range(29, 33):
            side = -1 if i in [29, 31] else 1
            landmarks[i] = [side * 0.2, 1.1, 0.0, 0.6]

        return landmarks

    def test_classification_accuracy(self, n_samples=100):
        """分類精度をテスト"""
        print(f"\n🎯 分類精度テスト (サンプル数: {n_samples})")

        correct_walk = 0
        correct_phone = 0

        # walkサンプルのテスト
        for _ in range(n_samples):
            walk_pose = self.create_walk_like_pose()
            prediction = self.classifier.predict(walk_pose)
            if prediction == 'walk':
                correct_walk += 1

        # phoneサンプルのテスト
        for _ in range(n_samples):
            phone_pose = self.create_phone_like_pose()
            prediction = self.classifier.predict(phone_pose)
            if prediction == 'phone':
                correct_phone += 1

        walk_accuracy = correct_walk / n_samples
        phone_accuracy = correct_phone / n_samples
        overall_accuracy = (correct_walk + correct_phone) / (2 * n_samples)

        print(f"📊 結果:")
        print(f"   Walk分類精度: {walk_accuracy:.3f} ({correct_walk}/{n_samples})")
        print(f"   Phone分類精度: {phone_accuracy:.3f} ({correct_phone}/{n_samples})")
        print(f"   総合精度: {overall_accuracy:.3f}")

        return overall_accuracy

    def test_inference_speed(self, n_iterations=1000):
        """推論速度をテスト"""
        print(f"\n⚡ 推論速度テスト (反復数: {n_iterations})")

        # テストデータ準備
        test_pose = self.create_walk_like_pose()

        # ウォームアップ
        for _ in range(10):
            self.classifier.predict(test_pose)

        # 実際の計測
        start_time = time.time()
        for _ in range(n_iterations):
            prediction = self.classifier.predict(test_pose, return_proba=True)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / n_iterations
        fps = 1 / avg_time

        print(f"📈 結果:")
        print(f"   総実行時間: {total_time:.3f}秒")
        print(f"   平均推論時間: {avg_time*1000:.2f}ms")
        print(f"   推論FPS: {fps:.1f}")

        return avg_time

    def test_probability_distribution(self, n_samples=50):
        """確率分布の確認"""
        print(f"\n📊 確率分布テスト (サンプル数: {n_samples})")

        walk_probs = []
        phone_probs = []

        # walkサンプル
        for _ in range(n_samples):
            walk_pose = self.create_walk_like_pose()
            _, probabilities = self.classifier.predict(walk_pose, return_proba=True)
            walk_probs.append(probabilities['walk'])

        # phoneサンプル
        for _ in range(n_samples):
            phone_pose = self.create_phone_like_pose()
            _, probabilities = self.classifier.predict(phone_pose, return_proba=True)
            phone_probs.append(probabilities['phone'])

        walk_mean = np.mean(walk_probs)
        walk_std = np.std(walk_probs)
        phone_mean = np.mean(phone_probs)
        phone_std = np.std(phone_probs)

        print(f"📋 Walk姿勢でのWalk確率:")
        print(f"   平均: {walk_mean:.3f} ± {walk_std:.3f}")
        print(f"   範囲: {np.min(walk_probs):.3f} - {np.max(walk_probs):.3f}")

        print(f"📋 Phone姿勢でのPhone確率:")
        print(f"   平均: {phone_mean:.3f} ± {phone_std:.3f}")
        print(f"   範囲: {np.min(phone_probs):.3f} - {np.max(phone_probs):.3f}")

        return walk_mean, phone_mean

    def run_full_test(self):
        """全テストを実行"""
        print("🚀 姿勢分類性能テスト開始")
        print("=" * 60)

        # テスト1: 分類精度
        accuracy = self.test_classification_accuracy(100)

        # テスト2: 推論速度
        avg_time = self.test_inference_speed(1000)

        # テスト3: 確率分布
        walk_conf, phone_conf = self.test_probability_distribution(50)

        print("\n" + "=" * 60)
        print("📊 性能テスト結果サマリー")
        print(f"   🎯 分類精度: {accuracy:.3f}")
        print(f"   ⚡ 推論時間: {avg_time*1000:.2f}ms")
        print(f"   📈 Walk信頼度: {walk_conf:.3f}")
        print(f"   📈 Phone信頼度: {phone_conf:.3f}")

        # 判定
        if accuracy > 0.7 and avg_time < 0.05:
            print("🎉 性能テスト通過! 修正は成功しています")
            return True
        else:
            print("⚠️  性能に課題があります")
            if accuracy <= 0.7:
                print(f"   - 精度が低い: {accuracy:.3f} <= 0.7")
            if avg_time >= 0.05:
                print(f"   - 推論時間が長い: {avg_time*1000:.2f}ms >= 50ms")
            return False

def main():
    try:
        tester = PoseClassificationPerformanceTest()
        success = tester.run_full_test()

        if success:
            print("\n✅ 修正したモデルは正常に動作しています!")
        else:
            print("\n❌ 修正に問題がある可能性があります")

    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        print(f"詳細: {traceback.format_exc()}")

if __name__ == "__main__":
    main()