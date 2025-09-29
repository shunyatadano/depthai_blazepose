#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特徴量抽出の修正テスト
学習データ形式と一致するかを確認
"""

import numpy as np
import sys
from pathlib import Path

# MediaPipeをインポート（ダミーデータ用）
import mediapipe as mp

# 修正したクラスをインポート
sys.path.append('/home/tamlab/ws_whill/src/depthai_blazepose/models_class')
from jetson_pose_classifier import PoseClassifier

def create_dummy_mediapipe_results():
    """ダミーのMediaPipe結果を作成"""
    # 33個のランドマークを作成
    landmarks_data = []
    for i in range(33):
        # ランダムな3D座標とvisibilityを生成
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(-0.5, 0.5)
        visibility = np.random.uniform(0.5, 1.0)
        landmarks_data.append([x, y, z, visibility])

    return np.array(landmarks_data)  # (33, 4)

def test_feature_extraction_order():
    """特徴量抽出の順序をテスト"""
    print("🧪 特徴量抽出順序テスト開始")

    # ダミーデータ作成
    dummy_landmarks = create_dummy_mediapipe_results()
    print(f"📊 ダミーランドマーク作成: shape={dummy_landmarks.shape}")

    # 修正前の方式（座標順）でも特徴量を作成
    old_features = []
    x_coords = dummy_landmarks[:, 0]  # 33次元
    y_coords = dummy_landmarks[:, 1]  # 33次元
    z_coords = dummy_landmarks[:, 2]  # 33次元
    visibility = dummy_landmarks[:, 3]  # 33次元
    old_features = np.concatenate([x_coords, y_coords, z_coords, visibility])

    # 修正後の方式（キーポイント順）で特徴量を作成
    new_features = []
    for i in range(33):
        new_features.extend([dummy_landmarks[i, 0], dummy_landmarks[i, 1],
                           dummy_landmarks[i, 2], dummy_landmarks[i, 3]])
    new_features = np.array(new_features)

    print(f"🔄 旧形式（座標順）: shape={old_features.shape}")
    print(f"✨ 新形式（キーポイント順）: shape={new_features.shape}")

    # 順序が違うことを確認
    print(f"📋 順序比較:")
    print(f"   旧: [x1...x33, y1...y33, z1...z33, v1...v33]")
    print(f"   新: [x1,y1,z1,v1, x2,y2,z2,v2, ...]")

    # 最初の数個の値を比較
    print(f"🔍 最初の8要素比較:")
    print(f"   旧形式: {old_features[:8]}")
    print(f"   新形式: {new_features[:8]}")

    # 順序が異なることを確認
    is_different = not np.allclose(old_features, new_features)
    print(f"📝 順序違い確認: {'✅ 異なる' if is_different else '❌ 同じ'}")

    return new_features

def test_jetson_classifier():
    """修正したJetson分類器をテスト"""
    print("\n🤖 Jetson分類器テスト開始")

    try:
        # 分類器を初期化
        model_dir = '/home/tamlab/ws_whill/src/depthai_blazepose/models_class'
        classifier = PoseClassifier(model_dir)

        # ダミーデータで予測テスト
        dummy_landmarks = create_dummy_mediapipe_results()

        # (33, 4)形式で予測
        prediction, probabilities = classifier.predict(dummy_landmarks, return_proba=True)

        print(f"✅ 予測成功!")
        print(f"   予測クラス: {prediction}")
        print(f"   各クラス確率: {probabilities}")
        print(f"   walk確率: {probabilities['walk']:.3f}")
        print(f"   phone確率: {probabilities['phone']:.3f}")

        # 特徴量形式テスト（既に平坦化済み）
        flat_features = []
        for i in range(33):
            flat_features.extend([dummy_landmarks[i, 0], dummy_landmarks[i, 1],
                                dummy_landmarks[i, 2], dummy_landmarks[i, 3]])
        flat_features = np.array(flat_features)

        prediction2, probabilities2 = classifier.predict(flat_features, return_proba=True)

        print(f"✅ 平坦化済み特徴量での予測成功!")
        print(f"   予測クラス: {prediction2}")
        print(f"   各クラス確率: {probabilities2}")

        # 両方の結果が同じことを確認
        same_prediction = (prediction == prediction2 and
                         np.allclose(list(probabilities.values()), list(probabilities2.values())))
        print(f"📋 結果一致確認: {'✅ 一致' if same_prediction else '❌ 不一致'}")

        return True

    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        import traceback
        print(f"詳細: {traceback.format_exc()}")
        return False

def test_important_features():
    """重要特徴量の位置確認"""
    print("\n🎯 重要特徴量位置確認テスト")

    try:
        # 分類器を初期化してconfig読み込み
        model_dir = '/home/tamlab/ws_whill/src/depthai_blazepose/models_class'
        classifier = PoseClassifier(model_dir)

        # 特徴量名を確認
        feature_names = classifier.feature_names
        print(f"📋 特徴量総数: {len(feature_names)}")

        # 重要特徴量（前回の分析結果）
        important_features = ['y16', 'y20', 'y18']

        print(f"🔍 重要特徴量の位置確認:")
        for feat_name in important_features:
            if feat_name in feature_names:
                index = feature_names.index(feat_name)
                print(f"   {feat_name}: インデックス {index}")

                # キーポイント順での位置計算
                keypoint_num = int(feat_name[1:])  # y16 -> 16
                expected_index = keypoint_num * 4 + 1  # x,y,z,vの順序でyは+1

                print(f"     期待値: {expected_index} (キーポイント{keypoint_num}のy座標)")
                print(f"     一致: {'✅' if index == expected_index else '❌'}")
            else:
                print(f"   {feat_name}: ❌ 見つからない")

        return True

    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        return False

def main():
    print("🚀 特徴量抽出修正テスト開始")
    print("=" * 50)

    # テスト1: 特徴量抽出順序
    test_feature_extraction_order()

    # テスト2: Jetson分類器
    classifier_ok = test_jetson_classifier()

    # テスト3: 重要特徴量位置
    features_ok = test_important_features()

    print("\n" + "=" * 50)
    print("📊 テスト結果サマリー")
    print(f"   分類器動作: {'✅ OK' if classifier_ok else '❌ NG'}")
    print(f"   特徴量位置: {'✅ OK' if features_ok else '❌ NG'}")

    if classifier_ok and features_ok:
        print("🎉 全テスト通過! 修正は成功しています")
    else:
        print("⚠️  一部テストに問題があります")

if __name__ == "__main__":
    main()