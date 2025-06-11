#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson用姿勢分類の使用例

使用方法:
    python example_usage.py
"""

import numpy as np
from jetson_pose_classifier import PoseClassifier

def main():
    # 分類器の初期化
    print("姿勢分類器を初期化中...")
    classifier = PoseClassifier()
    
    # モデル情報の表示
    info = classifier.get_model_info()
    print("\\nモデル情報:")
    print(f"  タイプ: {info['model_type']}")
    print(f"  バージョン: {info['version']}")
    print(f"  クラス: {info['classes']}")
    print(f"  特徴量数: {info['feature_count']}")
    print(f"  精度: {info['accuracy']:.3f}")
    print(f"  AUC: {info['auc_score']:.3f}")
    
    # ダミーデータでテスト予測
    print("\\nテスト予測を実行中...")
    
    # MediaPipeの33キーポイント形式 (x, y, z, visibility)
    dummy_keypoints = np.random.randn(33, 4)
    
    # 単一予測
    prediction, probabilities = classifier.predict(dummy_keypoints, return_proba=True)
    print("\\n予測結果:")
    print(f"  クラス: {prediction}")
    print("  確率:")
    for class_name, prob in probabilities.items():
        print(f"    {class_name}: {prob:.3f}")
    
    # バッチ予測のテスト
    print("\\nバッチ予測テスト...")
    batch_keypoints = [np.random.randn(33, 4) for _ in range(3)]
    batch_predictions, batch_probabilities = classifier.predict_batch(
        batch_keypoints, return_proba=True
    )
    
    for i, (pred, proba) in enumerate(zip(batch_predictions, batch_probabilities)):
        print(f"  サンプル{i+1}: {pred} (確率: {proba[pred]:.3f})")
    
    print("\\n推論テスト完了！")

if __name__ == "__main__":
    main()
