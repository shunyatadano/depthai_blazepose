# Jetson用姿勢分類モデル

歩行 vs 歩きスマホの分類を行うRandomForestモデルです。

## ファイル構成

- `pose_classifier_rf.pkl`: 学習済みRandomForestモデル
- `feature_scaler.pkl`: 特徴量標準化用スケーラー
- `label_encoder.pkl`: クラスラベルエンコーダー
- `model_config.json`: モデル設定とメタデータ
- `jetson_pose_classifier.py`: Jetson推論用クラス
- `example_usage.py`: 使用例
- `README.md`: このファイル

## モデル性能

- **テスト精度**: 0.951
- **AUC スコア**: 0.993
- **交差検証精度**: 0.988 ± 0.017

## 分類クラス

- `phone`: 歩きスマホ
- `walk`: 普通の歩行

## 使用方法

### 基本的な使用

```python
from jetson_pose_classifier import PoseClassifier
import numpy as np

# 分類器の初期化
classifier = PoseClassifier()

# MediaPipeの33キーポイント (x, y, z, visibility)
keypoints_3d = np.array([[x1, y1, z1, v1], [x2, y2, z2, v2], ...])  # shape: (33, 4)

# 予測
prediction = classifier.predict(keypoints_3d)
print(f"予測クラス: {prediction}")

# 確率付き予測
prediction, probabilities = classifier.predict(keypoints_3d, return_proba=True)
print(f"予測クラス: {prediction}")
print(f"確率: {probabilities}")
```

### バッチ予測

```python
# 複数サンプルの予測
keypoints_batch = [keypoints1, keypoints2, keypoints3]
predictions = classifier.predict_batch(keypoints_batch)
print(f"バッチ予測結果: {predictions}")
```

### MediaPipeとの連携

```python
import mediapipe as mp

# MediaPipe BlazePoseの結果から分類
def classify_pose_from_mediapipe(results):
    if results.pose_world_landmarks:
        # 3Dランドマークを取得
        landmarks_3d = []
        for landmark in results.pose_world_landmarks.landmark:
            landmarks_3d.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        keypoints_array = np.array(landmarks_3d)
        prediction = classifier.predict(keypoints_array)
        return prediction
    return None
```

## 依存関係

- numpy
- scikit-learn
- joblib

## インストール

```bash
pip install numpy scikit-learn joblib
```

## 注意事項

- 入力は33個のMediaPipe BlazePoseキーポイントである必要があります
- キーポイントの順序はMediaPipeの標準に従ってください
- モデルは正規化された3D座標で学習されています

## モデル詳細

- **アルゴリズム**: Random Forest
- **特徴量**: 33キーポイント × 4次元 (x, y, z, visibility) = 132特徴量
- **学習データ**: 合計304サンプル
- **主要特徴量**: 手首、肘、肩のキーポイントが重要

## 更新履歴

- v1.0 (2025-06-11): 初回リリース
