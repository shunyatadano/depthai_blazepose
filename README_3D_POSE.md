# MediaPipe 3D Pose Tracking for Jetson AGX Orin

Jetson AGX Orin (JetPack 5.1.2) と RealSense D455 を使用したMediaPipeによる3D骨格キーポイント検出とレンダリングのサンプルプロジェクトです。

## 特徴

- **リアルタイム3D姿勢推定**: MediaPipeを使用した高精度な人体姿勢検出
- **深度情報統合**: RealSense D455の深度データを活用した正確な3D座標計算
- **3Dビジュアライゼーション**: Open3Dを使用したインタラクティブな3D骨格表示
- **Jetson最適化**: Jetson AGX Orin向けに最適化された設定
- **データ保存**: JSON形式での姿勢データ保存機能

## システム要件

### ハードウェア
- NVIDIA Jetson AGX Orin
- Intel RealSense D455 カメラ
- USB 3.0ポート
- 8GB以上のRAM推奨

### ソフトウェア
- JetPack 5.1.2
- Python 3.8+
- CUDA 11.4+ (JetPackに含まれる)
- OpenGL対応ディスプレイ

## インストール手順

### 1. リポジトリのクローン
```bash
cd /home/tamlab/ws_whill/src/depthai_blazepose
```

### 2. 自動セットアップスクリプトの実行
```bash
chmod +x setup_jetson.sh
./setup_jetson.sh
```

このスクリプトは以下を自動的に行います：
- システムパッケージの更新
- 必要な依存関係のインストール
- Python仮想環境の作成
- MediaPipe、Open3D、RealSense SDKのインストール
- USB デバイスルールの設定

### 3. 手動インストール（必要に応じて）

仮想環境の作成：
```bash
python3 -m venv venv
source venv/bin/activate
```

依存関係のインストール：
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 環境の有効化
```bash
source venv/bin/activate
```

### 2. カメラテスト

#### Webカメラでの基本テスト
```bash
python3 demo_simple.py
```

#### RealSense D455のテスト
```bash
python3 test_realsense.py
```

### 3. 3D姿勢トラッキングの実行
```bash
python3 my_mediapipe_3d.py
```

## 操作方法

### メインアプリケーション（my_mediapipe_3d.py）
- **q**: アプリケーション終了
- **s**: 現在の姿勢データを保存
- **r**: 3Dビューをリセット

### RealSenseテスト（test_realsense.py）
- **q**: テスト終了
- **d**: 深度表示の切り替え
- **c**: フレームキャプチャ
- **i**: 情報表示の切り替え

### 簡単デモ（demo_simple.py）
- **q**: デモ終了
- **i**: 情報表示の切り替え

## ファイル構成

```
.
├── my_mediapipe_3d.py      # メインの3D姿勢トラッキングアプリ
├── demo_simple.py          # WebカメラでのMediaPipeデモ
├── test_realsense.py       # RealSense D455のテストスクリプト
├── setup_jetson.sh         # Jetson用自動セットアップスクリプト
├── requirements.txt        # Python依存関係リスト
└── README_3D_POSE.md      # このファイル
```

## 設定とカスタマイズ

### 解像度・フレームレートの変更
`my_mediapipe_3d.py`の`main()`関数内で設定を変更：

```python
config = {
    'width': 640,        # 解像度（幅）
    'height': 480,       # 解像度（高さ）
    'fps': 30,           # フレームレート
    'enable_depth': True, # 深度情報の使用
    'save_data': False,  # データ自動保存
    'show_2d': True      # 2D表示
}
```

### MediaPipe設定の調整
`RealSense3DPoseTracker.__init__()`内でMediaPipeパラメータを変更：

```python
self.pose = self.mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,              # 0-2 (高いほど精度向上、処理重い)
    enable_segmentation=False,
    min_detection_confidence=0.5,    # 検出信頼度閾値
    min_tracking_confidence=0.5      # トラッキング信頼度閾値
)
```

## トラブルシューティング

### RealSenseカメラが認識されない
1. USB 3.0ポートに接続されているか確認
2. 他のアプリケーションがカメラを使用していないか確認
3. デバイス権限を確認：
   ```bash
   ls -l /dev/video*
   groups $USER  # videoグループに属しているか確認
   ```

### パフォーマンスが低い場合
1. `model_complexity`を下げる（2 → 1 → 0）
2. 解像度を下げる（640x480 → 320x240）
3. フレームレートを下げる（30 → 15）
4. Jetsonのパフォーマンスモードを確認：
   ```bash
   sudo nvpmodel -q  # 現在のモード確認
   sudo nvpmodel -m 0  # 最大パフォーマンスモード
   ```

### 3D表示が正常に動作しない
1. X11フォワーディングの確認（SSH接続時）：
   ```bash
   echo $DISPLAY
   ssh -X user@jetson_ip
   ```
2. OpenGLサポートの確認：
   ```bash
   glxinfo | grep "direct rendering"
   ```

### メモリ不足エラー
1. 不要なプロセスを終了
2. スワップファイルの作成：
   ```bash
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

## データ出力

### 姿勢データの保存
- **形式**: JSON
- **内容**: タイムスタンプと33個のランドマークの3D座標
- **ファイル名**: `pose_data.json` または `pose_[timestamp].json`

### データ構造例
```json
{
  "poses": [
    {
      "timestamp": 1623456789.123,
      "landmarks": [
        {"x": 0.123, "y": 0.456, "z": 0.789},
        ...
      ]
    }
  ]
}
```

## パフォーマンス指標

### 推奨動作環境
- **解像度**: 640x480
- **フレームレート**: 15-30 FPS
- **レイテンシ**: < 100ms
- **CPU使用率**: < 70%
- **メモリ使用量**: < 4GB

### ベンチマーク例（Jetson AGX Orin）
- MediaPipe処理: ~30ms/frame
- 3D座標変換: ~5ms/frame
- 3D描画: ~15ms/frame
- 総処理時間: ~50ms/frame (20 FPS)

## 開発とカスタマイズ

### 新しい機能の追加
1. 関節角度計算
2. 動作認識
3. 複数人検出
4. トラッキング履歴

### API拡張例
```python
# カスタムトラッカークラスの作成
class CustomPoseTracker(RealSense3DPoseTracker):
    def calculate_joint_angles(self, landmarks):
        # 関節角度計算のカスタム実装
        pass
    
    def detect_gesture(self, pose_sequence):
        # ジェスチャー認識のカスタム実装
        pass
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

バグ報告や機能要求は、GitHubのIssueで受け付けています。

## 参考リンク

- [MediaPipe Documentation](https://mediapipe.dev/)
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)
- [Open3D Documentation](http://www.open3d.org/)
- [NVIDIA Jetson Developer Guide](https://developer.nvidia.com/embedded/jetson-agx-orin-developer-kit)
