import numpy as np
import joblib
import json
from pathlib import Path

class PoseClassifier:
    """
    Jetson推論用の姿勢分類クラス
    歩行 vs 歩きスマホの分類を行う
    """
    
    def __init__(self, model_dir=None):
        """
        モデルを初期化
        
        Args:
            model_dir (str): モデルファイルのディレクトリパス
        """
        if model_dir is None:
            model_dir = Path(__file__).parent
        else:
            model_dir = Path(model_dir)
        
        # モデルファイルの読み込み
        self.model = joblib.load(model_dir / 'pose_classifier_rf.pkl')
        self.scaler = joblib.load(model_dir / 'feature_scaler.pkl')
        self.label_encoder = joblib.load(model_dir / 'label_encoder.pkl')
        
        # 設定ファイルの読み込み
        with open(model_dir / 'model_config.json', 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.class_names = self.config['class_names']
        self.feature_names = self.config['feature_names']
        
        print(f"PoseClassifier initialized")
        print(f"Classes: {self.class_names}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Model accuracy: {self.config['performance']['test_accuracy']:.3f}")
    
    def preprocess_features(self, keypoints_3d):
        """
        3Dキーポイントから特徴量を抽出・前処理
        
        Args:
            keypoints_3d: array-like, shape=(33, 4) or (99+33,)
                MediaPipe BlazePoseの3Dキーポイント [x, y, z, visibility]
                または平坦化された特徴量ベクトル
        
        Returns:
            features: array, shape=(1, n_features)
                前処理済み特徴量
        """
        keypoints_3d = np.array(keypoints_3d)
        
        # 既に平坦化されている場合
        if keypoints_3d.shape == (len(self.feature_names),):
            features = keypoints_3d.reshape(1, -1)
        
        # (33, 4)形式の場合
        elif keypoints_3d.shape == (33, 4):
            # 特徴量ベクトルを構築（キーポイント順: x1,y1,z1,v1, x2,y2,z2,v2, ...）
            features_list = []
            for i in range(33):
                features_list.extend([keypoints_3d[i, 0], keypoints_3d[i, 1],
                                     keypoints_3d[i, 2], keypoints_3d[i, 3]])
            features = np.array(features_list).reshape(1, -1)
        
        else:
            raise ValueError(f"Unsupported keypoints shape: {keypoints_3d.shape}")
        
        # 標準化
        features_scaled = self.scaler.transform(features)
        
        return features_scaled
    
    def predict(self, keypoints_3d, return_proba=False):
        """
        姿勢を分類
        
        Args:
            keypoints_3d: array-like
                3Dキーポイントデータ
            return_proba: bool
                確率も返すかどうか
        
        Returns:
            prediction: str
                予測クラス名
            probability: float (return_proba=Trueの場合)
                予測確率
        """
        # 特徴量前処理
        features = self.preprocess_features(keypoints_3d)
        
        # 予測
        pred_encoded = self.model.predict(features)[0]
        pred_class = self.label_encoder.inverse_transform([pred_encoded])[0]
        
        if return_proba:
            pred_proba = self.model.predict_proba(features)[0]
            class_proba = dict(zip(self.class_names, pred_proba))
            return pred_class, class_proba
        
        return pred_class
    
    def predict_batch(self, keypoints_batch, return_proba=False):
        """
        バッチ予測
        
        Args:
            keypoints_batch: array-like, shape=(n_samples, ...)
                複数の3Dキーポイントデータ
            return_proba: bool
                確率も返すかどうか
        
        Returns:
            predictions: list
                予測クラス名のリスト
            probabilities: list (return_proba=Trueの場合)
                予測確率のリスト
        """
        predictions = []
        probabilities = [] if return_proba else None
        
        for keypoints in keypoints_batch:
            if return_proba:
                pred, proba = self.predict(keypoints, return_proba=True)
                predictions.append(pred)
                probabilities.append(proba)
            else:
                pred = self.predict(keypoints, return_proba=False)
                predictions.append(pred)
        
        if return_proba:
            return predictions, probabilities
        return predictions
    
    def get_model_info(self):
        """
        モデル情報を返す
        
        Returns:
            info: dict
                モデルの詳細情報
        """
        return {
            'model_type': self.config['model_type'],
            'version': self.config['model_version'],
            'classes': self.class_names,
            'feature_count': len(self.feature_names),
            'accuracy': self.config['performance']['test_accuracy'],
            'auc_score': self.config['performance']['auc_score']
        }


if __name__ == "__main__":
    # 使用例
    classifier = PoseClassifier()
    
    # ダミーデータでテスト
    dummy_keypoints = np.random.randn(33, 4)
    prediction = classifier.predict(dummy_keypoints, return_proba=True)
    print(f"Prediction: {prediction}")
    
    print("Model info:", classifier.get_model_info())
