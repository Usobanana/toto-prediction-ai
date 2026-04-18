"""
機械学習モデル群

- LogisticRegressionModel  : ロジスティック回帰
- RandomForestModel        : ランダムフォレスト
- XGBoostModel             : XGBoost
- LightGBMModel            : LightGBM
- MLPModel                 : 多層パーセプトロン
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from .base_model import BaseModel
from src.features.feature_builder import get_feature_columns


LABEL_MAP = {"1": 0, "0": 1, "2": 2}   # 内部ラベル
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class _SklearnBase(BaseModel):
    """scikit-learn 互換モデルの共通ラッパー"""

    def __init__(self, clf, include_odds: bool = True):
        self._clf = clf
        self._include_odds = include_odds
        self._feature_cols = get_feature_columns(include_odds=include_odds)

    def _get_X(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self._feature_cols if c in X.columns]
        return X[cols].fillna(0)

    def _encode_y(self, y: pd.Series) -> np.ndarray:
        return np.array([LABEL_MAP.get(str(v), 0) for v in y])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Xf = self._get_X(X)
        yf = self._encode_y(y)
        self._clf.fit(Xf, yf)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xf = self._get_X(X)
        raw = self._clf.predict(Xf)
        return np.array([INV_LABEL_MAP[int(r)] for r in raw])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xf = self._get_X(X)
        return self._clf.predict_proba(Xf)


class LogisticRegressionModel(_SklearnBase):
    name = "logistic_regression"

    def __init__(self):
        super().__init__(
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                multi_class="multinomial",
                solver="lbfgs",
                random_state=42,
            )
        )


class RandomForestModel(_SklearnBase):
    """
    RandomForestClassifier - Optuna 60試行最適化済みパラメータ (59特徴量)
    バックテスト: 48.15% (HierBayes 45.95% 比 +2.20%)
    """
    name = "random_forest"

    def __init__(self, include_odds: bool = True, class_weight=None):
        super().__init__(
            RandomForestClassifier(
                n_estimators=400,
                max_depth=5,
                min_samples_leaf=7,
                min_samples_split=8,
                max_features=0.3,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1,
            ),
            include_odds=include_odds,
        )

    def feature_importances(self, X: pd.DataFrame) -> pd.Series:
        cols = [c for c in self._feature_cols if c in X.columns]
        return pd.Series(
            self._clf.feature_importances_, index=cols
        ).sort_values(ascending=False)


class ExtraTreesModel(_SklearnBase):
    """
    ExtraTreesClassifier - Optuna 60試行最適化済みパラメータ (59特徴量)
    バックテスト: 47.99%
    """
    name = "extra_trees"

    def __init__(self, include_odds: bool = True):
        super().__init__(
            ExtraTreesClassifier(
                n_estimators=600,
                max_depth=5,
                min_samples_leaf=3,
                max_features=0.5,
                random_state=42,
                n_jobs=-1,
            ),
            include_odds=include_odds,
        )

    def feature_importances(self, X: pd.DataFrame) -> pd.Series:
        cols = [c for c in self._feature_cols if c in X.columns]
        return pd.Series(
            self._clf.feature_importances_, index=cols
        ).sort_values(ascending=False)


class RandomForestDrawModel(_SklearnBase):
    """
    引き分け予測強化版 RF。
    predict() は確率最大クラスが全体max_prob < draw_threshold なら引き分けに補正する。
    """
    name = "random_forest_draw"

    def __init__(self, include_odds: bool = True, draw_threshold: float = 0.38):
        super().__init__(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                class_weight={0: 1, 1: 2, 2: 1},   # 内部ラベル 0=Home, 1=Draw, 2=Away
                random_state=42,
                n_jobs=-1,
            ),
            include_odds=include_odds,
        )
        self._draw_threshold = draw_threshold

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        results = []
        for p in proba:
            max_p   = p.max()
            draw_p  = p[1]   # 内部ラベル1 = 引き分け
            if max_p < self._draw_threshold:
                # 確信度低 → 引き分けへフォールバック
                results.append(INV_LABEL_MAP[1])
            else:
                results.append(INV_LABEL_MAP[int(np.argmax(p))])
        return np.array(results)


class XGBoostModel(_SklearnBase):
    name = "xgboost"

    def __init__(self, include_odds: bool = True):
        if not HAS_XGB:
            raise ImportError("xgboost をインストールしてください: pip install xgboost")
        super().__init__(
            XGBClassifier(
                n_estimators=200,        # 少ないデータfoldでの過学習を防ぐ
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=10,     # 小さいサンプルへの過剰適合を防ぐ
                gamma=0.2,
                reg_alpha=0.5,
                reg_lambda=2.0,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            ),
            include_odds=include_odds,
        )


class LightGBMModel(_SklearnBase):
    name = "lightgbm"

    def __init__(self, include_odds: bool = True):
        if not HAS_LGB:
            raise ImportError("lightgbm をインストールしてください: pip install lightgbm")
        super().__init__(
            LGBMClassifier(
                n_estimators=200,        # 少ないデータfoldでの過学習を防ぐ
                num_leaves=15,           # より浅い木
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_samples=30,    # 小さいリーフを防ぐ
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            include_odds=include_odds,
        )


class MLPModel(_SklearnBase):
    name = "mlp"

    def __init__(self, include_odds: bool = True):
        super().__init__(
            MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                alpha=0.01,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            ),
            include_odds=include_odds,
        )
