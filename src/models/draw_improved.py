# -*- coding: utf-8 -*-
"""
引き分け予測改善モデル群

アプローチ:
  A: class_weight='balanced'  - 少数クラスに重みを付けて学習
  B: 引き分け専用特徴量追加   - Elo差小さい/両チームの引き分け率/得失点少ない
  C: 閾値チューニング         - P(draw) > threshold なら引き分けと予測
  D: A+B+C 組み合わせ (推奨)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from .base_model import BaseModel
from src.features.feature_builder import get_feature_columns

LABEL_MAP = {"1": 0, "0": 1, "2": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def _encode_y(y: pd.Series) -> np.ndarray:
    return np.array([LABEL_MAP.get(str(v), 0) for v in y])


def _decode_y(arr: np.ndarray) -> np.ndarray:
    return np.array([INV_LABEL_MAP[int(v)] for v in arr])


# ─────────────────────────────────────────────────────────────
# A: class_weight='balanced'
# ─────────────────────────────────────────────────────────────
class BalancedRFModel(BaseModel):
    """
    class_weight='balanced' で少数クラス(引き分け)を重視
    各クラスの重み = 全サンプル数 / (クラス数 × そのクラスのサンプル数)
    """
    name = "rf_balanced"

    def __init__(self):
        self._feature_cols = get_feature_columns(include_odds=False)
        self._clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",   # ← ここだけ変更
            random_state=42,
            n_jobs=-1,
        )

    def _get_X(self, X):
        cols = [c for c in self._feature_cols if c in X.columns]
        return X[cols].fillna(0)

    def fit(self, X, y):
        self._clf.fit(self._get_X(X), _encode_y(y))

    def predict(self, X):
        return _decode_y(self._clf.predict(self._get_X(X)))

    def predict_proba(self, X):
        return self._clf.predict_proba(self._get_X(X))


# ─────────────────────────────────────────────────────────────
# B: 引き分け専用特徴量
# ─────────────────────────────────────────────────────────────
DRAW_FEATURES = [
    "elo_diff_abs",          # Elo差の絶対値 (小→拮抗→引き分け↑)
    "home_draw_rate",        # ホームチームの直近引き分け率
    "away_draw_rate",        # アウェイチームの直近引き分け率
    "avg_draw_rate",         # 両チーム平均引き分け率
    "home_goals_for_avg",    # ホーム得点平均 (低→引き分け↑)
    "away_goals_for_avg",    # アウェイ得点平均
    "combined_goals_avg",    # 両チーム合計得点平均 (低→引き分け↑)
    "goal_diff_abs",         # ゴール差絶対値の平均 (小→引き分け↑)
]


def add_draw_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    既存特徴量DataFrameに引き分け専用特徴量を追加して返す
    """
    df = feat_df.copy()

    # Elo差絶対値
    if "elo_diff" in df.columns:
        df["elo_diff_abs"] = df["elo_diff"].abs()

    # 引き分け率 (home_form_draw_rate, away_form_draw_rate を流用)
    if "home_form_draw_rate" in df.columns:
        df["home_draw_rate"] = df["home_form_draw_rate"]
    if "away_form_draw_rate" in df.columns:
        df["away_draw_rate"] = df["away_form_draw_rate"]
    if "home_draw_rate" in df.columns and "away_draw_rate" in df.columns:
        df["avg_draw_rate"] = (df["home_draw_rate"] + df["away_draw_rate"]) / 2

    # 合計得点平均・ゴール差
    if "home_form_goals_for_avg" in df.columns and "away_form_goals_for_avg" in df.columns:
        df["combined_goals_avg"] = df["home_form_goals_for_avg"] + df["away_form_goals_for_avg"]
    if "home_form_goal_diff_avg" in df.columns and "away_form_goal_diff_avg" in df.columns:
        df["goal_diff_abs"] = (
            (df["home_form_goal_diff_avg"] - df["away_form_goal_diff_avg"]).abs()
        )

    return df


def get_draw_feature_columns() -> list[str]:
    base = get_feature_columns(include_odds=False)
    extras = [f for f in DRAW_FEATURES if f not in base]
    return base + extras


class DrawFeatureRFModel(BaseModel):
    """
    引き分け専用特徴量追加 + class_weight='balanced'
    """
    name = "rf_draw_features"

    def __init__(self):
        self._feature_cols = get_draw_feature_columns()
        self._clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    def _get_X(self, X):
        cols = [c for c in self._feature_cols if c in X.columns]
        return X[cols].fillna(0)

    def fit(self, X, y):
        self._clf.fit(self._get_X(X), _encode_y(y))

    def predict(self, X):
        return _decode_y(self._clf.predict(self._get_X(X)))

    def predict_proba(self, X):
        return self._clf.predict_proba(self._get_X(X))


# ─────────────────────────────────────────────────────────────
# C: 閾値チューニング
# ─────────────────────────────────────────────────────────────
class ThresholdTunedModel(BaseModel):
    """
    P(draw) > draw_threshold なら引き分けと予測するラッパー
    draw_threshold はバリデーションデータで自動チューニング
    """
    name = "rf_threshold_tuned"

    def __init__(self, draw_threshold: float = None, auto_tune: bool = True):
        self._feature_cols = get_draw_feature_columns()
        self._clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.draw_threshold = draw_threshold
        self.auto_tune = auto_tune

    def _get_X(self, X):
        cols = [c for c in self._feature_cols if c in X.columns]
        return X[cols].fillna(0)

    def fit(self, X, y):
        Xf = self._get_X(X)
        yf = _encode_y(y)

        if self.auto_tune:
            # 学習データの後半20%でバリデーション → 最適閾値を探す
            n = len(Xf)
            val_start = int(n * 0.8)
            self._clf.fit(Xf[:val_start], yf[:val_start])

            proba_val = self._clf.predict_proba(Xf[val_start:])
            y_val = yf[val_start:]

            best_thresh, best_f1 = 0.20, -1.0
            for thresh in np.arange(0.15, 0.45, 0.01):
                preds = self._apply_threshold(proba_val, thresh)
                # 引き分けF1スコアを最大化しつつ全体精度を維持
                draw_tp = ((preds == 1) & (y_val == 1)).sum()
                draw_fp = ((preds == 1) & (y_val != 1)).sum()
                draw_fn = ((preds != 1) & (y_val == 1)).sum()
                prec = draw_tp / (draw_tp + draw_fp + 1e-9)
                rec = draw_tp / (draw_tp + draw_fn + 1e-9)
                f1 = 2 * prec * rec / (prec + rec + 1e-9)
                overall_acc = (preds == y_val).mean()
                # 全体精度が元の95%以上を維持する条件で最大F1
                if overall_acc >= 0.38 and f1 > best_f1:
                    best_f1, best_thresh = f1, thresh

            self.draw_threshold = best_thresh
            # 全データで再学習
            self._clf.fit(Xf, yf)
        else:
            self._clf.fit(Xf, yf)

        if self.draw_threshold is None:
            self.draw_threshold = 0.28

    def _apply_threshold(self, proba: np.ndarray, threshold: float) -> np.ndarray:
        preds = proba.argmax(axis=1)
        # P(draw) が threshold 超えたら引き分けに上書き
        preds[proba[:, 1] >= threshold] = 1
        return preds

    def predict(self, X):
        proba = self._clf.predict_proba(self._get_X(X))
        raw = self._apply_threshold(proba, self.draw_threshold)
        return _decode_y(raw)

    def predict_proba(self, X):
        return self._clf.predict_proba(self._get_X(X))
