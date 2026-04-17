# -*- coding: utf-8 -*-
"""
スタッキングアンサンブル
========================
Level-1 モデルの予測確率を特徴量として
Level-2 ロジスティック回帰でメタ学習する。

時系列リークを防ぐため、Level-1 の予測は
TimeSeriesSplit で out-of-fold 予測を使う。
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from .base_model import BaseModel

LABEL_MAP = {"1": 0, "0": 1, "2": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class StackingModel(BaseModel):
    """
    スタッキングアンサンブル

    Parameters
    ----------
    base_models : list of (name, model) tuples
        Level-1 モデルのリスト。各モデルは fit(X, y) と predict_proba(X) を持つ。
    n_splits : int
        OOF 予測に使う時系列分割数
    """
    name = "stacking"

    def __init__(self, base_models, n_splits=5):
        self.base_models = base_models
        self.n_splits = n_splits
        self._meta = LogisticRegression(C=1.0, max_iter=1000, multi_class="multinomial",
                                         solver="lbfgs", random_state=42)
        self._fitted_base = []

    def _encode_y(self, y):
        return np.array([LABEL_MAP.get(str(v), 0) for v in y])

    def _base_feature_names(self):
        names = []
        for name, _ in self.base_models:
            names += [f"{name}_p1", f"{name}_p0", f"{name}_p2"]
        return names

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        y_enc = self._encode_y(y)
        n = len(X)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        # OOF predictions: shape (n, n_models * 3)
        oof_preds = np.zeros((n, len(self.base_models) * 3))

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y.iloc[train_idx]
            for j, (_, model) in enumerate(self.base_models):
                model.fit(X_tr, y_tr)
                proba = model.predict_proba(X_val)
                oof_preds[val_idx, j*3:(j+1)*3] = proba

        # Fit meta-learner on OOF predictions
        # Only use rows where OOF is filled (skip first fold's training rows)
        first_test_idx = list(tscv.split(X))[0][1]
        meta_X = oof_preds[first_test_idx[0]:]
        meta_y = y_enc[first_test_idx[0]:]
        self._meta.fit(meta_X, meta_y)

        # Re-fit all base models on full data
        self._fitted_base = []
        for name, model in self.base_models:
            model.fit(X, y)
            self._fitted_base.append((name, model))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.array([INV_LABEL_MAP[int(np.argmax(p))] for p in proba])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        base_feats = np.hstack([
            model.predict_proba(X)
            for _, model in self._fitted_base
        ])
        return self._meta.predict_proba(base_feats)
