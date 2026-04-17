# -*- coding: utf-8 -*-
"""
確率キャリブレーションラッパー
================================
sklearn の CalibratedClassifierCV を使い、
モデルの出力確率をより正確に calibrate する。
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from .base_model import BaseModel

LABEL_MAP = {"1": 0, "0": 1, "2": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class _SklearnCompatWrapper(BaseEstimator, ClassifierMixin):
    """BaseModel を sklearn 互換にするアダプター"""
    def __init__(self, base_model):
        self.base_model = base_model

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.base_model.fit(X, y)
        self.classes_ = np.array([0, 1, 2])
        return self

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        return self.base_model.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class CalibratedModel(BaseModel):
    """
    既存モデルの確率出力を等頻度ビニングで calibrate するラッパー。

    Parameters
    ----------
    base_model : BaseModel
    method : str
        "isotonic" or "sigmoid"
    cv : int
        calibration の CV 分割数
    """
    name = "calibrated"

    def __init__(self, base_model, method="isotonic", cv=3):
        self.base_model = base_model
        self.method = method
        self.cv = cv
        self._cal = None
        self.name = f"calibrated_{base_model.name}"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        y_enc = np.array([LABEL_MAP.get(str(v), 0) for v in y])
        wrapper = _SklearnCompatWrapper(self.base_model)
        self._cal = CalibratedClassifierCV(
            wrapper, method=self.method, cv=self.cv
        )
        self._cal.fit(X, y_enc)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.array([INV_LABEL_MAP[int(np.argmax(p))] for p in proba])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._cal.predict_proba(X)
