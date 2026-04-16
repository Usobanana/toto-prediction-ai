"""予想モデルの基底クラス"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """全予想モデルの共通インターフェース"""

    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """学習"""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測ラベル (1/0/2 の文字列配列)"""
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """確率予測 [P(1), P(0), P(2)] の配列。未実装モデルは均等確率を返す"""
        n = len(X)
        return np.full((n, 3), 1 / 3)
