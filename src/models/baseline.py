"""
ベースラインモデル群

1. MostFrequentModel  : 常に最頻値クラスを予測
2. HomeWinModel       : 常にホーム勝ち (1) を予測
3. TeamWinRateModel   : チームペアごとの過去勝率ベース
"""

import numpy as np
import pandas as pd
from .base_model import BaseModel


class MostFrequentModel(BaseModel):
    """訓練データで最も多いクラスを常に予測"""

    name = "most_frequent"

    def __init__(self):
        self._most_frequent = "1"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._most_frequent = y.mode()[0]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self._most_frequent] * len(X))


class OddsModel(BaseModel):
    """
    ベッティングオッズから予想するモデル (最小オッズ = 最大確率)
    特徴量: odds_home_avg, odds_draw_avg, odds_away_avg
    """

    name = "odds_based"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass  # 学習不要

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in X.iterrows():
            oh = row.get("odds_home_avg")
            od = row.get("odds_draw_avg")
            oa = row.get("odds_away_avg")

            if pd.isna(oh) or pd.isna(od) or pd.isna(oa):
                preds.append("1")  # オッズなし → ホーム勝ちにフォールバック
                continue

            # オッズが最小 = 最も確率が高い
            min_odds = min(oh, od, oa)
            if min_odds == oh:
                preds.append("1")
            elif min_odds == od:
                preds.append("0")
            else:
                preds.append("2")
        return np.array(preds)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = []
        for _, row in X.iterrows():
            oh = row.get("odds_home_avg", None)
            od = row.get("odds_draw_avg", None)
            oa = row.get("odds_away_avg", None)
            if pd.isna(oh) or pd.isna(od) or pd.isna(oa):
                proba.append([1/3, 1/3, 1/3])
                continue
            total = 1/oh + 1/od + 1/oa
            proba.append([1/oh/total, 1/od/total, 1/oa/total])
        return np.array(proba)


class HomeWinModel(BaseModel):
    """常にホーム勝ち (1) を予測するモデル"""

    name = "home_win"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(["1"] * len(X))


class TeamWinRateModel(BaseModel):
    """
    チームペアごとの過去対戦成績から予測。
    X には home_team, away_team 列が必要。
    """

    name = "team_win_rate"

    def __init__(self):
        self._pair_rates: dict = {}  # (home, away) -> {"1": p1, "0": p0, "2": p2}
        self._global_rates: dict = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        df = X.copy()
        df["result"] = y.values

        # グローバル分布
        counts = df["result"].value_counts(normalize=True)
        self._global_rates = {str(k): float(v) for k, v in counts.items()}
        for r in ["1", "0", "2"]:
            self._global_rates.setdefault(r, 0.0)

        # ペアごと
        for (home, away), grp in df.groupby(["home_team", "away_team"]):
            counts_pair = grp["result"].value_counts(normalize=True)
            self._pair_rates[(home, away)] = {
                str(k): float(v) for k, v in counts_pair.items()
            }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in X.iterrows():
            rates = self._pair_rates.get(
                (row["home_team"], row["away_team"]),
                self._global_rates,
            )
            # 確率が高いクラスを選択
            pred = max(rates, key=rates.get)
            preds.append(pred)
        return np.array(preds)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = []
        for _, row in X.iterrows():
            rates = self._pair_rates.get(
                (row["home_team"], row["away_team"]),
                self._global_rates,
            )
            proba.append([
                rates.get("1", 1/3),
                rates.get("0", 1/3),
                rates.get("2", 1/3),
            ])
        return np.array(proba)
