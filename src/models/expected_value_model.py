# -*- coding: utf-8 -*-
"""
期待値モデル (Expected Value Model)
=====================================
自モデルの確率と toto 投票率の乖離を使って
「市場が過小評価している結果」を特定する。

エッジ(期待値指標) = model_prob / vote_rate
  > 1.2 : モデルが市場より有利とみる (強気シグナル)
  < 0.8 : モデルが市場より不利とみる (弱気シグナル)
  ≈ 1.0 : 市場と概ね一致

活用方法:
1. シングル予想: エッジ最大の結果を選ぶ
2. マルチ予想: エッジ > 閾値 の結果のみ複数択に追加
3. 戦略: 「みんな知らないアウェイ勝ち」を狙う

fit() に渡す X には vote_rate_1, vote_rate_0, vote_rate_2 列が必要。
predict() では model_proba との組み合わせでエッジを計算する。
"""

import numpy as np
import pandas as pd
from .base_model import BaseModel


class ExpectedValueModel(BaseModel):
    """
    ベースモデルの確率を投票率で割ったエッジで予測するモデル。

    Parameters
    ----------
    base_model : BaseModel
        確率推定に使うベースモデル (Poisson, RF など)
    edge_threshold : float
        「強気」と判断するエッジの閾値 (デフォルト 1.0)
    vote_weight : float
        0=投票率を完全無視(base_model そのまま)
        1=エッジを完全に使う (デフォルト 1.0)
    """

    name = "expected_value"

    def __init__(self, base_model: BaseModel, edge_threshold: float = 1.0, vote_weight: float = 1.0):
        self.base_model = base_model
        self.edge_threshold = edge_threshold
        self.vote_weight = vote_weight

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """ベースモデルを学習（投票率は predict 時に使用）"""
        self.base_model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        labels = ["1", "0", "2"]
        return np.array([labels[np.argmax(p)] for p in proba])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        エッジ補正確率を返す。
        vote_rate_* 列がない場合はベースモデルの確率をそのまま返す。
        """
        base_proba = self.base_model.predict_proba(X)

        has_vote = all(c in X.columns for c in ["vote_rate_1", "vote_rate_0", "vote_rate_2"])
        if not has_vote:
            return base_proba

        results = []
        for i, (_, row) in enumerate(X.iterrows()):
            bp = base_proba[i]  # [P(1), P(0), P(2)]

            vr1 = row["vote_rate_1"] / 100.0
            vr0 = row["vote_rate_0"] / 100.0
            vr2 = row["vote_rate_2"] / 100.0

            vote_rates = np.array([vr1, vr0, vr2])
            # ゼロ除算防止
            vote_rates = np.clip(vote_rates, 0.01, 1.0)

            # エッジ = モデル確率 / 投票率
            edge = bp / vote_rates

            # エッジ補正確率: base × edge^weight
            adjusted = bp * (edge ** self.vote_weight)
            total = adjusted.sum()
            if total > 0:
                adjusted /= total

            results.append(adjusted.tolist())

        return np.array(results)

    def compute_edges(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        各試合・各結果のエッジを計算して DataFrame で返す（分析用）

        Returns
        -------
        DataFrame: columns = [match_idx, edge_1, edge_0, edge_2, recommended, strength]
        """
        base_proba = self.base_model.predict_proba(X)
        has_vote = all(c in X.columns for c in ["vote_rate_1", "vote_rate_0", "vote_rate_2"])

        rows = []
        for i, (idx, row) in enumerate(X.iterrows()):
            bp = base_proba[i]
            if has_vote:
                vr = np.array([
                    max(row["vote_rate_1"] / 100.0, 0.01),
                    max(row["vote_rate_0"] / 100.0, 0.01),
                    max(row["vote_rate_2"] / 100.0, 0.01),
                ])
                edge = bp / vr
            else:
                edge = np.ones(3)

            best_idx = np.argmax(edge)
            labels = ["1", "0", "2"]
            rows.append({
                "match_idx": idx,
                "model_prob_1": bp[0], "model_prob_0": bp[1], "model_prob_2": bp[2],
                "vote_rate_1": row.get("vote_rate_1", np.nan),
                "vote_rate_0": row.get("vote_rate_0", np.nan),
                "vote_rate_2": row.get("vote_rate_2", np.nan),
                "edge_1": edge[0], "edge_0": edge[1], "edge_2": edge[2],
                "recommended": labels[best_idx],
                "max_edge": edge[best_idx],
            })

        return pd.DataFrame(rows)


def analyze_vote_alignment(
    model_proba: np.ndarray,
    vote_rates: np.ndarray,
    match_labels: list[str] = None,
) -> pd.DataFrame:
    """
    モデル確率と投票率を比較して乖離を分析する（スタンドアロン関数）

    Parameters
    ----------
    model_proba : shape (n_matches, 3) = [P(1), P(0), P(2)]
    vote_rates  : shape (n_matches, 3) = [rate_1, rate_0, rate_2] (0-100 or 0-1)
    match_labels : 試合ラベルのリスト

    Returns
    -------
    DataFrame: 各試合のエッジ分析
    """
    n = len(model_proba)
    labels_col = match_labels or [f"試合{i+1}" for i in range(n)]

    # 投票率が 0-100 形式なら 0-1 に変換
    vr = np.array(vote_rates, dtype=float)
    if vr.max() > 1.5:
        vr /= 100.0
    vr = np.clip(vr, 0.01, 1.0)

    mp = np.array(model_proba, dtype=float)

    rows = []
    outcome_names = ["ホーム勝(1)", "引き分け(0)", "アウェイ勝(2)"]
    for i in range(n):
        edge = mp[i] / vr[i]
        max_edge_idx = np.argmax(edge)
        rows.append({
            "試合": labels_col[i],
            "モデル1": f"{mp[i,0]:.1%}",
            "投票率1": f"{vr[i,0]:.1%}",
            "エッジ1": f"{edge[0]:.2f}",
            "モデル0": f"{mp[i,1]:.1%}",
            "投票率0": f"{vr[i,1]:.1%}",
            "エッジ0": f"{edge[1]:.2f}",
            "モデル2": f"{mp[i,2]:.1%}",
            "投票率2": f"{vr[i,2]:.1%}",
            "エッジ2": f"{edge[2]:.2f}",
            "最大エッジ": f"{edge[max_edge_idx]:.2f}",
            "推奨": outcome_names[max_edge_idx],
        })
    return pd.DataFrame(rows)
