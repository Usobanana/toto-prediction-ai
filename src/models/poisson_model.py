# -*- coding: utf-8 -*-
"""
ポアソン分布モデル (Dixon-Coles補正付き)
=========================================

各チームの攻撃力・守備力パラメータを推定し、
ポアソン分布でスコア確率を計算してホーム勝/引き分け/アウェイ勝を予測する。

手法:
  λ_home = μ_home × attack[home] × defense[away]
  λ_away = μ_away × attack[away] × defense[home]

  P(home=x, away=y) = Poisson(x; λ_home) × Poisson(y; λ_away) × τ(x,y)
  τ(x,y) = Dixon-Coles補正 (0-0, 1-0, 0-1, 1-1 の過少/過多推定を補正)

  P(1) = Σ_{x>y} P(home=x, away=y)
  P(0) = Σ_{x=y} P(home=x, away=y)
  P(2) = Σ_{x<y} P(home=x, away=y)

fit() に渡す X には home_team, away_team, home_score, away_score が必要。
predict() に渡す X には home_team, away_team のみ必要。
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from collections import defaultdict
from .base_model import BaseModel


class PoissonModel(BaseModel):
    """
    Dixon-Coles補正付きポアソンモデル

    Parameters
    ----------
    max_goals : int
        スコア計算の上限ゴール数 (デフォルト10)
    dc_rho : float
        Dixon-Coles補正パラメータ (デフォルト -0.13)
        0 にすると補正なし
    form_window : int
        直近何試合を使うか (None = 全試合)
    time_decay : float
        指数時間減衰の半減期 (日数)。None = 減衰なし
    """

    name = "poisson"

    def __init__(
        self,
        max_goals: int = 10,
        dc_rho: float = -0.13,
        form_window: int = None,
        time_decay: float = None,
    ):
        self.max_goals = max_goals
        self.dc_rho = dc_rho
        self.form_window = form_window
        self.time_decay = time_decay

        # 学習済みパラメータ
        self._attack: dict[str, float] = {}
        self._defense: dict[str, float] = {}
        self._mu_home: float = 1.5
        self._mu_away: float = 1.1
        self._default_attack: float = 1.0
        self._default_defense: float = 1.0

    # ──────────────────────────────────────────────────────────────────
    # Fit
    # ──────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Parameters
        ----------
        X : DataFrame
            列: home_team, away_team, home_score, away_score
            (date 列があれば時間減衰に使用)
        y : Series  (使用しないが BaseModel の interface に合わせる)
        """
        df = X.copy()
        df["y"] = y.values

        # 得点データが必要
        if "home_score" not in df.columns or "away_score" not in df.columns:
            raise ValueError("Poisson モデルには home_score, away_score 列が必要です")

        df = df.dropna(subset=["home_score", "away_score"])

        # 時間減衰の重みを計算
        weights = self._calc_weights(df)

        # リーグ平均ゴール数（加重平均）
        w_total = weights.sum()
        self._mu_home = float(
            (df["home_score"].astype(float) * weights).sum() / w_total
        )
        self._mu_away = float(
            (df["away_score"].astype(float) * weights).sum() / w_total
        )

        # チームごとの攻撃力・守備力を反復推定（2回で収束に近い）
        self._attack = defaultdict(lambda: 1.0)
        self._defense = defaultdict(lambda: 1.0)

        for _ in range(20):  # 反復回数
            self._update_params(df, weights)

        # デフォルト値（未知チーム用）
        vals = list(self._attack.values())
        self._default_attack = float(np.mean(vals)) if vals else 1.0
        vals = list(self._defense.values())
        self._default_defense = float(np.mean(vals)) if vals else 1.0

    def _calc_weights(self, df: pd.DataFrame) -> np.ndarray:
        """試合ごとの重みを計算（時間減衰 or 均等）"""
        n = len(df)
        if self.time_decay is None or "date" not in df.columns:
            return np.ones(n)

        dates = pd.to_datetime(df["date"], errors="coerce")
        max_date = dates.max()
        days = (max_date - dates).dt.days.fillna(0).values
        weights = np.exp(-np.log(2) / self.time_decay * days)
        return weights

    def _update_params(self, df: pd.DataFrame, weights: np.ndarray):
        """Expectation-Maximization的にパラメータ更新（1ステップ）"""
        teams = pd.concat([df["home_team"], df["away_team"]]).unique()

        new_attack = {}
        new_defense = {}

        for team in teams:
            # ホーム試合での攻撃力
            home_mask = df["home_team"] == team
            away_mask = df["away_team"] == team

            # 攻撃: 自チームの得点 / (μ × 相手守備力)
            h_scored = df.loc[home_mask, "home_score"].astype(float).values
            h_opp_def = np.array([
                self._defense.get(t, 1.0)
                for t in df.loc[home_mask, "away_team"]
            ])
            h_w = weights[home_mask.values]

            a_scored = df.loc[away_mask, "away_score"].astype(float).values
            a_opp_def = np.array([
                self._defense.get(t, 1.0)
                for t in df.loc[away_mask, "home_team"]
            ])
            a_w = weights[away_mask.values]

            num_att = np.concatenate([h_scored * h_w, a_scored * a_w]).sum()
            den_att_h = (self._mu_home * h_opp_def * h_w).sum() if len(h_w) else 0
            den_att_a = (self._mu_away * a_opp_def * a_w).sum() if len(a_w) else 0
            den_att = den_att_h + den_att_a

            new_attack[team] = (num_att / den_att) if den_att > 0 else 1.0

            # 守備: 相手に与えた得点 / (μ × 相手攻撃力)
            h_conceded = df.loc[home_mask, "away_score"].astype(float).values
            h_opp_att = np.array([
                self._attack.get(t, 1.0)
                for t in df.loc[home_mask, "away_team"]
            ])
            a_conceded = df.loc[away_mask, "home_score"].astype(float).values
            a_opp_att = np.array([
                self._attack.get(t, 1.0)
                for t in df.loc[away_mask, "home_team"]
            ])

            num_def = np.concatenate([h_conceded * h_w, a_conceded * a_w]).sum()
            den_def_h = (self._mu_away * h_opp_att * h_w).sum() if len(h_w) else 0
            den_def_a = (self._mu_home * a_opp_att * a_w).sum() if len(a_w) else 0
            den_def = den_def_h + den_def_a

            new_defense[team] = (num_def / den_def) if den_def > 0 else 1.0

        # 正規化（平均が1になるように）
        att_vals = np.array(list(new_attack.values()))
        def_vals = np.array(list(new_defense.values()))
        att_mean = att_vals.mean() if len(att_vals) else 1.0
        def_mean = def_vals.mean() if len(def_vals) else 1.0

        for team in teams:
            self._attack[team] = new_attack[team] / att_mean
            self._defense[team] = new_defense[team] / def_mean

    # ──────────────────────────────────────────────────────────────────
    # Predict
    # ──────────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        # 最大確率クラス → "1"(index0), "0"(index1), "2"(index2)
        labels = ["1", "0", "2"]
        return np.array([labels[np.argmax(p)] for p in proba])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        results = []
        for _, row in X.iterrows():
            home = row.get("home_team", "")
            away = row.get("away_team", "")

            att_h = self._attack.get(home, self._default_attack)
            def_h = self._defense.get(home, self._default_defense)
            att_a = self._attack.get(away, self._default_attack)
            def_a = self._defense.get(away, self._default_defense)

            lam_h = self._mu_home * att_h * def_a
            lam_a = self._mu_away * att_a * def_h

            p1, p0, p2 = self._score_probs(lam_h, lam_a)
            results.append([p1, p0, p2])

        return np.array(results)

    def _score_probs(self, lam_h: float, lam_a: float) -> tuple[float, float, float]:
        """ポアソン分布でP(1), P(0), P(2)を計算"""
        mg = self.max_goals
        rho = self.dc_rho

        # ゴール数ごとのポアソン確率（0〜max_goals）
        ph = np.array([poisson.pmf(k, lam_h) for k in range(mg + 1)])
        pa = np.array([poisson.pmf(k, lam_a) for k in range(mg + 1)])

        # スコア行列 M[x][y] = P(home=x) × P(away=y)
        M = np.outer(ph, pa)

        # Dixon-Coles補正（低スコア域のみ）
        if rho != 0:
            tau = self._dc_tau(lam_h, lam_a, rho)
            M[0, 0] *= tau[0]
            M[1, 0] *= tau[1]
            M[0, 1] *= tau[2]
            M[1, 1] *= tau[3]

        # 正規化
        total = M.sum()
        if total > 0:
            M /= total

        p_home = float(np.tril(M, -1).sum())   # x > y (home wins)
        p_draw = float(np.trace(M))              # x = y
        p_away = float(np.triu(M, 1).sum())      # x < y (away wins)

        return p_home, p_draw, p_away

    @staticmethod
    def _dc_tau(lam_h: float, lam_a: float, rho: float) -> list[float]:
        """Dixon-Coles補正係数 τ(0,0), τ(1,0), τ(0,1), τ(1,1)"""
        return [
            1 - lam_h * lam_a * rho,
            1 + lam_a * rho,
            1 + lam_h * rho,
            1 - rho,
        ]

    # ──────────────────────────────────────────────────────────────────
    # 便利メソッド
    # ──────────────────────────────────────────────────────────────────

    def team_params(self, team: str) -> dict:
        """チームのパラメータを返す（デバッグ用）"""
        return {
            "attack":  self._attack.get(team, self._default_attack),
            "defense": self._defense.get(team, self._default_defense),
        }

    def score_probability(
        self, home_team: str, away_team: str, max_score: int = 5
    ) -> pd.DataFrame:
        """
        特定カードのスコア確率行列を返す（mini toto / totoGOAL3 活用用）

        Returns
        -------
        DataFrame: 行=ホームゴール数, 列=アウェイゴール数
        """
        att_h = self._attack.get(home_team, self._default_attack)
        def_h = self._defense.get(home_team, self._default_defense)
        att_a = self._attack.get(away_team, self._default_attack)
        def_a = self._defense.get(away_team, self._default_defense)

        lam_h = self._mu_home * att_h * def_a
        lam_a = self._mu_away * att_a * def_h

        mg = max_score
        ph = np.array([poisson.pmf(k, lam_h) for k in range(mg + 1)])
        pa = np.array([poisson.pmf(k, lam_a) for k in range(mg + 1)])
        M = np.outer(ph, pa)

        if self.dc_rho != 0:
            tau = self._dc_tau(lam_h, lam_a, self.dc_rho)
            M[0, 0] *= tau[0]
            M[1, 0] *= tau[1]
            M[0, 1] *= tau[2]
            M[1, 1] *= tau[3]

        M /= M.sum()

        index = [f"ホーム{i}点" for i in range(mg + 1)]
        cols  = [f"アウェイ{i}点" for i in range(mg + 1)]
        return pd.DataFrame(M * 100, index=index, columns=cols).round(2)
