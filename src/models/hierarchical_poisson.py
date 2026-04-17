# -*- coding: utf-8 -*-
"""
階層ベイズ ポアソンモデル (Hierarchical Bayesian Poisson)
==========================================================

通常のポアソンモデルとの違い:
  - 試合数が少ないチーム（J2/J3）のパラメータを
    「リーグ全体の平均」に向けて縮小（シュリンケージ）する
  - データが多いチームほど実測値を信頼、少ないチームほど
    リーグ平均を借用 → Eloフォールバックより精度が高い

数学的根拠（Empirical Bayes / 正規正規共役事前分布）:
  事前分布: attack[team] ~ Normal(μ_league, σ_league²)
  尤度:     observed_goals | attack, λ ~ Poisson
  事後推定: attack_posterior = (n × attack_raw + k × μ_league) / (n + k)
    ここで k = prior_strength（デフォルト10試合分の仮想観測）

  n が大きい（J1 強豪）→ attack_raw を信頼
  n が小さい（J3 新参）→ μ_league（リーグ平均）に縮小

実装方式: Empirical Bayes（超パラメータをデータから推定）
  → PyMC 等の MCMC 不要、高速でバックテストに使用可能
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from collections import defaultdict
from typing import Optional
from .base_model import BaseModel


class HierarchicalPoissonModel(BaseModel):
    """
    階層ベイズ縮小推定付きポアソンモデル

    Parameters
    ----------
    prior_strength : float
        縮小の強さ（仮想観測試合数）。大きいほどリーグ平均に近づく。
        デフォルト 10 ≒ 「10試合分のリーグ平均をアンカーとして使う」
    max_goals : int
        スコア計算の上限ゴール数
    dc_rho : float
        Dixon-Coles 補正パラメータ（0 = 補正なし）
    time_decay : float or None
        指数時間減衰の半減期（日数）。None = 減衰なし
    n_iter : int
        EM 反復回数
    """

    name = "hierarchical_poisson"

    def __init__(
        self,
        prior_strength: float = 10.0,
        max_goals: int = 10,
        dc_rho: float = -0.13,
        time_decay: Optional[float] = None,
        n_iter: int = 30,
    ):
        self.prior_strength = prior_strength
        self.max_goals = max_goals
        self.dc_rho = dc_rho
        self.time_decay = time_decay
        self.n_iter = n_iter

        # 学習済みパラメータ
        self._attack: dict[str, float] = {}
        self._defense: dict[str, float] = {}
        self._mu_home: float = 1.5
        self._mu_away: float = 1.1
        self._n_games: dict[str, int] = {}      # チームごとの試合数
        self._sigma_att: float = 0.3             # 攻撃力の事前分散（推定）
        self._sigma_def: float = 0.3             # 守備力の事前分散（推定）
        self._default_attack: float = 1.0
        self._default_defense: float = 1.0

    # ──────────────────────────────────────────────────────────────────
    # Fit
    # ──────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        df = X.copy()
        df["y"] = y.values

        if "home_score" not in df.columns or "away_score" not in df.columns:
            raise ValueError("home_score, away_score 列が必要です")

        df = df.dropna(subset=["home_score", "away_score"])
        weights = self._calc_weights(df)

        # リーグ平均ゴール
        w_sum = weights.sum()
        self._mu_home = float((df["home_score"].astype(float) * weights).sum() / w_sum)
        self._mu_away = float((df["away_score"].astype(float) * weights).sum() / w_sum)

        # チームごとの試合数を記録
        teams = pd.concat([df["home_team"], df["away_team"]]).unique()
        for team in teams:
            n_h = (df["home_team"] == team).sum()
            n_a = (df["away_team"] == team).sum()
            self._n_games[team] = int(n_h + n_a)

        # EM 反復で attack/defense を推定（階層縮小あり）
        self._attack  = {t: 1.0 for t in teams}
        self._defense = {t: 1.0 for t in teams}

        for iteration in range(self.n_iter):
            new_att, new_def = self._em_step(df, weights)

            # ── Hierarchical shrinkage ──────────────────────────────────
            # リーグ平均（=1.0 に正規化済み）に向けて縮小
            # posterior = (n_games × raw + prior_strength × league_mean) / (n_games + prior_strength)
            league_att_mean = np.mean(list(new_att.values()))
            league_def_mean = np.mean(list(new_def.values()))

            for team in teams:
                n = self._n_games.get(team, 0)
                k = self.prior_strength

                # 縮小後の推定値
                self._attack[team]  = (n * new_att.get(team, league_att_mean) + k * league_att_mean) / (n + k)
                self._defense[team] = (n * new_def.get(team, league_def_mean) + k * league_def_mean) / (n + k)

            # 正規化（平均=1.0）
            att_mean = np.mean(list(self._attack.values()))
            def_mean = np.mean(list(self._defense.values()))
            for t in teams:
                self._attack[t]  /= att_mean
                self._defense[t] /= def_mean

        # 超パラメータ推定（事前分散 σ）
        att_vals = np.array(list(self._attack.values()))
        def_vals = np.array(list(self._defense.values()))
        self._sigma_att = float(np.std(att_vals)) + 1e-6
        self._sigma_def = float(np.std(def_vals)) + 1e-6

        # デフォルト値（未知チーム = リーグ平均）
        self._default_attack  = float(np.mean(att_vals))
        self._default_defense = float(np.mean(def_vals))

    def _calc_weights(self, df: pd.DataFrame) -> np.ndarray:
        if self.time_decay is None or "date" not in df.columns:
            return np.ones(len(df))
        dates = pd.to_datetime(df["date"], errors="coerce")
        days = (dates.max() - dates).dt.days.fillna(0).values
        return np.exp(-np.log(2) / self.time_decay * days)

    def _em_step(self, df: pd.DataFrame, weights: np.ndarray) -> tuple[dict, dict]:
        """縮小なしの EM 1ステップ"""
        teams = list(set(df["home_team"].tolist() + df["away_team"].tolist()))
        new_attack  = {}
        new_defense = {}

        for team in teams:
            hm = df["home_team"] == team
            am = df["away_team"] == team

            h_scored   = df.loc[hm, "home_score"].astype(float).values
            h_opp_def  = np.array([self._defense.get(t, 1.0) for t in df.loc[hm, "away_team"]])
            h_w        = weights[hm.values]

            a_scored   = df.loc[am, "away_score"].astype(float).values
            a_opp_def  = np.array([self._defense.get(t, 1.0) for t in df.loc[am, "home_team"]])
            a_w        = weights[am.values]

            num_att = (h_scored * h_w).sum() + (a_scored * a_w).sum()
            den_att = (self._mu_home * h_opp_def * h_w).sum() + (self._mu_away * a_opp_def * a_w).sum()
            new_attack[team] = num_att / den_att if den_att > 0 else 1.0

            h_conceded  = df.loc[hm, "away_score"].astype(float).values
            h_opp_att   = np.array([self._attack.get(t, 1.0) for t in df.loc[hm, "away_team"]])
            a_conceded  = df.loc[am, "home_score"].astype(float).values
            a_opp_att   = np.array([self._attack.get(t, 1.0) for t in df.loc[am, "home_team"]])

            num_def = (h_conceded * h_w).sum() + (a_conceded * a_w).sum()
            den_def = (self._mu_away * h_opp_att * h_w).sum() + (self._mu_home * a_opp_att * a_w).sum()
            new_defense[team] = num_def / den_def if den_def > 0 else 1.0

        return new_attack, new_defense

    # ──────────────────────────────────────────────────────────────────
    # Predict
    # ──────────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        labels = ["1", "0", "2"]
        return np.array([labels[np.argmax(p)] for p in proba])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        results = []
        for _, row in X.iterrows():
            home = row.get("home_team", "")
            away = row.get("away_team", "")

            att_h = self._get_attack(home)
            def_h = self._get_defense(home)
            att_a = self._get_attack(away)
            def_a = self._get_defense(away)

            lam_h = self._mu_home * att_h * def_a
            lam_a = self._mu_away * att_a * def_h

            p1, p0, p2 = self._score_probs(lam_h, lam_a)
            results.append([p1, p0, p2])
        return np.array(results)

    def _get_attack(self, team: str) -> float:
        """未知チームはリーグ平均に強く縮小した値を返す"""
        if team in self._attack:
            return self._attack[team]
        # 未知チームは prior のみ = league_mean = default_attack
        return self._default_attack

    def _get_defense(self, team: str) -> float:
        if team in self._defense:
            return self._defense[team]
        return self._default_defense

    def _score_probs(self, lam_h: float, lam_a: float) -> tuple[float, float, float]:
        mg  = self.max_goals
        rho = self.dc_rho
        ph  = np.array([poisson.pmf(k, lam_h) for k in range(mg + 1)])
        pa  = np.array([poisson.pmf(k, lam_a) for k in range(mg + 1)])
        M   = np.outer(ph, pa)

        if rho != 0:
            tau = [
                1 - lam_h * lam_a * rho,
                1 + lam_a * rho,
                1 + lam_h * rho,
                1 - rho,
            ]
            M[0, 0] *= tau[0]
            M[1, 0] *= tau[1]
            M[0, 1] *= tau[2]
            M[1, 1] *= tau[3]

        total = M.sum()
        if total > 0:
            M /= total

        return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())

    # ──────────────────────────────────────────────────────────────────
    # 診断用
    # ──────────────────────────────────────────────────────────────────

    def team_params(self, team: str, verbose: bool = False) -> dict:
        """チームパラメータと縮小率を返す"""
        n = self._n_games.get(team, 0)
        k = self.prior_strength
        shrinkage = k / (n + k)  # 0 = 縮小なし, 1 = 完全縮小

        params = {
            "team":       team,
            "n_games":    n,
            "attack":     round(self._get_attack(team), 3),
            "defense":    round(self._get_defense(team), 3),
            "shrinkage":  round(shrinkage, 3),
        }
        if verbose:
            print(f"  {team}: attack={params['attack']:.3f}, defense={params['defense']:.3f}, "
                  f"n={n}, shrinkage={shrinkage:.1%}")
        return params

    def rank_teams(self, top_n: int = 20) -> pd.DataFrame:
        """チームランキング（攻撃力 × 守備力から総合強度を計算）"""
        rows = []
        for team, att in self._attack.items():
            def_ = self._defense.get(team, 1.0)
            n = self._n_games.get(team, 0)
            k = self.prior_strength
            shrinkage = k / (n + k)
            # 総合強度: 攻撃力が高く守備力（数値）が低いほど強い
            strength = att / def_
            rows.append({
                "チーム": team, "攻撃力": round(att, 3),
                "守備脆弱性": round(def_, 3), "総合強度": round(strength, 3),
                "試合数": n, "縮小率": round(shrinkage, 3),
            })
        df = pd.DataFrame(rows).sort_values("総合強度", ascending=False)
        return df.head(top_n).reset_index(drop=True)
