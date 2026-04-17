"""
特徴量エンジニアリング

試合ごとに以下の特徴量を生成:
1. チームエンコーディング (One-Hot / Label)
2. 過去N試合の勝率・得点・失点平均 (直近フォーム)
3. ホーム/アウェイ別の成績
4. 対戦相手との直接対決成績 (Head-to-Head)
5. Eloレーティング (動的チーム強度)
6. ゴール差平均
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional


class FeatureBuilder:
    """
    試合データから特徴量を構築する。
    時系列リークを防ぐため、各試合の特徴量は「その試合以前」のデータのみで計算する。
    """

    def __init__(self, form_window: int = 5, elo_k: float = 32.0, elo_init: float = 1500.0):
        self.form_window = form_window  # 直近何試合を参照するか
        self.elo_k = elo_k
        self.elo_init = elo_init

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        入力: 試合結果DataFrame
            必須列: date, home_team, away_team, home_score, away_score, result
        出力: 特徴量付きDataFrame (時系列順)
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

        # シーズン進捗計算のために年ごとの日付範囲を事前収集
        year_date_ranges: dict[int, tuple] = {}
        for _, row in df.iterrows():
            d = row["date"]
            if pd.isna(d):
                continue
            y = d.year
            if y not in year_date_ranges:
                year_date_ranges[y] = (d, d)
            else:
                mn, mx = year_date_ranges[y]
                year_date_ranges[y] = (min(mn, d), max(mx, d))

        # 特徴量格納用
        features = []
        elo_ratings = defaultdict(lambda: self.elo_init)
        team_history: dict[str, list[dict]] = defaultdict(list)
        team_last_date: dict[str, pd.Timestamp] = {}

        for idx, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]

            home_hist = team_history[home]
            away_hist = team_history[away]

            feat = {
                "idx": idx,
                "date": row["date"],
                "home_team": home,
                "away_team": away,
                "result": row["result"],
                "home_score": row.get("home_score", np.nan),
                "away_score": row.get("away_score", np.nan),
            }

            # --- Eloレーティング ---
            feat["home_elo"] = elo_ratings[home]
            feat["away_elo"] = elo_ratings[away]
            feat["elo_diff"] = elo_ratings[home] - elo_ratings[away]

            # --- 直近フォーム (全試合) ---
            for prefix, hist in [("home", home_hist), ("away", away_hist)]:
                recent = hist[-self.form_window:]
                feat[f"{prefix}_form_win_rate"] = self._win_rate(recent)
                feat[f"{prefix}_form_draw_rate"] = self._draw_rate(recent)
                feat[f"{prefix}_form_goals_for_avg"] = self._goals_for_avg(recent)
                feat[f"{prefix}_form_goals_against_avg"] = self._goals_against_avg(recent)
                feat[f"{prefix}_form_goal_diff_avg"] = self._goal_diff_avg(recent)
                feat[f"{prefix}_games_played"] = len(hist)

            # --- ホーム限定・アウェイ限定フォーム ---
            home_home_hist = [h for h in home_hist if h["is_home"]]
            away_away_hist = [h for h in away_hist if not h["is_home"]]
            for prefix, hist in [("home_home", home_home_hist), ("away_away", away_away_hist)]:
                recent = hist[-self.form_window:]
                feat[f"{prefix}_win_rate"] = self._win_rate(recent)
                feat[f"{prefix}_goals_for_avg"] = self._goals_for_avg(recent)
                feat[f"{prefix}_goals_against_avg"] = self._goals_against_avg(recent)

            # --- Head-to-Head ---
            h2h = [h for h in home_hist if h["opponent"] == away]
            feat["h2h_home_win_rate"] = self._win_rate(h2h[-5:]) if h2h else 0.33
            feat["h2h_count"] = len(h2h)

            # --- 休養日数 ---
            current_date = row["date"]
            feat["rest_days_home"] = (current_date - team_last_date[home]).days if home in team_last_date and not pd.isna(current_date) else 30
            feat["rest_days_away"] = (current_date - team_last_date[away]).days if away in team_last_date and not pd.isna(current_date) else 30

            # --- Elo差の絶対値 (引き分け傾向指標) ---
            feat["elo_diff_abs"] = abs(elo_ratings[home] - elo_ratings[away])

            # --- 両チームの引き分け率平均 ---
            feat["both_draw_rate"] = (feat["home_form_draw_rate"] + feat["away_form_draw_rate"]) / 2

            # --- 直近3試合の勝率・得点平均 ---
            for prefix, hist in [("home", home_hist), ("away", away_hist)]:
                recent3 = hist[-3:]
                feat[f"{prefix}_form_win_rate_3"] = self._win_rate(recent3)
                feat[f"{prefix}_form_goals_for_avg_3"] = self._goals_for_avg(recent3)

            # --- シーズン進捗 ---
            season_progress = 0.5  # デフォルト
            if not pd.isna(current_date):
                y = current_date.year
                if y in year_date_ranges:
                    yr_min, yr_max = year_date_ranges[y]
                    total_days = (yr_max - yr_min).days
                    if total_days > 0:
                        season_progress = (current_date - yr_min).days / total_days
                    else:
                        season_progress = 0.0
            feat["season_progress"] = season_progress

            # --- ベッティングオッズ由来の特徴量 ---
            oh = row.get("odds_home_avg")
            od = row.get("odds_draw_avg")
            oa = row.get("odds_away_avg")
            feat["odds_home_avg"] = oh
            feat["odds_draw_avg"] = od
            feat["odds_away_avg"] = oa
            # インプライド確率 (マージン除去なし)
            if oh and od and oa:
                total = 1/oh + 1/od + 1/oa
                feat["implied_prob_home"] = (1/oh) / total
                feat["implied_prob_draw"] = (1/od) / total
                feat["implied_prob_away"] = (1/oa) / total
            else:
                feat["implied_prob_home"] = np.nan
                feat["implied_prob_draw"] = np.nan
                feat["implied_prob_away"] = np.nan

            features.append(feat)

            # --- 試合後に履歴・Eloを更新 ---
            result = str(row["result"])
            home_score = int(row.get("home_score", 0) or 0)
            away_score = int(row.get("away_score", 0) or 0)
            goal_diff = home_score - away_score

            home_won = 1 if result == "1" else 0
            away_won = 1 if result == "2" else 0
            drew = 1 if result == "0" else 0

            team_history[home].append({
                "is_home": True,
                "opponent": away,
                "won": home_won,
                "drew": drew,
                "goals_for": home_score,
                "goals_against": away_score,
                "goal_diff": goal_diff,
            })
            team_history[away].append({
                "is_home": False,
                "opponent": home,
                "won": away_won,
                "drew": drew,
                "goals_for": away_score,
                "goals_against": home_score,
                "goal_diff": -goal_diff,
            })

            # Elo更新
            home_elo, away_elo = self._update_elo(
                elo_ratings[home], elo_ratings[away], result
            )
            elo_ratings[home] = home_elo
            elo_ratings[away] = away_elo

            # 最終試合日を更新
            if not pd.isna(row["date"]):
                team_last_date[home] = row["date"]
                team_last_date[away] = row["date"]

        return pd.DataFrame(features)

    # --- Elo計算 ---
    def _update_elo(self, home_elo: float, away_elo: float, result: str):
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        expected_away = 1 - expected_home

        if result == "1":
            actual_home, actual_away = 1.0, 0.0
        elif result == "0":
            actual_home, actual_away = 0.5, 0.5
        else:
            actual_home, actual_away = 0.0, 1.0

        new_home = home_elo + self.elo_k * (actual_home - expected_home)
        new_away = away_elo + self.elo_k * (actual_away - expected_away)
        return new_home, new_away

    # --- 統計ヘルパー ---
    def _win_rate(self, hist: list) -> float:
        if not hist:
            return 0.33
        return sum(h["won"] for h in hist) / len(hist)

    def _draw_rate(self, hist: list) -> float:
        if not hist:
            return 0.33
        return sum(h["drew"] for h in hist) / len(hist)

    def _goals_for_avg(self, hist: list) -> float:
        if not hist:
            return 1.0
        return sum(h["goals_for"] for h in hist) / len(hist)

    def _goals_against_avg(self, hist: list) -> float:
        if not hist:
            return 1.0
        return sum(h["goals_against"] for h in hist) / len(hist)

    def _goal_diff_avg(self, hist: list) -> float:
        if not hist:
            return 0.0
        return sum(h["goal_diff"] for h in hist) / len(hist)


def get_feature_columns(include_odds: bool = False) -> list[str]:
    """モデル学習に使う特徴量列名を返す"""
    base = [
        "home_elo", "away_elo", "elo_diff",
        "home_form_win_rate", "home_form_draw_rate",
        "home_form_goals_for_avg", "home_form_goals_against_avg", "home_form_goal_diff_avg",
        "away_form_win_rate", "away_form_draw_rate",
        "away_form_goals_for_avg", "away_form_goals_against_avg", "away_form_goal_diff_avg",
        "home_home_win_rate", "home_home_goals_for_avg", "home_home_goals_against_avg",
        "away_away_win_rate", "away_away_goals_for_avg", "away_away_goals_against_avg",
        "h2h_home_win_rate", "h2h_count",
        "home_games_played", "away_games_played",
        "rest_days_home", "rest_days_away",
        "elo_diff_abs", "both_draw_rate",
        "home_form_win_rate_3", "away_form_win_rate_3",
        "home_form_goals_for_avg_3", "away_form_goals_for_avg_3",
        "season_progress",
    ]
    if include_odds:
        base += ["odds_home_avg", "odds_draw_avg", "odds_away_avg",
                 "implied_prob_home", "implied_prob_draw", "implied_prob_away"]
    return base
