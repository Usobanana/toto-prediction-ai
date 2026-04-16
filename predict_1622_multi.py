# -*- coding: utf-8 -*-
"""
toto 第1622回 マルチ予想 (予算5000円)
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from src.features.feature_builder import FeatureBuilder, get_feature_columns
from src.models.ml_models import RandomForestModel
from src.strategy.multi_optimizer import MatchPrediction, MultiOptimizer

# ── チーム名マッピング ───────────────────────────────────────────────────
TEAM_MAP = {
    "横浜FM": "Yokohama F. Marinos",
    "川崎F":  "Kawasaki Frontale",
    "鹿島":   "Kashima Antlers",
    "浦和":   "Urawa Reds",
    "C大阪":  "Cerezo Osaka",
    "京都":   "Kyoto",
    "G大阪":  "Gamba Osaka",
    "岡山":   "Okayama",
    "名古屋": "Nagoya Grampus",
    "福岡":   "Avispa Fukuoka",
    "大宮":   "Omiya Ardija",
    "磐田":   "Iwata",
    "東京V":  "Verdy",
    "柏":     "Kashiwa Reysol",
    "仙台":   "Vegalta Sendai",
    "新潟":   "Albirex Niigata",
    "甲府":   "Kofu",
    "横浜FC": "Yokohama FC",
    "山形":   "Montedio Yamagata",
    # データなし
    "水戸": None, "千葉": None, "今治": None,
    "藤枝": None, "栃木C": None, "八戸": None, "秋田": None,
}

MATCHES_1622 = [
    ("横浜FM",  "川崎F"),    #  1
    ("C大阪",   "京都"),     #  2
    ("水戸",    "柏"),       #  3
    ("G大阪",   "岡山"),     #  4
    ("名古屋",  "福岡"),     #  5
    ("鹿島",    "浦和"),     #  6
    ("東京V",   "千葉"),     #  7
    ("仙台",    "栃木C"),    #  8
    ("新潟",    "今治"),     #  9
    ("甲府",    "藤枝"),     # 10
    ("秋田",    "横浜FC"),   # 11
    ("山形",    "八戸"),     # 12
    ("大宮",    "磐田"),     # 13
]


def elo_proba(h_elo: float, a_elo: float) -> list[float]:
    """Eloレーティングから確率推定 (ホームアドバンテージ +50)"""
    diff = h_elo - a_elo + 50
    ph = 1 / (1 + 10 ** (-diff / 400))
    pa = 1 / (1 + 10 ** (diff / 400))
    # 引き分け率は実績値から補正 (~27%)
    pd_ = 0.27
    total = ph + pa + pd_
    return [ph / total, pd_ / total, pa / total]


def main():
    # ── データ読み込み・学習 ──────────────────────────────────────────
    df = pd.read_csv("data/raw/jleague_results.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    builder = FeatureBuilder(form_window=5)
    feat_df = builder.build(df)

    feature_cols = [c for c in get_feature_columns(include_odds=False) if c in feat_df.columns]
    rf = RandomForestModel()
    rf.fit(feat_df[feature_cols].fillna(0), feat_df["result"].astype(str))

    # 各チームの最新特徴量 & Elo
    latest = {}
    elo_dict = {}
    for _, row in feat_df.iterrows():
        latest[row["home_team"]] = row
        latest[row["away_team"]] = row
        elo_dict[row["home_team"]] = row["home_elo"]
        elo_dict[row["away_team"]] = row["away_elo"]

    # ── 各試合の確率推定 ──────────────────────────────────────────────
    match_predictions = []
    for i, (home_jp, away_jp) in enumerate(MATCHES_1622, 1):
        home_en = TEAM_MAP.get(home_jp)
        away_en = TEAM_MAP.get(away_jp)

        if home_en and away_en and home_en in latest and away_en in latest:
            feat_row = latest[home_en][feature_cols].fillna(0)
            pred_df = pd.DataFrame([feat_row])
            proba = rf.predict_proba(pred_df)[0].tolist()
            method = "RF"
        else:
            h_elo = elo_dict.get(home_en, 1500) if home_en else 1500
            a_elo = elo_dict.get(away_en, 1500) if away_en else 1500
            proba = elo_proba(h_elo, a_elo)
            method = "Elo"

        match_predictions.append(MatchPrediction(
            no=i,
            home=home_jp,
            away=away_jp,
            proba=proba,
            method=method,
        ))

    # ── マルチ最適化 (複数予算シナリオ) ──────────────────────────────
    budgets = [1000, 3200, 4800]

    print()
    print("=" * 72)
    print(" toto 第1622回 マルチ予想 設計・最適化")
    print("=" * 72)

    # 確率サマリー表示
    print()
    print("[各試合の確率]")
    print(f"{'No':>2}  {'ホーム':<9} {'':>2} {'アウェイ':<9}  {'1位':<8}  {'2位':<8}  {'不確実度':>5}  手法")
    print("-" * 68)
    for m in match_predictions:
        s = m.sorted_outcomes
        top1_lbl, top1_p = s[0]
        top2_lbl, top2_p = s[1]
        # 不確実度 = 1 - top1確率 (高いほど読みにくい)
        uncertainty = 1 - top1_p
        bar = "#" * int(uncertainty * 10)
        print(
            f"{m.no:>2}. {m.home:<9} vs {m.away:<9}  "
            f"{top1_lbl}:{top1_p:.2f}    {top2_lbl}:{top2_p:.2f}    "
            f"{uncertainty:.2f} {bar:<10}  {m.method}"
        )

    # 各予算でのシナリオ表示
    for budget in budgets:
        optimizer = MultiOptimizer(budget_yen=budget, allow_triple=True)
        result = optimizer.optimize(match_predictions)
        print()
        print(result.summary())

    # ── 推奨プラン (5000円) ──────────────────────────────────────────
    print()
    print("=" * 72)
    print(" [推奨] 予算5000円 最適プラン詳細")
    print("=" * 72)
    optimizer_5000 = MultiOptimizer(budget_yen=5000, allow_triple=True)
    best = optimizer_5000.optimize(match_predictions)
    print(best.summary())

    # ダブル/トリプルにした試合の理由を説明
    print()
    print("[複数択にした試合の理由]")
    for m, k in zip(best.matches, best.selections):
        if k > 1:
            s = m.sorted_outcomes
            top2_gain = m.covered_prob(2) - m.covered_prob(1)
            print(
                f"  試合{m.no:>2} {m.home} vs {m.away}: "
                f"{k}択 → カバー率 {m.covered_prob(1):.2f} → {m.covered_prob(k):.2f} "
                f"(+{top2_gain:.2f})"
            )
    print()


if __name__ == "__main__":
    main()
