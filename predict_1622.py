# -*- coding: utf-8 -*-
"""
toto 第1622回 予想スクリプト
==============================
モデル: ExtraTrees+Optuna+オッズ (バックテスト正答率 47.97%)
  - ExtraTreesClassifier (n=600, depth=8, leaf=13) + Optuna最適化
  - オッズ由来特徴量 + 移動距離・疲労・引き分け専用特徴量
  - バックテスト: HierBayes=45.95% → RF+オッズ=47.36% → ExtraTrees=47.97%
"""
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from src.features.feature_builder import FeatureBuilder, get_feature_columns
from src.models.ml_models import ExtraTreesModel

# ── チーム名マッピング (日本語 → football-data.co.uk 英語表記) ──────────
TEAM_MAP = {
    "横浜FM":  "Yokohama F. Marinos",
    "川崎F":   "Kawasaki Frontale",
    "鹿島":    "Kashima Antlers",
    "浦和":    "Urawa Reds",
    "C大阪":   "Cerezo Osaka",
    "京都":    "Kyoto",
    "G大阪":   "Gamba Osaka",
    "岡山":    "Okayama",
    "名古屋":  "Nagoya Grampus",
    "福岡":    "Avispa Fukuoka",
    "大宮":    "Omiya Ardija",
    "磐田":    "Iwata",
    "東京V":   "Verdy",
    "柏":      "Kashiwa Reysol",
    "仙台":    "Vegalta Sendai",
    "新潟":    "Albirex Niigata",
    "甲府":    "Kofu",
    "横浜FC":  "Yokohama FC",
    "山形":    "Montedio Yamagata",
    "FC東京":  "FC Tokyo",
    "広島":    "Sanfrecce Hiroshima",
    "神戸":    "Vissel Kobe",
    "湘南":    "Shonan Bellmare",
    # J2/J3でデータなし
    "水戸":    None,
    "千葉":    None,
    "今治":    None,
    "藤枝":    None,
    "栃木C":   None,
    "八戸":    None,
    "秋田":    None,
}

# ── 第1622回 対戦カード ──────────────────────────────────────────────────
MATCHES = [
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

LABEL = {"1": "ホーム勝", "0": "引き分け", "2": "アウェイ勝"}


def build_elo_dict(df, feat_df):
    """全チームの最新Eloを辞書で返す"""
    elo = {}
    for _, row in feat_df.iterrows():
        # 試合後のEloは次試合の home_elo/away_elo に反映されているので
        # 最後の出場試合の相手方向から取得
        elo[row["home_team"]] = row["home_elo"]
        elo[row["away_team"]] = row["away_elo"]
    return elo


def predict_with_elo(home_elo, away_elo):
    """Eloだけで確率推定 (ロジスティック変換)"""
    diff = home_elo - away_elo
    # ホーム有利ボーナス +50
    adj_diff = diff + 50
    p_home = 1 / (1 + 10 ** (-adj_diff / 400))
    p_away = 1 / (1 + 10 ** (adj_diff / 400))
    p_draw = max(0.0, 1 - p_home - p_away) * 0.8  # 引き分けは低め
    # 正規化
    total = p_home + p_draw + p_away
    p_home /= total
    p_draw /= total
    p_away /= total
    return p_home, p_draw, p_away


def main():
    # データ読み込み・学習
    df = pd.read_csv("data/raw/jleague_results.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    builder = FeatureBuilder(form_window=5)
    feat_df = builder.build(df)

    feature_cols = [c for c in get_feature_columns(include_odds=True) if c in feat_df.columns]
    X_all = feat_df[feature_cols].fillna(0)
    y_all = feat_df["result"].astype(str)

    rf = ExtraTreesModel(include_odds=True)
    rf.fit(X_all, y_all)

    # 各チームの最新フォーム行と Elo を取得
    latest = {}
    for team in feat_df["home_team"].unique():
        rows = feat_df[feat_df["home_team"] == team]
        latest[team] = rows.iloc[-1]
    for team in feat_df["away_team"].unique():
        rows = feat_df[feat_df["away_team"] == team]
        if team not in latest:
            latest[team] = rows.iloc[-1]

    elo_dict = build_elo_dict(df, feat_df)

    # 予想出力
    print()
    print("=" * 70)
    print(" toto 第1622回 予想 (RF+オッズ / Eloフォールバック)")
    print("=" * 70)
    print(f"{'No':>2}  {'ホーム':<9} {'':>2} {'アウェイ':<9}  {'予想':<8}  確率(1/0/2)")
    print("-" * 70)

    total_pred = []
    for i, (home_jp, away_jp) in enumerate(MATCHES, 1):
        home_en = TEAM_MAP.get(home_jp)
        away_en = TEAM_MAP.get(away_jp)

        method = "RF"

        if home_en and away_en and home_en in latest and away_en in latest:
            feat_row = latest[home_en][feature_cols].fillna(0)
            pred_df = pd.DataFrame([feat_row])
            pred = rf.predict(pred_df)[0]
            proba = rf.predict_proba(pred_df)[0]

        elif home_en or away_en:
            # どちらかだけデータあり → Elo推定
            method = "Elo"
            h_elo = elo_dict.get(home_en, 1500) if home_en else 1500
            a_elo = elo_dict.get(away_en, 1500) if away_en else 1500
            ph, pd_, pa = predict_with_elo(h_elo, a_elo)
            proba = [ph, pd_, pa]
            pred = "1" if ph == max(ph, pd_, pa) else ("0" if pd_ == max(ph, pd_, pa) else "2")

        else:
            # 両チームともデータなし → ホームアドバンテージ
            method = "基本"
            proba = [0.45, 0.27, 0.28]
            pred = "1"

        label = LABEL[pred]
        total_pred.append(pred)
        print(
            f"{i:>2}. {home_jp:<9} vs {away_jp:<9}  "
            f"【{label:<6}】  "
            f"1:{proba[0]:.2f} / 0:{proba[1]:.2f} / 2:{proba[2]:.2f}  ({method})"
        )

    print("-" * 70)
    print(f"  予想配列: [{' / '.join(total_pred)}]")
    print()
    print("  ※ RF = RF+オッズモデル (Eloレーティング・直近フォーム・オッズ特徴量を使用)")
    print("  ※ Elo = Eloレーティングのみによる推定 (過去データが少ないチーム)")
    print("  ※ 基本 = ホームアドバンテージ推定 (データなし)")
    print()
    print("  [注意] 本予想は機械学習モデルによるものです。")
    print("    バックテスト正答率: 47.97% (ExtraTrees+Optuna) | 参考情報としてご利用ください。")
    print("=" * 70)


if __name__ == "__main__":
    main()
