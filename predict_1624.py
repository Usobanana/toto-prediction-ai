# -*- coding: utf-8 -*-
"""
toto 第1624回 予想スクリプト
==============================
モデル: RandomForest+Optuna+オッズ (バックテスト正答率 48.15%)
  - 61特徴量 (会場別H2H 8特徴量を新たに追加)
  - toto公式対戦データ (rival_team_scraper) による H2H 補完対応

第1624回 対戦カード (2026/04/25):
  1. 仙台 vs 山形
  2. 清水 vs 名古屋
  3. 岡山 vs 福岡
  4. 浦和 vs 横浜FM
  5. 川崎F vs 千葉
  6. 長崎 vs G大阪
  7. 札幌 vs いわき
  8. 藤枝 vs 大宮
  9. 相模原 vs 湘南
 10. 今治 vs 富山
 11. 福島 vs 甲府
 12. 宮崎 vs 鹿児島
 13. 新潟 vs FC大阪
"""
import sys
import csv
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
from src.features.feature_builder import FeatureBuilder, get_feature_columns
from src.models.ml_models import RandomForestModel

# ── チーム名マッピング (日本語toto名 → football-data.co.uk 英語表記) ─────
TEAM_MAP = {
    "横浜FM":  "Yokohama F. Marinos",
    "川崎F":   "Kawasaki Frontale",
    "川崎Ｆ":  "Kawasaki Frontale",
    "鹿島":    "Kashima Antlers",
    "浦和":    "Urawa Reds",
    "C大阪":   "Cerezo Osaka",
    "京都":    "Kyoto",
    "G大阪":   "Gamba Osaka",
    "Ｇ大阪":  "Gamba Osaka",
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
    "清水":    "Shimizu S-Pulse",
    "長崎":    "V-Varen Nagasaki",
    "札幌":    "Hokkaido Consadole Sapporo",
    # データなし (J2/J3/新興)
    "千葉":    None,
    "いわき":  None,
    "藤枝":    None,
    "相模原":  None,
    "今治":    None,
    "富山":    None,
    "福島":    None,
    "宮崎":    None,
    "鹿児島":  None,
    "FC大阪":  None,
    "水戸":    None,
    "栃木C":   None,
    "八戸":    None,
    "秋田":    None,
}

# ── 第1624回 対戦カード ─────────────────────────────────────────────────────
MATCHES = [
    ("仙台",   "山形"),    #  1
    ("清水",   "名古屋"),  #  2
    ("岡山",   "福岡"),    #  3
    ("浦和",   "横浜FM"),  #  4
    ("川崎F",  "千葉"),    #  5
    ("長崎",   "G大阪"),   #  6
    ("札幌",   "いわき"),  #  7
    ("藤枝",   "大宮"),    #  8
    ("相模原", "湘南"),    #  9
    ("今治",   "富山"),    # 10
    ("福島",   "甲府"),    # 11
    ("宮崎",   "鹿児島"),  # 12
    ("新潟",   "FC大阪"),  # 13
]

LABEL = {"1": "ホーム勝", "0": "引き分け", "2": "アウェイ勝"}

# ── toto公式H2Hデータ読み込み ────────────────────────────────────────────────
def load_toto_h2h(hold_cnt_id: int = 1624) -> dict:
    """
    rival_team_scraper が生成した toto_h2h_summary.csv を読み込む。
    { (home_jp, away_jp): {h2h列: 値, ...} } 形式で返す。
    """
    path = Path("data/raw/toto_h2h_summary.csv")
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        df = df[df["hold_cnt_id"] == hold_cnt_id]
        result = {}
        for _, row in df.iterrows():
            key = (row["home_team"], row["away_team"])
            result[key] = row.to_dict()
        return result
    except Exception:
        return {}


# ── ユーティリティ ───────────────────────────────────────────────────────────
def build_elo_dict(feat_df):
    elo = {}
    for _, row in feat_df.iterrows():
        elo[row["home_team"]] = row["home_elo"]
        elo[row["away_team"]] = row["away_elo"]
    return elo


def predict_with_elo(home_elo, away_elo):
    diff = home_elo - away_elo + 50
    p_home = 1 / (1 + 10 ** (-diff / 400))
    p_away = 1 / (1 + 10 ** (diff / 400))
    p_draw = max(0.0, 1 - p_home - p_away) * 0.8
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


def best_label(proba):
    idx = int(np.argmax(proba))
    return ["1", "0", "2"][idx]


def main():
    # ── データ読み込み・特徴量構築 ──────────────────────────────────────────
    df = pd.read_csv("data/raw/jleague_results.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    builder = FeatureBuilder(form_window=5)
    feat_df = builder.build(df)

    feature_cols = [c for c in get_feature_columns(include_odds=True) if c in feat_df.columns]
    X_all = feat_df[feature_cols].fillna(0)
    y_all = feat_df["result"].astype(str)

    rf = RandomForestModel(include_odds=True)
    rf.fit(X_all, y_all)

    # 各チームの最新特徴量行
    latest = {}
    for col, team_col in [("home_team", "home_team"), ("away_team", "away_team")]:
        for team in feat_df[team_col].unique():
            rows = feat_df[feat_df[team_col] == team]
            if team not in latest:
                latest[team] = rows.iloc[-1]

    elo_dict = build_elo_dict(feat_df)

    # toto公式H2Hデータ
    toto_h2h = load_toto_h2h(1624)
    h2h_loaded = len(toto_h2h) > 0

    # ── 予想出力 ────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(" toto 第1624回 予想 (RF+Optuna 48.15% / 61特徴量 + 会場別H2H)")
    if h2h_loaded:
        print(f" ※ toto公式対戦データ取得済み ({len(toto_h2h)}試合)")
    print("=" * 72)
    print(f"{'No':>2}  {'ホーム':<9} {'':>2} {'アウェイ':<9}  {'予想':<8}  確率(1/0/2)")
    print("-" * 72)

    total_pred = []
    for i, (home_jp, away_jp) in enumerate(MATCHES, 1):
        home_en = TEAM_MAP.get(home_jp)
        away_en = TEAM_MAP.get(away_jp)
        method = "RF"

        if home_en and away_en and home_en in latest and away_en in latest:
            feat_row = latest[home_en][feature_cols].copy().fillna(0)

            # toto公式H2Hデータで上書き (取得済みの場合)
            toto_key_jp = (home_jp, away_jp)
            if toto_key_jp in toto_h2h:
                h2h_row = toto_h2h[toto_key_jp]
                h2h_col_map = {
                    "h2h_home_win_home":  "h2h_home_venue_win_rate",
                    "h2h_away_win_home":  "h2h_home_venue_away_win_rate",
                    "h2h_draw_home":      "h2h_home_venue_draw_rate",
                    "h2h_home_win_away":  "h2h_away_venue_win_rate",
                    "h2h_away_win_away":  "h2h_away_venue_away_win_rate",
                    "h2h_draw_away":      "h2h_away_venue_draw_rate",
                }
                home_total = (h2h_row.get("h2h_home_win_home", 0) or 0) + (h2h_row.get("h2h_draw_home", 0) or 0) + (h2h_row.get("h2h_away_win_home", 0) or 0)
                away_total = (h2h_row.get("h2h_home_win_away", 0) or 0) + (h2h_row.get("h2h_draw_away", 0) or 0) + (h2h_row.get("h2h_away_win_away", 0) or 0)
                if home_total > 0:
                    feat_row["h2h_home_venue_win_rate"]      = (h2h_row.get("h2h_home_win_home", 0) or 0) / home_total
                    feat_row["h2h_home_venue_draw_rate"]     = (h2h_row.get("h2h_draw_home", 0) or 0) / home_total
                    feat_row["h2h_home_venue_away_win_rate"] = (h2h_row.get("h2h_away_win_home", 0) or 0) / home_total
                    feat_row["h2h_home_venue_count"]         = home_total
                if away_total > 0:
                    feat_row["h2h_away_venue_win_rate"]      = (h2h_row.get("h2h_home_win_away", 0) or 0) / away_total
                    feat_row["h2h_away_venue_draw_rate"]     = (h2h_row.get("h2h_draw_away", 0) or 0) / away_total
                    feat_row["h2h_away_venue_away_win_rate"] = (h2h_row.get("h2h_away_win_away", 0) or 0) / away_total
                    feat_row["h2h_away_venue_count"]         = away_total
                method = "RF+H2H"

            pred_df = pd.DataFrame([feat_row])
            proba = rf.predict_proba(pred_df)[0]
            pred = best_label(proba)

        elif home_en or away_en:
            method = "Elo"
            h_elo = elo_dict.get(home_en, 1500) if home_en else 1500
            a_elo = elo_dict.get(away_en, 1500) if away_en else 1500
            ph, pd_, pa = predict_with_elo(h_elo, a_elo)
            proba = [ph, pd_, pa]
            pred = best_label(proba)

        else:
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

    print("-" * 72)
    print(f"  予想配列: [{' / '.join(total_pred)}]")
    print()
    col_note = f"  ※ RF = RandomForest+Optuna ({len(feature_cols)}特徴量: Elo・フォーム・オッズ・スタンディング・市場価値・会場別H2H)"
    print(col_note)
    if h2h_loaded:
        print("  ※ RF+H2H = toto公式対戦データでH2H特徴量を補正")
    print("  ※ Elo = Eloレーティングのみ (データ少チーム)")
    print("  ※ 基本 = ホームアドバンテージ推定 (データなし)")
    print()
    print("  [注意] バックテスト正答率: 48.15% | 参考情報としてご利用ください。")
    print("=" * 72)


if __name__ == "__main__":
    main()
