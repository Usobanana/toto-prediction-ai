# -*- coding: utf-8 -*-
"""
特徴量重要度分析
================
RF+オッズモデルの特徴量重要度を表示する。

実行:
  python analyze_features.py
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import warnings; warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from src.features.feature_builder import FeatureBuilder, get_feature_columns
from src.models.ml_models import RandomForestModel

def main():
    print("データ読み込み・学習中...")
    df = pd.read_csv("data/raw/jleague_results.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    feat_df = FeatureBuilder(form_window=5).build(df)

    fc = get_feature_columns(include_odds=True)
    cols = [c for c in fc if c in feat_df.columns]
    X = feat_df[cols].fillna(0)
    y = feat_df["result"].astype(str)

    rf = RandomForestModel(include_odds=True)
    rf.fit(X, y)

    imp = rf.feature_importances(X)

    print()
    print("=" * 55)
    print("  RF+オッズ 特徴量重要度 TOP20")
    print("=" * 55)
    print(f"  {'特徴量':<35}  {'重要度':>8}  バー")
    print("  " + "-" * 52)
    for feat_name, importance in imp.head(20).items():
        bar = "#" * int(importance * 500)
        print(f"  {feat_name:<35}  {importance:.5f}  {bar}")

    print()
    print("  [グループ別集計]")
    groups = {
        "オッズ/implied_prob": [c for c in cols if "odds" in c or "implied" in c],
        "ELO": [c for c in cols if "elo" in c],
        "フォーム(直近5試合)": [c for c in cols if "form" in c and "_3" not in c],
        "フォーム(直近3試合)": [c for c in cols if "_3" in c],
        "ホーム/アウェイ成績": [c for c in cols if "home_home" in c or "away_away" in c],
        "H2H": [c for c in cols if "h2h" in c],
        "その他": [c for c in cols if "rest_days" in c or "draw_rate" in c or "season" in c or "games_played" in c],
    }
    for grp, gcols in groups.items():
        total = imp[gcols].sum() if gcols else 0
        print(f"  {grp:<25}: {total:.4f} ({total*100:.1f}%)")

if __name__ == "__main__":
    main()
