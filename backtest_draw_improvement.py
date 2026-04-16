# -*- coding: utf-8 -*-
"""
引き分け予測改善 バックテスト比較
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

from src.features.feature_builder import FeatureBuilder, get_feature_columns
from src.models.ml_models import RandomForestModel
from src.models.draw_improved import (
    BalancedRFModel, DrawFeatureRFModel, ThresholdTunedModel, add_draw_features
)

N_SPLITS = 5
MIN_TRAIN = 200


def run_backtest(model, X, y, feature_cols):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    all_true, all_pred = [], []

    for train_idx, test_idx in tscv.split(X):
        if len(train_idx) < MIN_TRAIN:
            continue
        train_cols = feature_cols + [c for c in ["home_team","away_team"] if c in X.columns]
        avail_cols = [c for c in train_cols if c in X.columns]
        model.fit(X.iloc[train_idx][avail_cols], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx][avail_cols])
        all_true.extend(y.iloc[test_idx].tolist())
        all_pred.extend(preds.tolist())

    return np.array(all_true), np.array(all_pred)


def print_result(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    # クラス別
    for r, label in [("1","ホーム勝"), ("0","引き分け"), ("2","アウェイ勝")]:
        tp = ((y_pred == r) & (y_true == r)).sum()
        fp = ((y_pred == r) & (y_true != r)).sum()
        fn = ((y_pred != r) & (y_true == r)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2*prec*rec / (prec+rec+1e-9)
        pred_n = (y_pred == r).sum()
        true_n = (y_true == r).sum()
        marker = " <<<" if r == "0" else ""
        print(f"    {label}: 予測{pred_n:>4}件 / 実際{true_n}件  P:{prec:.2f} R:{rec:.2f} F1:{f1:.2f}{marker}")

    return acc


def main():
    # データ準備
    df = pd.read_csv("data/raw/jleague_results.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    builder = FeatureBuilder(form_window=5)
    feat_df = builder.build(df)
    feat_df = add_draw_features(feat_df)  # 引き分け専用特徴量を追加

    base_cols = [c for c in get_feature_columns(include_odds=False) if c in feat_df.columns]

    y = feat_df["result"].astype(str)

    models = [
        ("現状 (ベースライン)",     RandomForestModel(),      base_cols),
        ("A: balanced weight",     BalancedRFModel(),         base_cols),
        ("B: 引き分け特徴量+balanced", DrawFeatureRFModel(),  None),  # None=モデル内で決定
        ("C: 閾値チューニング",    ThresholdTunedModel(),     None),
    ]

    results = {}

    print()
    print("=" * 65)
    print(" 引き分け予測改善 バックテスト比較")
    print("=" * 65)

    for name, model, feat_cols in models:
        print(f"\n[{name}]")

        if feat_cols is None:
            # モデルが内部でfeature列を管理
            from src.models.draw_improved import get_draw_feature_columns
            feat_cols = [c for c in get_draw_feature_columns() if c in feat_df.columns]

        y_true, y_pred = run_backtest(model, feat_df, y, feat_cols)
        acc = print_result(name, y_true, y_pred)
        results[name] = {"acc": acc, "y_true": y_true, "y_pred": y_pred}
        print(f"    全体正答率: {acc:.4f}")

        # 閾値チューニングモデルは使用閾値も表示
        if hasattr(model, 'draw_threshold'):
            print(f"    使用閾値: {model.draw_threshold:.2f}")

    # 比較サマリー
    print()
    print("=" * 65)
    print(" 比較サマリー")
    print("=" * 65)
    print(f"  {'モデル':<28} {'全体正答率':>8}  {'引き分け予測数':>12}  {'引き分けF1':>10}")
    print("-" * 65)

    for name, res in results.items():
        y_true, y_pred = res["y_true"], res["y_pred"]
        draw_pred_n = (y_pred == "0").sum()
        tp = ((y_pred == "0") & (y_true == "0")).sum()
        fp = ((y_pred == "0") & (y_true != "0")).sum()
        fn = ((y_pred != "0") & (y_true == "0")).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2*prec*rec / (prec+rec+1e-9)
        print(
            f"  {name:<28} {res['acc']:>8.4f}  {draw_pred_n:>12}件  {f1:>10.3f}"
        )

    print("=" * 65)
    print()
    print("  * 引き分けF1 = 2 × Precision × Recall / (Precision + Recall)")
    print("  * 0に近いほど引き分けを予測できていない")


if __name__ == "__main__":
    main()
