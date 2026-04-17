# -*- coding: utf-8 -*-
"""
ポアソンモデル バックテスト＆比較
==================================
既存モデルとポアソンモデルを時系列5分割CVで比較。
ポアソンモデルには home_score / away_score が必要なため
専用のラッパーで Backtester に渡す。

実行:
  python backtest_poisson.py
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

from src.features.feature_builder import FeatureBuilder, get_feature_columns
from src.models.ml_models import RandomForestModel
from src.models.poisson_model import PoissonModel


# ──────────────────────────────────────────────────────────────────────
# データ準備
# ──────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv("data/raw/jleague_results.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def build_features(df):
    builder = FeatureBuilder(form_window=5)
    feat_df = builder.build(df)
    return feat_df


# ──────────────────────────────────────────────────────────────────────
# 時系列CV ユーティリティ
# ──────────────────────────────────────────────────────────────────────

def timeseries_cv(feat_df, fit_fn, predict_fn, n_splits=5, min_train=200):
    """汎用時系列CV。fit_fn / predict_fn はインデックスを受け取る関数"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_true, all_pred = [], []

    idx_arr = feat_df.index.values
    y = feat_df["result"].astype(str)

    fold_accs = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(idx_arr)):
        if len(train_idx) < min_train:
            continue

        train_df = feat_df.iloc[train_idx]
        test_df  = feat_df.iloc[test_idx]
        y_train  = y.iloc[train_idx]
        y_test   = y.iloc[test_idx]

        preds = predict_fn(train_df, test_df, y_train)
        acc = accuracy_score(y_test, preds)
        fold_accs.append(acc)

        split_date = str(feat_df["date"].iloc[test_idx[0]])[:10]
        print(f"  Fold {fold+1}: acc={acc:.3f}  (train={len(train_idx)}, test={len(test_idx)}, from={split_date})")

        all_true.extend(y_test.tolist())
        all_pred.extend(preds.tolist())

    return all_true, all_pred, fold_accs


# ──────────────────────────────────────────────────────────────────────
# 各モデルの fit/predict ファクトリ
# ──────────────────────────────────────────────────────────────────────

def rf_fit_predict(train_df, test_df, y_train):
    feature_cols = [c for c in get_feature_columns(include_odds=False)
                    if c in train_df.columns]
    model = RandomForestModel()
    model.fit(train_df[feature_cols].fillna(0), y_train)
    return model.predict(test_df[feature_cols].fillna(0))


def poisson_basic_fit_predict(train_df, test_df, y_train):
    model = PoissonModel(dc_rho=0.0)  # 補正なし
    cols = ["home_team", "away_team", "home_score", "away_score"]
    model.fit(train_df[cols], y_train)
    return model.predict(test_df[["home_team", "away_team"]])


def poisson_dc_fit_predict(train_df, test_df, y_train):
    model = PoissonModel(dc_rho=-0.13)  # Dixon-Coles補正あり
    cols = ["home_team", "away_team", "home_score", "away_score"]
    model.fit(train_df[cols], y_train)
    return model.predict(test_df[["home_team", "away_team"]])


def poisson_decay_fit_predict(train_df, test_df, y_train):
    model = PoissonModel(dc_rho=-0.13, time_decay=365)  # 1年半減期
    cols = ["home_team", "away_team", "home_score", "away_score", "date"]
    model.fit(train_df[cols], y_train)
    return model.predict(test_df[["home_team", "away_team"]])


# ──────────────────────────────────────────────────────────────────────
# 結果表示
# ──────────────────────────────────────────────────────────────────────

def print_detail(name, all_true, all_pred, fold_accs):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # フォルド精度
    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    print(f"  平均正答率: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  フォルド別: {[f'{a:.3f}' for a in fold_accs]}")

    # クラス別精度
    report = classification_report(
        all_true, all_pred,
        labels=["1", "0", "2"],
        target_names=["ホーム勝(1)", "引き分け(0)", "アウェイ勝(2)"],
        zero_division=0,
    )
    print()
    print(report)

    # 引き分け予測数
    n_draw_pred = sum(1 for p in all_pred if p == "0")
    n_draw_true = sum(1 for t in all_true if t == "0")
    print(f"  引き分け予測数: {n_draw_pred} / 実際: {n_draw_true} ({n_draw_true/len(all_true)*100:.1f}%)")

    return mean_acc, std_acc


# ──────────────────────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────────────────────

def main():
    print("データ読み込み中...")
    df = load_data()
    feat_df = build_features(df)
    print(f"  試合数: {len(feat_df)}")

    configs = [
        ("RandomForest (既存)",         rf_fit_predict),
        ("Poisson (補正なし)",           poisson_basic_fit_predict),
        ("Poisson (Dixon-Coles補正)",    poisson_dc_fit_predict),
        ("Poisson (DC+時間減衰365日)",   poisson_decay_fit_predict),
    ]

    summary = []

    print()
    print("=" * 60)
    print("  バックテスト実行 (時系列5分割CV)")
    print("=" * 60)

    for name, fn in configs:
        print(f"\n>> {name}")
        all_true, all_pred, fold_accs = timeseries_cv(feat_df, None, fn)
        mean_acc, std_acc = print_detail(name, all_true, all_pred, fold_accs)
        n_draw = sum(1 for p in all_pred if p == "0")
        summary.append({
            "モデル": name,
            "平均正答率": f"{mean_acc:.4f}",
            "標準偏差":   f"{std_acc:.4f}",
            "引き分け予測数": n_draw,
        })

    # 比較テーブル
    print()
    print("=" * 60)
    print("  【まとめ比較】")
    print("=" * 60)
    summary_df = pd.DataFrame(summary).sort_values("平均正答率", ascending=False)
    print(summary_df.to_string(index=False))
    print()

    # 最良モデルでスコア行列の例示（横浜FM vs 川崎F）
    print("=" * 60)
    print("  【スコア確率行列の例】横浜FM vs 川崎F")
    print("  (第1622回 試合1 参考)")
    print("=" * 60)
    df_full = feat_df
    y_full = df_full["result"].astype(str)
    cols = ["home_team", "away_team", "home_score", "away_score", "date"]
    best_model = PoissonModel(dc_rho=-0.13, time_decay=365)
    best_model.fit(df_full[cols], y_full)

    try:
        score_matrix = best_model.score_probability(
            "Yokohama F. Marinos", "Kawasaki Frontale", max_score=5
        )
        print()
        print(score_matrix.to_string())
        print()
        p1, p0, p2 = best_model._score_probs(
            best_model._mu_home * best_model._attack.get("Yokohama F. Marinos", 1)
            * best_model._defense.get("Kawasaki Frontale", 1),
            best_model._mu_away * best_model._attack.get("Kawasaki Frontale", 1)
            * best_model._defense.get("Yokohama F. Marinos", 1),
        )
        print(f"  P(ホーム勝)={p1:.3f}  P(引き分け)={p0:.3f}  P(アウェイ勝)={p2:.3f}")
    except Exception as e:
        print(f"  (スコア行列の表示に失敗: {e})")

    print()


if __name__ == "__main__":
    main()
