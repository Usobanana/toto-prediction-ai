# -*- coding: utf-8 -*-
"""
Optuna によるランダムフォレスト ハイパーパラメータ最適化
==========================================================
5-fold 時系列 CV の平均正答率を最大化するパラメータを探索。

実行:
  python optimize_rf.py [--trials 50]

最適パラメータを表示し、ml_models.py への反映方法を案内する。
"""
import io, sys, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.features.feature_builder import FeatureBuilder, get_feature_columns

MIN_TRAIN = 500
N_SPLITS  = 5
LABEL_MAP = {"1": 0, "0": 1, "2": 2}


def load():
    df = pd.read_csv("data/raw/jleague_results.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return FeatureBuilder(form_window=5).build(df)


def cv_score(feat_df, clf_factory):
    """TimeSeriesSplit 5-fold の平均正答率を返す"""
    fc   = get_feature_columns(include_odds=True)
    y    = feat_df["result"].astype(str)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    accs = []

    for train_idx, test_idx in tscv.split(feat_df):
        if len(train_idx) < MIN_TRAIN:
            continue
        tr = feat_df.iloc[train_idx]
        te = feat_df.iloc[test_idx]
        y_tr = np.array([LABEL_MAP.get(v, 0) for v in y.iloc[train_idx]])
        y_te = np.array([LABEL_MAP.get(v, 0) for v in y.iloc[test_idx]])

        cols = [c for c in fc if c in tr.columns]
        clf  = clf_factory()
        clf.fit(tr[cols].fillna(0), y_tr)
        pred = clf.predict(te[cols].fillna(0))
        accs.append(accuracy_score(y_te, pred))

    return float(np.mean(accs)) if accs else 0.0


def objective_rf(trial, feat_df):
    n_est    = trial.suggest_int("n_estimators", 200, 600, step=100)
    max_d    = trial.suggest_int("max_depth", 4, 16)
    min_samp = trial.suggest_int("min_samples_leaf", 2, 15)
    max_feat = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7])
    min_split= trial.suggest_int("min_samples_split", 2, 20)

    def make():
        return RandomForestClassifier(
            n_estimators=n_est, max_depth=max_d,
            min_samples_leaf=min_samp, max_features=max_feat,
            min_samples_split=min_split,
            random_state=42, n_jobs=-1,
        )
    return cv_score(feat_df, make)


def objective_extra(trial, feat_df):
    n_est    = trial.suggest_int("n_estimators", 200, 600, step=100)
    max_d    = trial.suggest_int("max_depth", 4, 16)
    min_samp = trial.suggest_int("min_samples_leaf", 2, 15)
    max_feat = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7])

    def make():
        return ExtraTreesClassifier(
            n_estimators=n_est, max_depth=max_d,
            min_samples_leaf=min_samp, max_features=max_feat,
            random_state=42, n_jobs=-1,
        )
    return cv_score(feat_df, make)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40, help="Optuna試行回数")
    args = parser.parse_args()

    print("データ読み込み中...")
    feat_df = load()
    print(f"  試合数: {len(feat_df)}  特徴量数: {len(get_feature_columns(include_odds=True))}")

    # ── ベースライン ──────────────────────────────────────────────────────
    base_score = cv_score(
        feat_df,
        lambda: RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
    )
    print(f"\nベースライン RF+odds: {base_score:.4f}")

    # ── RF 最適化 ─────────────────────────────────────────────────────────
    print(f"\nOptuna RF 探索 ({args.trials}試行)...")
    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(
        lambda t: objective_rf(t, feat_df),
        n_trials=args.trials,
        show_progress_bar=True,
    )
    best_rf = study_rf.best_trial
    print(f"\n[RF] 最良スコア: {best_rf.value:.4f}  (+{best_rf.value - base_score:+.4f})")
    print(f"[RF] 最良パラメータ:")
    for k, v in best_rf.params.items():
        print(f"  {k}: {v}")

    # ── ExtraTrees 最適化 ─────────────────────────────────────────────────
    print(f"\nOptuna ExtraTrees 探索 ({args.trials}試行)...")
    study_et = optuna.create_study(direction="maximize")
    study_et.optimize(
        lambda t: objective_extra(t, feat_df),
        n_trials=args.trials,
        show_progress_bar=True,
    )
    best_et = study_et.best_trial
    print(f"\n[ExtraTrees] 最良スコア: {best_et.value:.4f}  (+{best_et.value - base_score:+.4f})")
    print(f"[ExtraTrees] 最良パラメータ:")
    for k, v in best_et.params.items():
        print(f"  {k}: {v}")

    # ── まとめ ────────────────────────────────────────────────────────────
    winner = "RF" if best_rf.value >= best_et.value else "ExtraTrees"
    best_score = max(best_rf.value, best_et.value)
    print(f"\n==== 結論 ====")
    print(f"ベースライン: {base_score:.4f}")
    print(f"最良モデル:   {winner}  {best_score:.4f}  ({best_score - base_score:+.4f})")

    # 最良パラメータを JSON で保存
    import json
    best_params = {
        "model": winner,
        "score": round(best_score, 4),
        "baseline": round(base_score, 4),
        "params": best_rf.params if winner == "RF" else best_et.params,
    }
    with open("data/best_rf_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    print(f"\n  → data/best_rf_params.json に保存")


if __name__ == "__main__":
    main()
