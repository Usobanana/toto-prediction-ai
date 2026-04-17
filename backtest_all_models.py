# -*- coding: utf-8 -*-
"""
全モデル 総合バックテスト
==========================
精度改善の各施策を 5-fold 時系列 CV で一括比較。

実行:
  python backtest_all_models.py
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, log_loss

from src.features.feature_builder import FeatureBuilder, get_feature_columns
from src.models.poisson_model import PoissonModel
from src.models.hierarchical_poisson import HierarchicalPoissonModel
from src.models.ml_models import (
    RandomForestModel, RandomForestDrawModel, XGBoostModel, LightGBMModel, MLPModel
)
from src.models.stacking_model import StackingModel
from src.models.calibrated_model import CalibratedModel

MIN_TRAIN = 500
N_SPLITS  = 5


# ──────────────────────────────────────────────────────────────────────
# データ読み込み
# ──────────────────────────────────────────────────────────────────────

def load():
    df = pd.read_csv("data/raw/jleague_results.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    feat = FeatureBuilder(form_window=5).build(df)
    return feat


# ──────────────────────────────────────────────────────────────────────
# fit/predict ファクトリ
# ──────────────────────────────────────────────────────────────────────

def make_fn_sklearn(ModelClass, include_odds=True):
    """特徴量ベース (sklearn 系) のfit/predict関数を返す"""
    fc = get_feature_columns(include_odds=include_odds)
    def fn(train_df, test_df, y_train):
        cols = [c for c in fc if c in train_df.columns]
        m = ModelClass()
        m.fit(train_df[cols].fillna(0), y_train)
        return m.predict(test_df[cols].fillna(0))
    return fn

def make_fn_sklearn_proba(ModelClass, include_odds=True):
    """確率も返す版"""
    fc = get_feature_columns(include_odds=include_odds)
    def fn(train_df, test_df, y_train):
        cols = [c for c in fc if c in train_df.columns]
        m = ModelClass()
        m.fit(train_df[cols].fillna(0), y_train)
        pred  = m.predict(test_df[cols].fillna(0))
        proba = m.predict_proba(test_df[cols].fillna(0))
        return pred, proba
    return fn

def hier_fn(k=5):
    cols = ["home_team", "away_team", "home_score", "away_score", "date"]
    def fn(train_df, test_df, y_train):
        m = HierarchicalPoissonModel(prior_strength=k, dc_rho=-0.13, time_decay=365)
        m.fit(train_df[cols], y_train)
        return m.predict(test_df[["home_team", "away_team"]])
    return fn

def poisson_fn(train_df, test_df, y_train):
    cols = ["home_team", "away_team", "home_score", "away_score", "date"]
    m = PoissonModel(dc_rho=-0.13, time_decay=365)
    m.fit(train_df[cols], y_train)
    return m.predict(test_df[["home_team", "away_team"]])

def calibrated_fn(ModelClass, include_odds=True):
    fc = get_feature_columns(include_odds=include_odds)
    def fn(train_df, test_df, y_train):
        cols = [c for c in fc if c in train_df.columns]
        base = ModelClass()
        m = CalibratedModel(base, method="isotonic", cv=3)
        m.fit(train_df[cols].fillna(0), y_train)
        return m.predict(test_df[cols].fillna(0))
    return fn

def _make_hier_wrapper():
    class HierWrapper:
        def __init__(self):
            self._m = HierarchicalPoissonModel(prior_strength=5, dc_rho=-0.13, time_decay=365)
        def fit(self, X, y):
            c = ["home_team","away_team","home_score","away_score","date"]
            self._m.fit(X[c], y)
        def predict_proba(self, X):
            return self._m.predict_proba(X[["home_team","away_team"]])
        def predict(self, X):
            return self._m.predict(X[["home_team","away_team"]])
    return HierWrapper()

def _make_col_filter(model_cls, cols):
    class ColFilter:
        def __init__(self):
            self._m = model_cls(); self._cols = cols
        def fit(self, X, y):
            c = [cc for cc in self._cols if cc in X.columns]
            self._m.fit(X[c].fillna(0), y)
        def predict_proba(self, X):
            c = [cc for cc in self._cols if cc in X.columns]
            return self._m.predict_proba(X[c].fillna(0))
        def predict(self, X):
            c = [cc for cc in self._cols if cc in X.columns]
            return self._m.predict(X[c].fillna(0))
    return ColFilter()

def stacking_fn(train_df, test_df, y_train):
    """v1: HierBayes + LightGBM(odds) + RF のスタッキング"""
    cols = get_feature_columns(include_odds=True)
    base_models = [
        ("hier",    _make_hier_wrapper()),
        ("lgbm",    _make_col_filter(LightGBMModel, cols)),
        ("rf",      _make_col_filter(RandomForestModel, cols)),
    ]
    m = StackingModel(base_models, n_splits=3)
    m.fit(train_df, y_train)
    return m.predict(test_df)


def stacking_v2_fn(train_df, test_df, y_train):
    """v2: HierBayes + RF+odds + XGBoost (最強base構成)"""
    cols = get_feature_columns(include_odds=True)
    base_models = [
        ("hier",    _make_hier_wrapper()),
        ("rf",      _make_col_filter(RandomForestModel, cols)),
        ("xgb",     _make_col_filter(XGBoostModel, cols)),
    ]
    m = StackingModel(base_models, n_splits=3)
    m.fit(train_df, y_train)
    return m.predict(test_df)


# ──────────────────────────────────────────────────────────────────────
# CV 実行
# ──────────────────────────────────────────────────────────────────────

def cv_run(feat_df, fn, label=""):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    y = feat_df["result"].astype(str)
    all_true, all_pred, fold_accs = [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(feat_df)):
        if len(train_idx) < MIN_TRAIN:
            continue
        train_df = feat_df.iloc[train_idx]
        test_df  = feat_df.iloc[test_idx]
        y_train  = y.iloc[train_idx]
        y_test   = y.iloc[test_idx]

        result = fn(train_df, test_df, y_train)
        # stacking_fn などが tuple を返す場合
        preds = result[0] if isinstance(result, tuple) else result

        acc = accuracy_score(y_test, preds)
        fold_accs.append(acc)
        all_true.extend(y_test.tolist())
        all_pred.extend(preds.tolist() if hasattr(preds, "tolist") else list(preds))

    mean_acc = np.mean(fold_accs) if fold_accs else 0
    std_acc  = np.std(fold_accs)  if fold_accs else 0

    report = classification_report(
        all_true, all_pred,
        labels=["1", "0", "2"],
        target_names=["Home(1)", "Draw(0)", "Away(2)"],
        zero_division=0, output_dict=True
    )
    draw_f1   = report["Draw(0)"]["f1-score"]
    draw_prec = report["Draw(0)"]["precision"]
    draw_rec  = report["Draw(0)"]["recall"]

    return mean_acc, std_acc, draw_f1, draw_prec, draw_rec


# ──────────────────────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────────────────────

def main():
    print("データ読み込み・特徴量生成中...")
    feat_df = load()
    print(f"  試合数: {len(feat_df)}  /  特徴量数: {len(get_feature_columns(include_odds=True))}")
    print()

    configs = [
        # ── ベースライン ──────────────────────────────────────────
        ("① Poisson DC+Decay          [旧最良]",    poisson_fn),
        ("② HierBayes k=5             [旧最良]",    hier_fn(5)),
        # ── 特徴量なし → あり ────────────────────────────────────
        ("③ RF  オッズなし             [旧RF]",     make_fn_sklearn(RandomForestModel, include_odds=False)),
        ("④ RF  オッズあり             [①odds]",   make_fn_sklearn(RandomForestModel, include_odds=True)),
        # ── 新モデル ─────────────────────────────────────────────
        ("⑤ XGBoost オッズあり         [②XGB]",    make_fn_sklearn(XGBoostModel,      include_odds=True)),
        ("⑥ LightGBM オッズあり        [③LGB]",    make_fn_sklearn(LightGBMModel,     include_odds=True)),
        ("⑦ MLP オッズあり             [④MLP]",    make_fn_sklearn(MLPModel,          include_odds=True)),
        # ── キャリブレーション ───────────────────────────────────
        ("⑧ LightGBM+Calibration      [⑤Cal]",    calibrated_fn(LightGBMModel,       include_odds=True)),
        # ── スタッキング ─────────────────────────────────────────
        ("⑨ Stacking(Hier+LGB+RF)     [⑥Stack]",  stacking_fn),
        ("⑩ Stacking v2(Hier+RF+XGB)  [⑦Stk2]",  stacking_v2_fn),
        # ── 引き分け強化 ─────────────────────────────────────────
        ("⑪ RF+Draw閾値補正           [⑧DrawRF]", make_fn_sklearn(RandomForestDrawModel, include_odds=True)),
    ]

    print("=" * 75)
    print("  5-fold 時系列 CV  全モデル比較")
    print("=" * 75)
    print(f"  {'モデル':<38}  {'正答率':>7}  {'±':>5}  {'引分F1':>6}  {'引分P':>6}  {'引分R':>6}")
    print("  " + "-" * 72)

    summary = []
    prev_best = 0.0

    for label, fn in configs:
        sys.stdout.write(f"  {label:<38}  実行中...\r")
        sys.stdout.flush()
        mean_acc, std_acc, draw_f1, draw_prec, draw_rec = cv_run(feat_df, fn, label)

        delta = mean_acc - prev_best
        delta_str = f"({delta:+.3f})" if prev_best > 0 else "       "
        print(f"  {label:<38}  {mean_acc:.4f}  ±{std_acc:.3f}  {draw_f1:.3f}   {draw_prec:.3f}   {draw_rec:.3f}  {delta_str}")
        sys.stdout.flush()

        summary.append({
            "モデル": label.split("[")[0].strip(),
            "正答率": f"{mean_acc:.4f}",
            "±":     f"{std_acc:.3f}",
            "引分F1": f"{draw_f1:.3f}",
            "引分P":  f"{draw_prec:.3f}",
            "引分R":  f"{draw_rec:.3f}",
        })
        if mean_acc > prev_best:
            prev_best = mean_acc

    print()
    print("=" * 75)
    print("  【最終ランキング】")
    print("=" * 75)
    df_sum = pd.DataFrame(summary).sort_values("正答率", ascending=False)
    print(df_sum.to_string(index=False))

    best = df_sum.iloc[0]
    base = float(summary[1]["正答率"])   # HierBayes k=5 が旧ベースライン
    best_acc = float(best["正答率"])
    print()
    print(f"  旧ベースライン (HierBayes k=5): {base:.4f}")
    print(f"  最良モデル: {best['モデル']}  →  {best_acc:.4f}  ({best_acc - base:+.4f})")
    print()
    print("  [解釈]")
    print("  ・引分F1 = 引き分け予測の精度。高いほど難しい引分を当てられる。")
    print("  ・±(標準偏差)が小さいほど安定している。")
    print("  ・オッズ特徴量(implied_prob_*)がある時点で市場の知恵を取り込める。")


if __name__ == "__main__":
    main()
