# -*- coding: utf-8 -*-
"""
階層ベイズモデル バックテスト
================================
通常ポアソン vs 階層ベイズポアソン を 5-fold 時系列 CV で比較。
縮小パラメータ(prior_strength)の感度も確認。

実行:
  python backtest_hierarchical.py
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

from src.features.feature_builder import FeatureBuilder
from src.models.poisson_model import PoissonModel
from src.models.hierarchical_poisson import HierarchicalPoissonModel
from src.models.ml_models import RandomForestModel


def load_data():
    df = pd.read_csv("data/raw/jleague_results.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    builder = FeatureBuilder(form_window=5)
    feat_df = builder.build(df)
    return feat_df


def cv_run(feat_df, fit_predict_fn, n_splits=5, min_train=200, label=""):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    y = feat_df["result"].astype(str)
    all_true, all_pred, fold_accs = [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(feat_df)):
        if len(train_idx) < min_train:
            continue
        train_df = feat_df.iloc[train_idx]
        test_df  = feat_df.iloc[test_idx]
        y_train  = y.iloc[train_idx]
        y_test   = y.iloc[test_idx]

        preds = fit_predict_fn(train_df, test_df, y_train)
        acc   = accuracy_score(y_test, preds)
        fold_accs.append(acc)
        all_true.extend(y_test.tolist())
        all_pred.extend(preds.tolist())

        split_date = str(feat_df["date"].iloc[test_idx[0]])[:10]
        print(f"    Fold {fold+1}: acc={acc:.3f} (from {split_date})")

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    n_draw_pred = sum(1 for p in all_pred if p == "0")
    n_draw_true = sum(1 for t in all_true if t == "0")

    print(f"  -> 平均正答率: {mean_acc:.4f} +/- {std_acc:.4f}")
    report = classification_report(
        all_true, all_pred,
        labels=["1", "0", "2"],
        target_names=["Home(1)", "Draw(0)", "Away(2)"],
        zero_division=0, output_dict=True
    )
    draw_f1 = report["Draw(0)"]["f1-score"]
    print(f"     引き分けF1: {draw_f1:.3f}  予測数: {n_draw_pred} / 実際: {n_draw_true}")

    return mean_acc, std_acc, draw_f1, all_true, all_pred


def rf_fn(train_df, test_df, y_train):
    from src.features.feature_builder import get_feature_columns
    fc = [c for c in get_feature_columns(include_odds=False) if c in train_df.columns]
    m = RandomForestModel()
    m.fit(train_df[fc].fillna(0), y_train)
    return m.predict(test_df[fc].fillna(0))


def poisson_fn(train_df, test_df, y_train):
    cols = ["home_team", "away_team", "home_score", "away_score", "date"]
    m = PoissonModel(dc_rho=-0.13, time_decay=365)
    m.fit(train_df[cols], y_train)
    return m.predict(test_df[["home_team", "away_team"]])


def hier_fn(prior_strength):
    def fn(train_df, test_df, y_train):
        cols = ["home_team", "away_team", "home_score", "away_score", "date"]
        m = HierarchicalPoissonModel(
            prior_strength=prior_strength, dc_rho=-0.13, time_decay=365
        )
        m.fit(train_df[cols], y_train)
        return m.predict(test_df[["home_team", "away_team"]])
    return fn


def main():
    print("データ読み込み中...")
    feat_df = load_data()
    print(f"  試合数: {len(feat_df)}")

    configs = [
        ("RandomForest (ベースライン)",       rf_fn),
        ("Poisson DC+Decay (前回最良)",        poisson_fn),
        ("HierBayes k=5  (弱い事前分布)",      hier_fn(5)),
        ("HierBayes k=10 (標準)",              hier_fn(10)),
        ("HierBayes k=20 (強い事前分布)",      hier_fn(20)),
        ("HierBayes k=50 (強制リーグ平均)",    hier_fn(50)),
    ]

    summary = []
    print()
    print("=" * 65)
    print("  5-fold 時系列 CV バックテスト")
    print("=" * 65)

    for name, fn in configs:
        print(f"\n>> {name}")
        mean_acc, std_acc, draw_f1, _, _ = cv_run(feat_df, fn)
        summary.append({
            "モデル": name,
            "平均正答率": f"{mean_acc:.4f}",
            "標準偏差":   f"{std_acc:.4f}",
            "引き分けF1": f"{draw_f1:.3f}",
        })

    print()
    print("=" * 65)
    print("  【比較結果】")
    print("=" * 65)
    df_sum = pd.DataFrame(summary)
    print(df_sum.to_string(index=False))

    # ── チームランキング & 縮小率の確認 ──────────────────────────────
    print()
    print("=" * 65)
    print("  【チームランキング & 階層縮小率】 (全データで学習)")
    print("=" * 65)
    feat_full = feat_df
    y_full    = feat_full["result"].astype(str)
    cols      = ["home_team", "away_team", "home_score", "away_score", "date"]

    best_model = HierarchicalPoissonModel(prior_strength=10, dc_rho=-0.13, time_decay=365)
    best_model.fit(feat_full[cols], y_full)

    print()
    print("  強豪チームTOP10:")
    rank_df = best_model.rank_teams(top_n=10)
    print(rank_df.to_string(index=False))

    print()
    print("  J2/J3 チームの縮小率確認:")
    j2_j3_teams = [
        "Okayama", "Iwata", "Verdy", "Kashiwa Reysol",
        "Albirex Niigata", "Kofu", "Yokohama FC", "Montedio Yamagata",
    ]
    for team in j2_j3_teams:
        p = best_model.team_params(team, verbose=True)

    print()
    print("  縮小率の解釈:")
    print("    k=10 の場合 → 試合数が少ないほどリーグ平均に近づく")
    print("    n=10試合  → 縮小率 50%  (半分はリーグ平均)")
    print("    n=50試合  → 縮小率 17%  (ほぼ実測値)")
    print("    n=200試合 → 縮小率  5%  (実測値をほぼそのまま使用)")
    print()

    # ── 第1622回 予想に適用 ───────────────────────────────────────────
    print("=" * 65)
    print("  【第1622回への適用例】 HierBayes k=10")
    print("=" * 65)

    MATCHES_1622 = [
        ("Yokohama F. Marinos", "Kawasaki Frontale"),
        ("Cerezo Osaka",        "Kyoto"),
        (None,                  "Kashiwa Reysol"),   # 水戸=未知
        ("Gamba Osaka",         "Okayama"),
        ("Nagoya Grampus",      "Avispa Fukuoka"),
        ("Kashima Antlers",     "Urawa Reds"),
        ("Verdy",               None),               # 千葉=未知
        (None,                  None),               # 仙台vs栃木C=未知
        ("Albirex Niigata",     None),               # 今治=未知
        ("Kofu",                None),               # 藤枝=未知
        (None,                  "Yokohama FC"),      # 秋田=未知
        ("Montedio Yamagata",   None),               # 八戸=未知
        ("Omiya Ardija",        "Iwata"),
    ]
    MATCH_LABELS = [
        "横浜FM vs 川崎F", "C大阪 vs 京都", "水戸 vs 柏",
        "G大阪 vs 岡山", "名古屋 vs 福岡", "鹿島 vs 浦和",
        "東京V vs 千葉", "仙台 vs 栃木C", "新潟 vs 今治",
        "甲府 vs 藤枝", "秋田 vs 横浜FC", "山形 vs 八戸",
        "大宮 vs 磐田",
    ]

    print()
    print(f"  {'No':>2}  {'試合':<18}  {'1(H勝)':>7}  {'0(分)':>7}  {'2(A勝)':>7}  {'予想':>8}  縮小率")
    print("  " + "-" * 65)

    for i, ((home_en, away_en), label) in enumerate(zip(MATCHES_1622, MATCH_LABELS), 1):
        home = home_en or "UNKNOWN"
        away = away_en or "UNKNOWN"

        att_h = best_model._get_attack(home)
        def_h = best_model._get_defense(home)
        att_a = best_model._get_attack(away)
        def_a = best_model._get_defense(away)

        lam_h = best_model._mu_home * att_h * def_a
        lam_a = best_model._mu_away * att_a * def_h
        p1, p0, p2 = best_model._score_probs(lam_h, lam_a)

        pred = "1" if p1 == max(p1, p0, p2) else ("0" if p0 == max(p1, p0, p2) else "2")
        pred_label = {"1": "ホーム勝", "0": "引き分け", "2": "アウェイ勝"}[pred]

        n_h = best_model._n_games.get(home, 0)
        n_a = best_model._n_games.get(away, 0)
        k   = best_model.prior_strength
        shrink_avg = ((k / (n_h + k)) + (k / (n_a + k))) / 2

        print(f"  {i:>2}. {label:<18}  {p1:>6.1%}  {p0:>6.1%}  {p2:>6.1%}  "
              f"{pred_label:>8}  {shrink_avg:.0%}")


if __name__ == "__main__":
    main()
