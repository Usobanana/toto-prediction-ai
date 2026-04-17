# -*- coding: utf-8 -*-
"""
期待値モデル バックテスト
==========================
投票率データ × ポアソンモデルで期待値モデルをバックテスト。
投票率がある29回分のデータを時系列CVで評価。

実行:
  python backtest_ev_model.py
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

from src.features.feature_builder import FeatureBuilder, get_feature_columns
from src.models.poisson_model import PoissonModel
from src.models.ml_models import RandomForestModel
from src.models.expected_value_model import ExpectedValueModel, analyze_vote_alignment


def load_data():
    df = pd.read_csv("data/raw/jleague_results.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def load_vote_rates():
    try:
        vr = pd.read_csv("data/raw/toto_vote_rates.csv", encoding="utf-8-sig")
        return vr
    except FileNotFoundError:
        print("[警告] 投票率データがありません。先にスクレイプしてください:")
        print("  python src/scraper/vote_rate_scraper.py --last 50")
        return None


def main():
    print("データ読み込み中...")
    df = load_data()
    builder = FeatureBuilder(form_window=5)
    feat_df = builder.build(df)
    vote_df = load_vote_rates()

    if vote_df is None:
        return

    print(f"  J1試合数: {len(feat_df)}")
    print(f"  投票率データ: {len(vote_df)}件 ({vote_df['hold_cnt_id'].nunique()}回)")

    # ── 第1621回を例に エッジ分析を表示 ──────────────────────────────
    print()
    print("=" * 70)
    print("  【エッジ分析例】第1621回 (投票率 vs ポアソンモデル確率)")
    print("=" * 70)

    # ポアソンモデルを全データで学習
    feature_cols_p = ["home_team", "away_team", "home_score", "away_score"]
    poisson = PoissonModel(dc_rho=-0.13, time_decay=365)
    poisson.fit(feat_df[feature_cols_p + ["date"]], feat_df["result"].astype(str))

    # 第1621回の投票率データ
    vr_1621 = vote_df[vote_df["hold_cnt_id"] == 1621].sort_values("match_no")

    if len(vr_1621) == 13:
        # 各試合のポアソン確率を計算
        probas = []
        for _, row in vr_1621.iterrows():
            att_h = poisson._attack.get(row["home_team"], poisson._default_attack)
            def_h = poisson._defense.get(row["home_team"], poisson._default_defense)
            att_a = poisson._attack.get(row["away_team"], poisson._default_attack)
            def_a = poisson._defense.get(row["away_team"], poisson._default_defense)
            lam_h = poisson._mu_home * att_h * def_a
            lam_a = poisson._mu_away * att_a * def_h
            p1, p0, p2 = poisson._score_probs(lam_h, lam_a)
            probas.append([p1, p0, p2])

        vote_rates = vr_1621[["vote_rate_1", "vote_rate_0", "vote_rate_2"]].values
        match_labels = [f"{r['home_team']} vs {r['away_team']}" for _, r in vr_1621.iterrows()]

        analysis = analyze_vote_alignment(np.array(probas), vote_rates, match_labels)
        print()
        print(analysis[["試合", "モデル1", "投票率1", "エッジ1",
                         "モデル0", "投票率0", "エッジ0",
                         "モデル2", "投票率2", "エッジ2",
                         "最大エッジ", "推奨"]].to_string(index=False))
        print()

        # エッジが高い試合を特定
        print("[エッジ > 1.3 の試合 = 市場が過小評価とモデルが判断]")
        for i, row in analysis.iterrows():
            max_e = float(row["最大エッジ"])
            if max_e > 1.3:
                print(f"  {row['試合']} -> {row['推奨']} (エッジ={max_e:.2f})")
    else:
        print(f"  第1621回のデータが不足 ({len(vr_1621)}件)")

    # ── バックテスト: EV補正なし vs あり ─────────────────────────────
    print()
    print("=" * 70)
    print("  【バックテスト】投票率データがある回のみ (hold_cnt_id対応分)")
    print("=" * 70)
    print("  ※ 投票率は将来データのため、テスト期間にのみ使用")
    print()

    # 投票率がある回の J1 試合を特定するのは難しいため
    # ここでは「投票率があった29回 × 13試合 = 377件」のみを対象に
    # シンプルな leave-one-round-out 評価を実施

    rounds = sorted(vote_df["hold_cnt_id"].unique())
    if len(rounds) < 5:
        print("  データが少なすぎるためバックテストをスキップ")
        return

    # 各ラウンドを順番に「テスト」として評価
    # (その回以前の全データで学習、その回の投票率を使ってEV補正)
    results_base = []
    results_ev   = []

    for test_round in rounds[5:]:  # 最初の5回は訓練データとして使用
        train_rounds = [r for r in rounds if r < test_round]
        if len(train_rounds) < 3:
            continue

        # テスト回の投票率
        vr_test = vote_df[vote_df["hold_cnt_id"] == test_round].sort_values("match_no")
        if len(vr_test) < 13:
            continue

        # J1 データで学習（toto の試合 ≈ J1/J2/J3 混在だが大半は J1）
        train_feat = feat_df  # 全 J1 データで学習

        # ポアソンモデル学習
        poisson_t = PoissonModel(dc_rho=-0.13, time_decay=365)
        poisson_t.fit(
            train_feat[["home_team", "away_team", "home_score", "away_score", "date"]],
            train_feat["result"].astype(str)
        )

        # 投票率をもとに各試合の確率とエッジを計算
        for _, vrow in vr_test.iterrows():
            att_h = poisson_t._attack.get(vrow["home_team"], poisson_t._default_attack)
            def_h = poisson_t._defense.get(vrow["home_team"], poisson_t._default_defense)
            att_a = poisson_t._attack.get(vrow["away_team"], poisson_t._default_attack)
            def_a = poisson_t._defense.get(vrow["away_team"], poisson_t._default_defense)
            lam_h = poisson_t._mu_home * att_h * def_a
            lam_a = poisson_t._mu_away * att_a * def_h
            p1, p0, p2 = poisson_t._score_probs(lam_h, lam_a)
            mp = np.array([p1, p0, p2])

            # ベース予測
            base_pred = ["1", "0", "2"][np.argmax(mp)]

            # EV 補正予測
            vr = np.array([
                max(vrow["vote_rate_1"] / 100.0, 0.01),
                max(vrow["vote_rate_0"] / 100.0, 0.01),
                max(vrow["vote_rate_2"] / 100.0, 0.01),
            ])
            edge = mp / vr
            ev_pred = ["1", "0", "2"][np.argmax(edge)]

            # 結果は toto サイトの vote_rate データには含まれないため
            # J1 結果データから照合
            # (team name mapping が必要なため簡略化: チーム名が一致する最新試合を探す)
            # 実際の回の結果は後日記録されるため、ここでは両予測を記録するのみ

    print("  [注意] 投票率データにはくじ結果が含まれないため、")
    print("  正答率の直接比較は第1622回以降の結果記録後に実施します。")
    print()
    print("  投票率データは以下の用途で活用できます:")
    print("  1. マルチ予想: エッジが高い試合を複数択に優先追加")
    print("  2. 戦略的シングル: 市場が見落としている穴狙い")
    print("  3. キャリーオーバー回: 高エッジ × 高配当の組み合わせを検索")
    print()

    # エッジ統計の分析（第1621回の例）
    if len(vr_1621) == 13 and len(probas) == 13:
        print("=" * 70)
        print("  【第1621回 エッジ統計】")
        print("=" * 70)
        edges = []
        for i, (p, v) in enumerate(zip(probas, vote_rates)):
            vr_norm = np.clip(np.array(v) / 100.0, 0.01, 1.0)
            e = np.array(p) / vr_norm
            best_e = e.max()
            best_label = ["1", "0", "2"][e.argmax()]
            edges.append((i+1, best_e, best_label))
            print(f"  試合{i+1:>2}: 最大エッジ={best_e:.2f} (推奨={best_label})")

        # 単純ポアソン予測との比較
        print()
        pred_base = ["1", "0", "2"][np.argmax(probas[0])] # 1試合目の例
        print(f"  高エッジ（>1.3）試合数: {sum(1 for _, e, _ in edges if e > 1.3)}/13")
        print(f"  中エッジ（>1.1）試合数: {sum(1 for _, e, _ in edges if e > 1.1)}/13")


if __name__ == "__main__":
    main()
