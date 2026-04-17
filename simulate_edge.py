# -*- coding: utf-8 -*-
"""
エッジ戦略 収支シミュレーション
================================
HierBayes k=5 x 投票率エッジ を使った過去29回分の収支シミュレーション。

比較戦略:
  Simple       : argmax(model_prob)              モデル確率最大を選ぶ
  Edge-All     : argmax(model_prob/vote_rate)    常にエッジ最大を選ぶ
  Edge-Smart   : モデルデータあり -> argmax(edge), なし -> argmax(model_prob)
  Favorite     : argmax(vote_rate)               市場人気（投票率最大）を選ぶ

toto シングル（1枚=100円）を想定したシミュレーション:
  近年の平均的な等級別期待額をベースにした近似値。

実行:
  python simulate_edge.py
"""

import warnings
warnings.filterwarnings("ignore")
import io, sys

# Windows コンソール向け UTF-8 出力
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from collections import defaultdict

from src.features.feature_builder import FeatureBuilder
from src.models.hierarchical_poisson import HierarchicalPoissonModel
from src.strategy.edge_analyzer import EdgeAnalyzer

# ──────────────────────────────────────────────────────────────────────
# toto シングル 近似賞金テーブル（1枚=100円）
# ──────────────────────────────────────────────────────────────────────
PRIZE_TABLE = {
    13: 2_000_000,   # 1等: 全的中   ≈ 200万円
    12:   100_000,   # 2等: 12的中   ≈ 10万円
    11:     8_000,   # 3等: 11的中   ≈ 8,000円
    # 10未満は外れ扱い
}
TICKET_COST = 100


def load_jleague():
    df = pd.read_csv("data/raw/jleague_results.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    builder = FeatureBuilder(form_window=5)
    return builder.build(df)


def load_vote_rates():
    return pd.read_csv("data/raw/toto_vote_rates.csv", encoding="utf-8-sig")


def load_actual_results():
    ar = pd.read_csv("data/raw/toto_actual_results.csv", encoding="utf-8-sig")
    ar["result"] = ar["result"].astype(str)
    return ar


def train_model(feat_df):
    cols = ["home_team", "away_team", "home_score", "away_score", "date"]
    model = HierarchicalPoissonModel(prior_strength=5, dc_rho=-0.13, time_decay=365)
    model.fit(feat_df[cols], feat_df["result"].astype(str))
    return model


def prize(n_correct):
    for threshold in sorted(PRIZE_TABLE.keys(), reverse=True):
        if n_correct >= threshold:
            return PRIZE_TABLE[threshold]
    return 0


def bar_graph(ratio, width=20):
    filled = int(ratio * width)
    return "#" * filled + "-" * (width - filled)


# ──────────────────────────────────────────────────────────────────────
def simulate():
    print("データ読み込み中...")
    feat_df   = load_jleague()
    vote_df   = load_vote_rates()
    result_df = load_actual_results()

    print(f"  J1試合数: {len(feat_df)}")
    print(f"  投票率データ: {vote_df['hold_cnt_id'].nunique()}回 / {len(vote_df)}件")
    print(f"  実績データ:   {result_df['hold_cnt_id'].nunique()}回 / {len(result_df)}件")

    print("\nHierBayes k=5 モデルを全データで学習中...")
    model    = train_model(feat_df)
    analyzer = EdgeAnalyzer(model)

    rounds = sorted(vote_df["hold_cnt_id"].unique())

    # 集計変数
    strategy_names = ["simple", "edge_all", "edge_smart", "favorite"]
    stats = {k: {"correct": 0, "total": 0, "pnl": 0, "n_rounds": 0, "round_pnl": []}
             for k in strategy_names}

    # エッジバケット別: モデルデータあり試合のみ
    edge_bucket_stats = defaultdict(lambda: {"correct_edge": 0, "correct_simple": 0, "total": 0})

    round_rows = []

    print()
    print("=" * 85)
    print("  ラウンド別シミュレーション")
    print("=" * 85)
    hdr = (f"  {'回':>5}  {'Simple':>7}  {'Edge-A':>7}  {'Edge-S':>7}  {'Fav':>7}  "
           f"{'Simple損益':>10}  {'EdgeS損益':>10}  勝負件数")
    print(hdr)
    print("  " + "-" * 80)

    for hold_cnt_id in rounds:
        vr = vote_df[vote_df["hold_cnt_id"] == hold_cnt_id].sort_values("match_no")
        ar = result_df[result_df["hold_cnt_id"] == hold_cnt_id]
        if len(vr) < 13 or len(ar) < 13:
            continue

        report = analyzer.analyze_round(hold_cnt_id, vote_df, result_df)
        if not report.matches:
            continue

        correct = {k: 0 for k in strategy_names}
        n_data_matches = 0

        for m in report.matches:
            if not m.actual_result:
                continue

            # 各戦略の予測
            pred_simple = m.pred_simple
            pred_edge_a = m.pred_edge  # 常にエッジ最大
            # Edge-Smart: モデルデータがある試合のみエッジ、なければ simple
            pred_edge_s = m.pred_edge if m.has_model_data else m.pred_simple
            pred_fav    = ["1", "0", "2"][np.argmax([m.vote_rate_1,
                                                      m.vote_rate_0,
                                                      m.vote_rate_2])]

            correct["simple"]     += (pred_simple == m.actual_result)
            correct["edge_all"]   += (pred_edge_a == m.actual_result)
            correct["edge_smart"] += (pred_edge_s == m.actual_result)
            correct["favorite"]   += (pred_fav    == m.actual_result)

            # エッジバケット（モデルデータあり試合のみ）
            if m.has_model_data:
                n_data_matches += 1
                for bucket, lo, hi in [
                    ("edge<1.0",    0.0, 1.0),
                    ("1.0<=e<1.2",  1.0, 1.2),
                    ("1.2<=e<1.5",  1.2, 1.5),
                    ("e>=1.5",      1.5, 9.9),
                ]:
                    if lo <= m.best_edge < hi:
                        edge_bucket_stats[bucket]["total"] += 1
                        edge_bucket_stats[bucket]["correct_edge"]   += (pred_edge_a == m.actual_result)
                        edge_bucket_stats[bucket]["correct_simple"] += (pred_simple == m.actual_result)

        for k in strategy_names:
            pnl = prize(correct[k]) - TICKET_COST
            stats[k]["correct"]   += correct[k]
            stats[k]["total"]     += 13
            stats[k]["pnl"]       += pnl
            stats[k]["n_rounds"]  += 1
            stats[k]["round_pnl"].append(pnl)

        n_upset = len(report.upset_matches)
        print(
            f"  {hold_cnt_id:>5}回  "
            f"{correct['simple']:>2}/13({correct['simple']/13:.0%})  "
            f"{correct['edge_all']:>2}/13({correct['edge_all']/13:.0%})  "
            f"{correct['edge_smart']:>2}/13({correct['edge_smart']/13:.0%})  "
            f"{correct['favorite']:>2}/13({correct['favorite']/13:.0%})  "
            f"{prize(correct['simple'])-TICKET_COST:>+10,}  "
            f"{prize(correct['edge_smart'])-TICKET_COST:>+10,}  "
            f"[{n_upset}/{n_data_matches}]"
        )

        round_rows.append({
            "回": hold_cnt_id,
            "Simple": correct["simple"],
            "Edge-All": correct["edge_all"],
            "Edge-Smart": correct["edge_smart"],
            "Favorite": correct["favorite"],
            "S-損益": prize(correct["simple"]) - TICKET_COST,
            "ES-損益": prize(correct["edge_smart"]) - TICKET_COST,
            "勝負件数": n_upset,
            "データあり": n_data_matches,
        })

    # ──────────────────────────────────────────────────────────────────
    n_rounds = stats["simple"]["n_rounds"]
    print()
    print("=" * 80)
    print(f"  【総合サマリ】 {n_rounds}回分 × 13試合 × 100円/回")
    print("=" * 80)
    print()
    print(f"  {'戦略':<12}  {'正答率':>8}  {'累計損益':>10}  {'avg損益/回':>12}  "
          f"{'最高':>9}  {'最低':>9}")
    print("  " + "-" * 68)

    for k, label in [
        ("simple",     "Simple      "),
        ("edge_all",   "Edge-All    "),
        ("edge_smart", "Edge-Smart  "),
        ("favorite",   "Favorite    "),
    ]:
        s = stats[k]
        acc      = s["correct"] / s["total"] if s["total"] > 0 else 0
        total_pnl = s["pnl"]
        avg_pnl  = total_pnl / n_rounds if n_rounds > 0 else 0
        max_pnl  = max(s["round_pnl"])
        min_pnl  = min(s["round_pnl"])
        print(f"  {label:<12}  {acc:>7.2%}  {total_pnl:>+10,}  {avg_pnl:>+11,.0f}  "
              f"{max_pnl:>+9,}  {min_pnl:>+9,}")

    print()
    total_cost = n_rounds * TICKET_COST
    print(f"  ※ 合計投資額 = {total_cost:,}円 ({n_rounds}回 x {TICKET_COST}円)")

    # ──────────────────────────────────────────────────────────────────
    # エッジバケット分析（モデルデータあり試合のみ）
    # ──────────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("  【エッジバケット別 正答率】 モデルデータあり試合のみ")
    print("  ※ Edge = argmax(P/Q) の予測 vs Simple = argmax(P) の予測")
    print("=" * 80)
    print()
    print(f"  {'エッジ帯':<14}  {'試合数':>6}  {'Edge的中':>8}  {'Edge正答率':>10}  "
          f"{'Simp的中':>8}  {'Simple正答率':>12}")
    print("  " + "-" * 70)

    for bucket in ["edge<1.0", "1.0<=e<1.2", "1.2<=e<1.5", "e>=1.5"]:
        b = edge_bucket_stats[bucket]
        n = b["total"]
        if n == 0:
            continue
        e_acc = b["correct_edge"] / n
        s_acc = b["correct_simple"] / n
        e_bar = bar_graph(e_acc)
        print(f"  {bucket:<14}  {n:>6}  {b['correct_edge']:>8}  "
              f"{e_acc:>9.1%}  {b['correct_simple']:>8}  {s_acc:>11.1%}  {e_bar}")

    # ──────────────────────────────────────────────────────────────────
    # カテゴリ別 正答率
    # ──────────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("  【試合カテゴリ別 正答率】 Edge-Smart 戦略")
    print("=" * 80)
    print()

    cat_stats = defaultdict(lambda: {"es": 0, "sim": 0, "total": 0})
    for hold_cnt_id in rounds:
        vr_c = vote_df[vote_df["hold_cnt_id"] == hold_cnt_id]
        ar_c = result_df[result_df["hold_cnt_id"] == hold_cnt_id]
        if len(vr_c) < 13 or len(ar_c) < 13:
            continue
        report = analyzer.analyze_round(hold_cnt_id, vote_df, result_df)
        for m in report.matches:
            if not m.actual_result:
                continue
            pred_es  = m.pred_edge if m.has_model_data else m.pred_simple
            pred_sim = m.pred_simple
            cat_stats[m.category]["total"] += 1
            cat_stats[m.category]["es"]    += (pred_es  == m.actual_result)
            cat_stats[m.category]["sim"]   += (pred_sim == m.actual_result)

    print(f"  {'カテゴリ':<16}  {'試合数':>6}  {'EdgeS的中':>9}  {'EdgeS正答率':>11}  "
          f"{'Simple的中':>10}  {'Simple正答率':>12}")
    print("  " + "-" * 72)

    for cat, label in [
        ("勝負レース", "勝負(edge>=1.2) "),
        ("中立",       "中立(1.05<e<1.2)"),
        ("順当レース", "順当(edge<=1.05)"),
    ]:
        c = cat_stats[cat]
        n = c["total"]
        if n == 0:
            continue
        es_acc  = c["es"]  / n
        sim_acc = c["sim"] / n
        print(f"  {label:<16}  {n:>6}  {c['es']:>9}  {es_acc:>10.1%}  "
              f"{c['sim']:>10}  {sim_acc:>11.1%}")

    # ──────────────────────────────────────────────────────────────────
    # ランキング TOP/WORST
    # ──────────────────────────────────────────────────────────────────
    if round_rows:
        rdf = pd.DataFrame(round_rows)
        print()
        print("=" * 80)
        print("  【Edge-Smart 戦略 ラウンド別 TOP5 / WORST5】")
        print("=" * 80)
        print()
        sorted_df = rdf.sort_values("ES-損益", ascending=False)
        print("  [好成績 TOP5]")
        for _, r in sorted_df.head(5).iterrows():
            print(f"    第{int(r['回']):>4}回  "
                  f"EdgeS={int(r['Edge-Smart'])}/13  Simple={int(r['Simple'])}/13  "
                  f"損益:{int(r['ES-損益']):>+9,}円  勝負{int(r['勝負件数'])}件")
        print()
        print("  [低成績 WORST5]")
        for _, r in sorted_df.tail(5).iterrows():
            print(f"    第{int(r['回']):>4}回  "
                  f"EdgeS={int(r['Edge-Smart'])}/13  Simple={int(r['Simple'])}/13  "
                  f"損益:{int(r['ES-損益']):>+9,}円  勝負{int(r['勝負件数'])}件")

        # Edge-Smart vs Simple: 差が大きい回
        rdf["diff"] = rdf["Edge-Smart"] - rdf["Simple"]
        gained = rdf[rdf["diff"] > 0].sort_values("diff", ascending=False)
        lost   = rdf[rdf["diff"] < 0].sort_values("diff")

        print()
        print("=" * 80)
        print("  【Edge-Smart が Simple を上回った回 / 下回った回】")
        print("=" * 80)
        print()
        if len(gained) > 0:
            print(f"  Edge-Smart > Simple: {len(gained)}回")
            for _, r in gained.head(5).iterrows():
                print(f"    第{int(r['回']):>4}回  "
                      f"EdgeS={int(r['Edge-Smart'])}/13  Simple={int(r['Simple'])}/13  "
                      f"差={int(r['diff']):+d}")
        if len(lost) > 0:
            print(f"  Edge-Smart < Simple: {len(lost)}回")
            for _, r in lost.head(5).iterrows():
                print(f"    第{int(r['回']):>4}回  "
                      f"EdgeS={int(r['Edge-Smart'])}/13  Simple={int(r['Simple'])}/13  "
                      f"差={int(r['diff']):+d}")

    # ──────────────────────────────────────────────────────────────────
    # 解釈メモ
    # ──────────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("  【解釈メモ】")
    print("=" * 80)
    print()
    print("  ・toto シングル 1枚=100円。11/13以上的中で賞金発生。")
    print("  ・Edge-All: 常にエッジ最大を選ぶ。未知チームを含む試合で精度が落ちる。")
    print("  ・Edge-Smart: モデルデータがある試合のみエッジ補正を適用。")
    print("    未知チーム（J3等）はモデル確率(Simple)にフォールバック。")
    print("  ・Favorite: 市場投票率最大を選ぶ。市場の集合知を使う基準線。")
    print("  ・勝負レース(edge>=1.2): モデルが市場より高く評価している結果。")
    print("    市場が見落としている「穴」を狙う戦略。的中率は低いが")
    print("    長期的な期待値改善を目指す指標として活用する。")
    print()

    # ──────────────────────────────────────────────────────────────────
    # 最終ラウンド詳細レポート
    # ──────────────────────────────────────────────────────────────────
    last_round = max(rounds)
    vr_last = vote_df[vote_df["hold_cnt_id"] == last_round]
    ar_last = result_df[result_df["hold_cnt_id"] == last_round]
    if len(vr_last) >= 13 and len(ar_last) >= 13:
        print()
        print("=" * 80)
        print(f"  【第{last_round}回 詳細エッジレポート】")
        print("=" * 80)
        report_last = analyzer.analyze_round(last_round, vote_df, result_df)
        analyzer.print_round_report(report_last)


if __name__ == "__main__":
    simulate()
