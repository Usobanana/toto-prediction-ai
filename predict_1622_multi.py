# -*- coding: utf-8 -*-
"""
toto 第1622回 マルチ予想 (HierBayes + Edge-Aware 最適化)
==========================================================

ベースモデル: HierarchicalPoissonModel k=5 (バックテスト最良)
マルチ最適化: エッジ(P/Q)を使った優先度付きグリーディー昇格
  - edge >= 1.5 or edge < 0.7 → エッジ不確実 → 優先でダブル/トリプル
  - Simple確率 < 50%          → 確率不確実  → エッジ信号と組み合わせ

実行:
  python predict_1622_multi.py
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from src.features.feature_builder import FeatureBuilder
from src.models.hierarchical_poisson import HierarchicalPoissonModel
from src.strategy.multi_optimizer import (
    MatchPrediction, MultiOptimizer,
    EDGE_HIGH_THR, EDGE_LOW_THR, PROB_THRESHOLD,
)

HOLD_CNT_ID = 1622

# ── 対戦カード (日本語チーム名 × 試合番号) ──────────────────────────────
MATCHES_1622 = [
    ( 1, "横浜FM",  "川崎F"),
    ( 2, "C大阪",   "京都"),
    ( 3, "水戸",    "柏"),
    ( 4, "G大阪",   "岡山"),
    ( 5, "名古屋",  "福岡"),
    ( 6, "鹿島",    "浦和"),
    ( 7, "東京V",   "千葉"),
    ( 8, "仙台",    "栃木C"),
    ( 9, "新潟",    "今治"),
    (10, "甲府",    "藤枝"),
    (11, "秋田",    "横浜FC"),
    (12, "山形",    "八戸"),
    (13, "大宮",    "磐田"),
]

LABEL = {"1": "ホーム勝", "0": "引き分け", "2": "アウェイ勝"}


# ── データ読み込み ────────────────────────────────────────────────────────

def load_jleague():
    df = pd.read_csv("data/raw/jleague_results.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def load_vote_rates(hold_cnt_id: int):
    """指定回の投票率を読み込む。ない場合は None を返す"""
    try:
        vr = pd.read_csv("data/raw/toto_vote_rates.csv", encoding="utf-8-sig")
        sub = vr[vr["hold_cnt_id"] == hold_cnt_id].sort_values("match_no")
        if len(sub) < 13:
            return None
        return sub.reset_index(drop=True)
    except FileNotFoundError:
        return None


# ── モデル学習 ────────────────────────────────────────────────────────────

def train_hierbayes(df: pd.DataFrame) -> HierarchicalPoissonModel:
    feat_df = FeatureBuilder(form_window=5).build(df)
    cols    = ["home_team", "away_team", "home_score", "away_score", "date"]
    model   = HierarchicalPoissonModel(prior_strength=5, dc_rho=-0.13, time_decay=365)
    model.fit(feat_df[cols], feat_df["result"].astype(str))
    return model


# ── 確率計算 ──────────────────────────────────────────────────────────────

def get_proba(model: HierarchicalPoissonModel, home_jp: str, away_jp: str):
    """
    JP チーム名から確率と has_model_data フラグを返す。
    JP_EN_MAP 経由でモデルの attack/defense を参照する。
    """
    from src.strategy.edge_analyzer import JP_EN_MAP
    home_en = JP_EN_MAP.get(home_jp)
    away_en = JP_EN_MAP.get(away_jp)

    att_h = model._get_attack(home_en or "")
    def_h = model._get_defense(home_en or "")
    att_a = model._get_attack(away_en or "")
    def_a = model._get_defense(away_en or "")

    lam_h = model._mu_home * att_h * def_a
    lam_a = model._mu_away * att_a * def_h
    p1, p0, p2 = model._score_probs(lam_h, lam_a)

    has_data = (home_en in model._attack) or (away_en in model._attack)
    return [p1, p0, p2], has_data


# ── メイン ────────────────────────────────────────────────────────────────

def main():
    print(f"第{HOLD_CNT_ID}回 マルチ予想を生成中...")
    print()

    # 学習
    df    = load_jleague()
    model = train_hierbayes(df)

    # 投票率の読み込み (任意)
    vr_df = load_vote_rates(HOLD_CNT_ID)
    has_vr = vr_df is not None
    if has_vr:
        print(f"  投票率データ: 第{HOLD_CNT_ID}回 ({len(vr_df)}試合) を読み込みました")
    else:
        print(f"  投票率データ: 第{HOLD_CNT_ID}回のデータがありません。エッジ分析なしで最適化します。")
        print(f"    (スクレイプ後に再実行すると投票率連動のマルチ選定になります)")
    print()

    # ── 各試合の確率・投票率・エッジを計算 ───────────────────────────────
    match_predictions = []
    for no, home_jp, away_jp in MATCHES_1622:
        proba, has_data = get_proba(model, home_jp, away_jp)

        # 投票率を取得
        vote_rate = None
        if has_vr:
            row = vr_df[vr_df["match_no"] == no]
            if len(row) == 1:
                r = row.iloc[0]
                vote_rate = [
                    float(r["vote_rate_1"]),
                    float(r["vote_rate_0"]),
                    float(r["vote_rate_2"]),
                ]

        mp = MatchPrediction(
            no=no,
            home=home_jp,
            away=away_jp,
            proba=proba,
            vote_rate=vote_rate,
            method="HierBayes" if has_data else "HierBayes(avg)",
            has_model_data=has_data,
        )
        match_predictions.append(mp)

    # ── 各試合の詳細表示 ───────────────────────────────────────────────
    print("=" * 88)
    print(f"  第{HOLD_CNT_ID}回 各試合 確率・投票率・エッジ")
    print("=" * 88)
    if has_vr:
        print(f"  {'No':>2}  {'試合':<18}  "
              f"{'P1/P0/P2':>16}  {'VR1/VR0/VR2':>14}  "
              f"{'bestEdge':>8}  {'優先度':>6}  {'判定'}")
    else:
        print(f"  {'No':>2}  {'試合':<18}  "
              f"{'P1/P0/P2':>16}  {'Simple':>6}  {'確率不確実':>8}  モデル")
    print("  " + "-" * 84)

    for mp in match_predictions:
        p_str = f"{mp.proba[0]:.0%}/{mp.proba[1]:.0%}/{mp.proba[2]:.0%}"
        top1  = mp.sorted_outcomes[0]
        pred_lbl = LABEL[top1[0]]

        if has_vr and mp.vote_rate:
            vr = mp.vote_rate
            vr_str  = f"{vr[0]:.0f}/{vr[1]:.0f}/{vr[2]:.0f}%"
            e       = mp.edge
            e_str   = f"{mp.best_edge:.2f}"
            pri_str = f"P{mp.priority}"
            flag    = ""
            if mp.best_edge >= EDGE_HIGH_THR:
                flag = "[!高エッジ]"
            elif mp.min_edge < EDGE_LOW_THR:
                flag = "[市場強気]"
            elif mp.prob_uncertain:
                flag = "[確率<50%]"
            else:
                flag = "[順当]"
            print(
                f"  {mp.no:>2}. {mp.home:<8} vs {mp.away:<8}  "
                f"{p_str:>16}  {vr_str:>14}  "
                f"{e_str:>8}  {pri_str:>6}  {flag}"
            )
        else:
            uncertain_flag = "[!]" if mp.prob_uncertain else "[ ]"
            data_flag      = "" if mp.has_model_data else " (avg)"
            print(
                f"  {mp.no:>2}. {mp.home:<8} vs {mp.away:<8}  "
                f"{p_str:>16}  {pred_lbl:<8}  {uncertain_flag}{data_flag}"
            )

    # ── マルチ最適化 (複数予算シナリオ) ───────────────────────────────
    budgets = [1000, 3200, 6400]

    print()
    print("=" * 88)
    print(f"  第{HOLD_CNT_ID}回 マルチ予想 最適化 (Edge-Aware)")
    print("=" * 88)

    all_results = []
    for budget in budgets:
        opt    = MultiOptimizer(budget_yen=budget, allow_triple=True)
        result = opt.optimize(match_predictions)
        all_results.append((budget, result))

    # 各シナリオのサマリー比較
    print()
    print(f"  {'予算':>8}  {'組合数':>6}  {'コスト':>8}  {'全的中率':>10}  {'平均カバー率':>12}")
    print("  " + "-" * 55)
    for budget, result in all_results:
        print(
            f"  {budget:>8,}円  {result.n_combinations:>6}通り  "
            f"{result.cost_yen:>7,}円  {result.p_all_correct * 100:>9.4f}%  "
            f"{result.avg_coverage:>11.3f}"
        )

    # 推奨プランの詳細 (中間予算)
    recommended_budget = 3200
    _, best = next((b, r) for b, r in all_results if b == recommended_budget)

    print()
    print(best.summary())

    # ── ダブル/トリプル試合の理由説明 ────────────────────────────────
    print()
    print("  [複数択にした試合と選定理由]")
    print("  " + "-" * 72)
    doubled = [(m, k) for m, k in zip(best.matches, best.selections) if k > 1]
    if doubled:
        for m, k in doubled:
            gain    = m.covered_prob(k) - m.covered_prob(k - 1)
            reasons = []
            if m.is_edge_uncertain:
                if m.best_edge >= EDGE_HIGH_THR:
                    reasons.append(f"edge={m.best_edge:.2f}≥{EDGE_HIGH_THR} (モデル↑市場)")
                if m.min_edge < EDGE_LOW_THR:
                    reasons.append(f"min_edge={m.min_edge:.2f}<{EDGE_LOW_THR} (市場強気)")
            if m.prob_uncertain:
                reasons.append(f"top1_prob={m.top1_prob:.0%}<{PROB_THRESHOLD:.0%}")
            if not reasons:
                reasons.append(f"効率(カバー率+{gain:.2f})")
            reason_str = " + ".join(reasons)
            print(
                f"  試合{m.no:>2} {m.home:<8} vs {m.away:<8}: "
                f"{k}択  カバー率 {m.covered_prob(1):.2f}→{m.covered_prob(k):.2f} "
                f"(+{gain:.2f})  [{reason_str}]"
            )
    else:
        print("  (全試合 1択)")

    # ── 全シナリオ詳細 ────────────────────────────────────────────────
    print()
    print("=" * 88)
    print("  [全シナリオ詳細]")
    for budget, result in all_results:
        print()
        print(result.summary())

    print()
    print("  [注意]")
    print(f"  ・確率はHierarchicalPoissonModel(k=5)。バックテスト正答率44.3%。")
    print(f"  ・edge閾値: 高={EDGE_HIGH_THR} / 低={EDGE_LOW_THR} / 確率={PROB_THRESHOLD:.0%}")
    if not has_vr:
        print(f"  ・投票率未取得のためエッジ分析なし。取得後に再実行してください。")
    print()


if __name__ == "__main__":
    main()
