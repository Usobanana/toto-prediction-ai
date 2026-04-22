# -*- coding: utf-8 -*-
"""
toto 第1624回 マルチ予想
=========================
モデル: RandomForest+Optuna (バックテスト 48.15%) + 会場別H2H補正
マルチ最適化: Edge-Aware 優先度付きグリーディー (1000 / 3200 / 6400円プラン)

実行:
  python predict_1624_multi.py
"""
import warnings
warnings.filterwarnings("ignore")

import csv
import pandas as pd
import numpy as np
from pathlib import Path
from src.features.feature_builder import FeatureBuilder, get_feature_columns
from src.models.ml_models import RandomForestModel
from src.strategy.multi_optimizer import MatchPrediction, MultiOptimizer, EDGE_HIGH_THR, EDGE_LOW_THR, PROB_THRESHOLD

HOLD_CNT_ID = 1624

MATCHES = [
    ( 1, "仙台",   "山形"),
    ( 2, "清水",   "名古屋"),
    ( 3, "岡山",   "福岡"),
    ( 4, "浦和",   "横浜FM"),
    ( 5, "川崎F",  "千葉"),
    ( 6, "長崎",   "G大阪"),
    ( 7, "札幌",   "いわき"),
    ( 8, "藤枝",   "大宮"),
    ( 9, "相模原", "湘南"),
    (10, "今治",   "富山"),
    (11, "福島",   "甲府"),
    (12, "宮崎",   "鹿児島"),
    (13, "新潟",   "FC大阪"),
]

TEAM_MAP = {
    "横浜FM": "Yokohama F. Marinos", "川崎F": "Kawasaki Frontale",
    "川崎Ｆ": "Kawasaki Frontale",   "鹿島":  "Kashima Antlers",
    "浦和":  "Urawa Reds",           "C大阪": "Cerezo Osaka",
    "京都":  "Kyoto",                "G大阪": "Gamba Osaka",
    "Ｇ大阪":"Gamba Osaka",          "岡山":  "Okayama",
    "名古屋":"Nagoya Grampus",       "福岡":  "Avispa Fukuoka",
    "大宮":  "Omiya Ardija",         "磐田":  "Iwata",
    "東京V": "Verdy",                "柏":    "Kashiwa Reysol",
    "仙台":  "Vegalta Sendai",       "新潟":  "Albirex Niigata",
    "甲府":  "Kofu",                 "横浜FC":"Yokohama FC",
    "山形":  "Montedio Yamagata",    "FC東京":"FC Tokyo",
    "広島":  "Sanfrecce Hiroshima",  "神戸":  "Vissel Kobe",
    "湘南":  "Shonan Bellmare",      "清水":  "Shimizu S-Pulse",
    "長崎":  "V-Varen Nagasaki",     "札幌":  "Hokkaido Consadole Sapporo",
    # データなし
    "千葉": None, "いわき": None, "藤枝": None, "相模原": None,
    "今治": None, "富山":  None,  "福島": None, "宮崎":  None,
    "鹿児島":None, "FC大阪":None,
}

LABEL = {"1": "ホーム勝", "0": "引き分け", "2": "アウェイ勝"}


# ── データ読み込み・学習 ─────────────────────────────────────────────────────

def load_and_train():
    df = pd.read_csv("data/raw/jleague_results.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    builder = FeatureBuilder(form_window=5)
    feat_df = builder.build(df)

    feature_cols = [c for c in get_feature_columns(include_odds=True) if c in feat_df.columns]
    X = feat_df[feature_cols].fillna(0)
    y = feat_df["result"].astype(str)

    rf = RandomForestModel(include_odds=True)
    rf.fit(X, y)

    # 各チームの最新特徴量
    latest = {}
    for col in ["home_team", "away_team"]:
        for team in feat_df[col].unique():
            if team not in latest:
                latest[team] = feat_df[feat_df[col] == team].iloc[-1]

    # Eloレーティング
    elo = {}
    for _, row in feat_df.iterrows():
        elo[row["home_team"]] = row["home_elo"]
        elo[row["away_team"]] = row["away_elo"]

    return rf, feature_cols, latest, elo


# ── toto公式 H2H データ ─────────────────────────────────────────────────────

def load_toto_h2h(hold_cnt_id: int) -> dict:
    path = Path("data/raw/toto_h2h_summary.csv")
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        df = df[df["hold_cnt_id"] == hold_cnt_id]
        return {(row["home_team"], row["away_team"]): row.to_dict() for _, row in df.iterrows()}
    except Exception:
        return {}


def load_vote_rates(hold_cnt_id: int):
    try:
        vr = pd.read_csv("data/raw/toto_vote_rates.csv", encoding="utf-8-sig")
        sub = vr[vr["hold_cnt_id"] == hold_cnt_id].sort_values("match_no")
        return sub.reset_index(drop=True) if len(sub) >= 13 else None
    except FileNotFoundError:
        return None


# ── 確率計算 ────────────────────────────────────────────────────────────────

def _elo_proba(h_elo, a_elo):
    diff = h_elo - a_elo + 50
    ph = 1 / (1 + 10 ** (-diff / 400))
    pa = 1 / (1 + 10 ** (diff / 400))
    pd_ = max(0.0, 1 - ph - pa) * 0.8
    total = ph + pd_ + pa
    return [ph / total, pd_ / total, pa / total]


def get_proba(no, home_jp, away_jp, rf, feature_cols, latest, elo, toto_h2h):
    home_en = TEAM_MAP.get(home_jp)
    away_en = TEAM_MAP.get(away_jp)
    method = "RF"

    if home_en and away_en and home_en in latest and away_en in latest:
        feat_row = latest[home_en][feature_cols].copy().fillna(0)

        # toto公式H2H上書き
        toto_key = (home_jp, away_jp)
        if toto_key in toto_h2h:
            h2h = toto_h2h[toto_key]
            ht = (h2h.get("h2h_home_win_home") or 0) + (h2h.get("h2h_draw_home") or 0) + (h2h.get("h2h_away_win_home") or 0)
            at = (h2h.get("h2h_home_win_away") or 0) + (h2h.get("h2h_draw_away") or 0) + (h2h.get("h2h_away_win_away") or 0)
            if ht > 0:
                feat_row["h2h_home_venue_win_rate"]      = (h2h.get("h2h_home_win_home") or 0) / ht
                feat_row["h2h_home_venue_draw_rate"]     = (h2h.get("h2h_draw_home") or 0) / ht
                feat_row["h2h_home_venue_away_win_rate"] = (h2h.get("h2h_away_win_home") or 0) / ht
                feat_row["h2h_home_venue_count"]         = ht
            if at > 0:
                feat_row["h2h_away_venue_win_rate"]      = (h2h.get("h2h_home_win_away") or 0) / at
                feat_row["h2h_away_venue_draw_rate"]     = (h2h.get("h2h_draw_away") or 0) / at
                feat_row["h2h_away_venue_away_win_rate"] = (h2h.get("h2h_away_win_away") or 0) / at
                feat_row["h2h_away_venue_count"]         = at
            method = "RF+H2H"

        proba = list(rf.predict_proba(pd.DataFrame([feat_row]))[0])
        has_data = True

    elif home_en or away_en:
        h_elo = elo.get(home_en, 1500) if home_en else 1500
        a_elo = elo.get(away_en, 1500) if away_en else 1500
        proba = _elo_proba(h_elo, a_elo)
        method = "Elo"
        has_data = bool(home_en or away_en)

    else:
        proba = [0.45, 0.27, 0.28]
        method = "基本"
        has_data = False

    return proba, method, has_data


# ── メイン ───────────────────────────────────────────────────────────────────

def main():
    print(f"第{HOLD_CNT_ID}回 マルチ予想を生成中...")

    rf, feature_cols, latest, elo = load_and_train()
    toto_h2h = load_toto_h2h(HOLD_CNT_ID)
    vr_df    = load_vote_rates(HOLD_CNT_ID)
    has_vr   = vr_df is not None

    print(f"  toto公式H2H: {len(toto_h2h)}試合 {'取得済み' if toto_h2h else '(なし)'}")
    if has_vr:
        print(f"  投票率データ: 第{HOLD_CNT_ID}回 ({len(vr_df)}試合)")
    else:
        print(f"  投票率データ: 第{HOLD_CNT_ID}回のデータなし（エッジ分析なし）")
    print()

    # ── 各試合の確率計算 ──────────────────────────────────────────────────
    match_predictions = []
    for no, home_jp, away_jp in MATCHES:
        proba, method, has_data = get_proba(no, home_jp, away_jp, rf, feature_cols, latest, elo, toto_h2h)

        vote_rate = None
        if has_vr:
            row = vr_df[vr_df["match_no"] == no]
            if len(row) == 1:
                r = row.iloc[0]
                vote_rate = [float(r["vote_rate_1"]), float(r["vote_rate_0"]), float(r["vote_rate_2"])]

        mp = MatchPrediction(
            no=no, home=home_jp, away=away_jp,
            proba=proba, vote_rate=vote_rate,
            method=method, has_model_data=has_data,
        )
        match_predictions.append(mp)

    # ── 各試合サマリー表示 ────────────────────────────────────────────────
    print("=" * 90)
    print(f"  第{HOLD_CNT_ID}回 各試合 確率・エッジ")
    print("=" * 90)
    if has_vr:
        print(f"  {'No':>2}  {'試合':<18}  {'P1/P0/P2':>16}  {'VR1/VR0/VR2':>14}  {'bestEdge':>8}  {'判定'}")
    else:
        print(f"  {'No':>2}  {'試合':<18}  {'P1/P0/P2':>16}  {'予想':>8}  {'不確実':>6}  方法")
    print("  " + "-" * 86)

    for mp in match_predictions:
        p_str  = f"{mp.proba[0]:.0%}/{mp.proba[1]:.0%}/{mp.proba[2]:.0%}"
        top1   = mp.sorted_outcomes[0][0]
        pred   = LABEL[top1]

        if has_vr and mp.vote_rate:
            vr     = mp.vote_rate
            vr_str = f"{vr[0]:.0f}/{vr[1]:.0f}/{vr[2]:.0f}%"
            flag   = "[!高エッジ]" if mp.best_edge >= EDGE_HIGH_THR else \
                     "[市場強気]" if mp.min_edge < EDGE_LOW_THR else \
                     "[確率<50%]" if mp.prob_uncertain else "[順当]"
            print(f"  {mp.no:>2}. {mp.home:<8} vs {mp.away:<8}  {p_str:>16}  {vr_str:>14}  {mp.best_edge:>8.2f}  {flag}")
        else:
            unc = "[!]" if mp.prob_uncertain else "[ ]"
            print(f"  {mp.no:>2}. {mp.home:<8} vs {mp.away:<8}  {p_str:>16}  {pred:<8}  {unc}    {mp.method}")

    # ── マルチ最適化 ──────────────────────────────────────────────────────
    budgets = [1000, 3200, 6400]

    print()
    print("=" * 90)
    print(f"  第{HOLD_CNT_ID}回 マルチ予想 最適化 (Edge-Aware)")
    print("=" * 90)

    all_results = []
    for budget in budgets:
        opt    = MultiOptimizer(budget_yen=budget, allow_triple=True)
        result = opt.optimize(match_predictions)
        all_results.append((budget, result))

    print()
    print(f"  {'予算':>8}  {'組合数':>6}  {'コスト':>8}  {'全的中率':>10}  {'平均カバー率':>12}")
    print("  " + "-" * 55)
    for budget, result in all_results:
        print(
            f"  {budget:>8,}円  {result.n_combinations:>6}通り  "
            f"{result.cost_yen:>7,}円  {result.p_all_correct * 100:>9.4f}%  "
            f"{result.avg_coverage:>11.3f}"
        )

    recommended_budget = 3200
    _, best = next((b, r) for b, r in all_results if b == recommended_budget)

    print()
    print(best.summary())

    # ── ダブル/トリプル選定理由 ───────────────────────────────────────────
    print()
    print("  [複数択にした試合と選定理由]")
    print("  " + "-" * 74)
    doubled = [(m, k) for m, k in zip(best.matches, best.selections) if k > 1]
    if doubled:
        for m, k in doubled:
            gain = m.covered_prob(k) - m.covered_prob(k - 1)
            reasons = []
            if m.is_edge_uncertain:
                if m.best_edge >= EDGE_HIGH_THR:
                    reasons.append(f"edge={m.best_edge:.2f}≥{EDGE_HIGH_THR}")
                if m.min_edge < EDGE_LOW_THR:
                    reasons.append(f"min_edge={m.min_edge:.2f}<{EDGE_LOW_THR} (市場強気)")
            if m.prob_uncertain:
                reasons.append(f"top1_prob={m.top1_prob:.0%}<{PROB_THRESHOLD:.0%}")
            if not reasons:
                reasons.append(f"カバー率効率 +{gain:.2f}")
            print(
                f"  試合{m.no:>2} {m.home:<8} vs {m.away:<8}: "
                f"{k}択  カバー率 {m.covered_prob(1):.2f}→{m.covered_prob(k):.2f} "
                f"(+{gain:.2f})  [{' + '.join(reasons)}]"
            )
    else:
        print("  (全試合 1択)")

    # ── 全シナリオ詳細 ────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("  [全シナリオ詳細]")
    for budget, result in all_results:
        print()
        print(result.summary())

    print()
    print("  [注意]")
    print(f"  ・確率はRF+Optuna(61特徴量)。バックテスト正答率48.15%。")
    if toto_h2h:
        print(f"  ・toto公式H2Hデータで仙台/相模原/札幌/福島/藤枝 のH2H特徴量を補正済み。")
    if not has_vr:
        print(f"  ・投票率未取得のためエッジ分析なし。取得後に再実行でエッジ連動選定になります。")
    print(f"  ・edge閾値: 高={EDGE_HIGH_THR} / 低={EDGE_LOW_THR} / 確率={PROB_THRESHOLD:.0%}")
    print()


if __name__ == "__main__":
    main()
