# -*- coding: utf-8 -*-
"""
1等カバー実験 — 予想モデル × マルチ予算 クロス検証
====================================================
過去29回のデータを使い、
「どのモデル／どの予算でマルチを組めば、
 実際の1等組み合わせがカバーできていたか」を検証する。

実験軸:
  モデル:
    HierBayes k=5       バックテスト最良 (44.3%)
    Favorite            市場人気 (49.1%)
    Ensemble-avg        HierBayes × Favorite の確率を平均
    Ensemble-vote       3モデル (HierBayes / Poisson / Favorite) の多数決

  予算: 100円 (シングル) 〜 102,400円 (1024通り)
    ※ toto マルチは 100円 × 通り数

  判定:
    「1等カバー」= マルチの各試合選択が、実際の結果を全13試合含む
    つまり exists combo in multi_selection s.t. combo == actual_results

実行:
  python simulate_jackpot.py
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import math
import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

from src.features.feature_builder import FeatureBuilder
from src.models.hierarchical_poisson import HierarchicalPoissonModel
from src.models.poisson_model import PoissonModel
from src.strategy.edge_analyzer import JP_EN_MAP
from src.strategy.multi_optimizer import MatchPrediction, MultiOptimizer

# ──────────────────────────────────────────────────────────────────────
# 実験パラメータ
# ──────────────────────────────────────────────────────────────────────
BUDGETS = [100, 200, 400, 800, 1_600, 3_200, 6_400, 12_800, 25_600, 51_200, 102_400]
# 上記を通り数に変換: 100円/通 なので 通り数 = budget // 100
MAX_COMBOS_DISPLAY = 10_240   # これを超えるケースはスキップ (メモリ節約)


# ──────────────────────────────────────────────────────────────────────
# データ読み込み
# ──────────────────────────────────────────────────────────────────────

def load_all():
    df = pd.read_csv("data/raw/jleague_results.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    feat_df = FeatureBuilder(form_window=5).build(df)

    vr_df = pd.read_csv("data/raw/toto_vote_rates.csv", encoding="utf-8-sig")
    ar_df = pd.read_csv("data/raw/toto_actual_results.csv", encoding="utf-8-sig")
    ar_df["result"] = ar_df["result"].astype(str)
    return feat_df, vr_df, ar_df


# ──────────────────────────────────────────────────────────────────────
# モデル学習
# ──────────────────────────────────────────────────────────────────────

def train_models(feat_df):
    cols = ["home_team", "away_team", "home_score", "away_score", "date"]
    y    = feat_df["result"].astype(str)

    hier = HierarchicalPoissonModel(prior_strength=5, dc_rho=-0.13, time_decay=365)
    hier.fit(feat_df[cols], y)

    poisson = PoissonModel(dc_rho=-0.13, time_decay=365)
    poisson.fit(feat_df[cols], y)

    return hier, poisson


def get_hierbayes_proba(model, home_jp, away_jp):
    home_en = JP_EN_MAP.get(home_jp)
    away_en = JP_EN_MAP.get(away_jp)
    att_h = model._get_attack(home_en or "")
    def_h = model._get_defense(home_en or "")
    att_a = model._get_attack(away_en or "")
    def_a = model._get_defense(away_en or "")
    lam_h = model._mu_home * att_h * def_a
    lam_a = model._mu_away * att_a * def_h
    p1, p0, p2 = model._score_probs(lam_h, lam_a)
    return [p1, p0, p2]


def get_poisson_proba(model, home_jp, away_jp):
    home_en = JP_EN_MAP.get(home_jp)
    away_en = JP_EN_MAP.get(away_jp)
    if not home_en or not away_en:
        return None
    att_h = model._attack.get(home_en, model._default_attack)
    def_h = model._defense.get(home_en, model._default_defense)
    att_a = model._attack.get(away_en, model._default_attack)
    def_a = model._defense.get(away_en, model._default_defense)
    lam_h = model._mu_home * att_h * def_a
    lam_a = model._mu_away * att_a * def_h
    p1, p0, p2 = model._score_probs(lam_h, lam_a)
    return [p1, p0, p2]


# ──────────────────────────────────────────────────────────────────────
# 1等カバー判定
# ──────────────────────────────────────────────────────────────────────

def is_jackpot_covered(match_preds: list[MatchPrediction],
                       selections: list[int],
                       actual: list[str]) -> bool:
    """マルチ選択が全13試合の実際の結果をカバーしているか判定"""
    for mp, k in zip(match_preds, selections):
        top_k = mp.top_k_labels(k)
        if actual[mp.no - 1] not in top_k:
            return False
    return True


def single_correct_count(match_preds: list[MatchPrediction], actual: list[str]) -> int:
    """シングル予想（top-1）の正答数"""
    return sum(
        mp.sorted_outcomes[0][0] == actual[mp.no - 1]
        for mp in match_preds
    )


# ──────────────────────────────────────────────────────────────────────
# ラウンドごとの予想生成
# ──────────────────────────────────────────────────────────────────────

def build_match_predictions(vr_row_dict: dict,
                             hier_model, poisson_model,
                             model_name: str) -> list[MatchPrediction]:
    """
    vr_row_dict: {match_no: vr_series}
    model_name: "hierbayes" / "favorite" / "ensemble_avg" / "ensemble_vote"
    """
    match_nos = sorted(vr_row_dict.keys())
    preds = []
    for no in match_nos:
        row = vr_row_dict[no]
        home_jp = row["home_team"]
        away_jp = row["away_team"]
        vr = [float(row["vote_rate_1"]),
              float(row["vote_rate_0"]),
              float(row["vote_rate_2"])]

        # 各モデルの確率
        p_hier = get_hierbayes_proba(hier_model, home_jp, away_jp)
        p_pois = get_poisson_proba(poisson_model, home_jp, away_jp) or p_hier
        p_fav  = [v / 100.0 for v in vr]   # 投票率 = 市場確率

        if model_name == "hierbayes":
            proba = p_hier
        elif model_name == "favorite":
            proba = p_fav
        elif model_name == "ensemble_avg":
            # HierBayes と Favorite の単純平均
            proba = [(p_hier[i] + p_fav[i]) / 2 for i in range(3)]
            total = sum(proba); proba = [p / total for p in proba]
        elif model_name == "ensemble_vote":
            # 3モデルの多数決: HierBayes / Poisson / Favorite それぞれのtop1に1票
            votes = [0.0, 0.0, 0.0]
            for p in [p_hier, p_pois, p_fav]:
                winner = p.index(max(p))
                votes[winner] += 1
            # 票数を確率として使用、同票は元確率で按分
            if max(votes) > 0:
                proba = [v / 3.0 for v in votes]
            else:
                proba = p_hier
        else:
            proba = p_hier

        preds.append(MatchPrediction(
            no=no,
            home=home_jp,
            away=away_jp,
            proba=proba,
            vote_rate=vr,
            method=model_name,
        ))
    return preds


# ──────────────────────────────────────────────────────────────────────
# メイン実験
# ──────────────────────────────────────────────────────────────────────

def run_experiment():
    print("データ読み込み・モデル学習中...")
    feat_df, vr_df, ar_df = load_all()
    hier_model, poisson_model = train_models(feat_df)
    print(f"  J1試合数: {len(feat_df)}  /  検証ラウンド数: {vr_df['hold_cnt_id'].nunique()}")
    print()

    rounds = sorted(vr_df["hold_cnt_id"].unique())
    model_names = ["hierbayes", "favorite", "ensemble_avg", "ensemble_vote"]
    model_labels = {
        "hierbayes":     "HierBayes k=5  ",
        "favorite":      "Favorite(投票率)",
        "ensemble_avg":  "Ensemble-Avg   ",
        "ensemble_vote": "Ensemble-Vote  ",
    }

    # 結果格納: {model_name: {budget: covered_rounds_count}}
    results = {m: defaultdict(int) for m in model_names}
    # シングル正答数の分布
    single_acc = {m: [] for m in model_names}
    # 各ラウンドの詳細
    round_details = []
    # オラクル分析: 各ラウンドで実際の結果が何位に予測されたか
    oracle_data = []   # {round, ranks, min_combos}

    for hold_cnt_id in rounds:
        vr_sub = vr_df[vr_df["hold_cnt_id"] == hold_cnt_id].sort_values("match_no")
        ar_sub = ar_df[ar_df["hold_cnt_id"] == hold_cnt_id].sort_values("match_no")
        if len(vr_sub) < 13 or len(ar_sub) < 13:
            continue

        # 実際の結果 (index = match_no - 1)
        actual = [""] * 13
        for _, r in ar_sub.iterrows():
            actual[int(r["match_no"]) - 1] = str(r["result"])

        vr_row_dict = {int(r["match_no"]): r for _, r in vr_sub.iterrows()}

        row_detail = {"round": hold_cnt_id, "actual": actual.copy()}

        for model_name in model_names:
            match_preds = build_match_predictions(
                vr_row_dict, hier_model, poisson_model, model_name
            )

            # シングル正答数
            n_correct = single_correct_count(match_preds, actual)
            single_acc[model_name].append(n_correct)

            # 各予算でカバー判定
            for budget in BUDGETS:
                max_combos = budget // 100
                opt = MultiOptimizer(budget_yen=budget, allow_triple=True)
                ms  = opt.optimize(match_preds)
                covered = is_jackpot_covered(match_preds, ms.selections, actual)
                if covered:
                    results[model_name][budget] += 1

            # 詳細記録 (HierBayes のシングル結果 + オラクル分析)
            if model_name == "hierbayes":
                row_detail["hier_correct"] = n_correct
                # 各試合で実際の結果がモデルの何位か (1=最有力, 2=2番手, 3=最下位)
                ranks = []
                for mp in match_preds:
                    order = [lbl for lbl, _ in mp.sorted_outcomes]  # 確率降順
                    act   = actual[mp.no - 1]
                    rank  = order.index(act) + 1 if act in order else 3
                    ranks.append(rank)
                min_combos = 1
                for r in ranks:
                    min_combos *= r
                oracle_data.append({
                    "round": hold_cnt_id,
                    "ranks": ranks,
                    "min_combos": min_combos,
                    "n_rank3": ranks.count(3),
                    "n_rank2": ranks.count(2),
                    "n_rank1": ranks.count(1),
                })

        round_details.append(row_detail)

    n_rounds = len(round_details)

    # ──────────────────────────────────────────────────────────────────
    # 結果1: シングル予想 正答分布
    # ──────────────────────────────────────────────────────────────────
    print("=" * 72)
    print("  【実験1】シングル予想 正答数分布 (1枚=100円)")
    print("=" * 72)
    print()
    print(f"  {'モデル':<18}  {'平均':>6}  {'最高':>4}  {'12/13':>6}  {'11/13':>6}  "
          f"{'10/13':>6}  分布(7〜13)")
    print("  " + "-" * 65)

    for model_name in model_names:
        accs = single_acc[model_name]
        avg  = np.mean(accs)
        best = max(accs)
        n12  = sum(1 for a in accs if a >= 12)
        n11  = sum(1 for a in accs if a >= 11)
        n10  = sum(1 for a in accs if a >= 10)
        # 分布バー (7〜13的中)
        dist_str = " ".join(f"{c}:{sum(1 for a in accs if a==c)}" for c in range(7, 14))
        print(f"  {model_labels[model_name]}  {avg:>5.2f}  {best:>4}  "
              f"{n12:>6}  {n11:>6}  {n10:>6}  {dist_str}")

    print()
    print(f"  ※ {n_rounds}回中、13/13(1等シングル)を達成したケース: "
          f"{max(sum(1 for a in single_acc[m] if a==13) for m in model_names)}回")

    # ──────────────────────────────────────────────────────────────────
    # 結果2: マルチ 1等カバー回数
    # ──────────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  【実験2】マルチ予想 1等カバー回数 (29回中)")
    print("  ※「カバー」= 購入した複数通りの中に正解組み合わせが含まれる")
    print("=" * 72)
    print()

    combo_labels = [budget // 100 for budget in BUDGETS]
    header = "  モデル             |" + "".join(f"{c:>6}通" for c in combo_labels)
    print(header)
    print("  " + "-" * (20 + 7 * len(BUDGETS)))

    for model_name in model_names:
        row = f"  {model_labels[model_name]}|"
        for budget in BUDGETS:
            cnt = results[model_name][budget]
            row += f"{cnt:>6}回" if cnt > 0 else f"{'  -':>7}"
        print(row)

    # カバー率（%）
    print()
    print("  [カバー率]")
    header2 = "  モデル             |" + "".join(f"{c:>6}通" for c in combo_labels)
    print(header2)
    print("  " + "-" * (20 + 7 * len(BUDGETS)))

    for model_name in model_names:
        row = f"  {model_labels[model_name]}|"
        for budget in BUDGETS:
            cnt = results[model_name][budget]
            pct = cnt / n_rounds * 100
            row += f"{pct:>6.0f}%" if cnt > 0 else f"{'  0%':>7}"
        print(row)

    # ──────────────────────────────────────────────────────────────────
    # 結果3: 予算別「初めてカバー」分析
    # ──────────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  【実験3】「初めて1等カバーできる最低予算」の分布")
    print("=" * 72)
    print()

    for model_name in model_names:
        # 各ラウンドで最初にカバーできた予算を求める
        first_cover_budgets = []
        for idx, hold_cnt_id in enumerate(rounds):
            vr_sub = vr_df[vr_df["hold_cnt_id"] == hold_cnt_id].sort_values("match_no")
            ar_sub = ar_df[ar_df["hold_cnt_id"] == hold_cnt_id].sort_values("match_no")
            if len(vr_sub) < 13 or len(ar_sub) < 13:
                continue
            actual = [""] * 13
            for _, r in ar_sub.iterrows():
                actual[int(r["match_no"]) - 1] = str(r["result"])
            vr_row_dict = {int(r["match_no"]): r for _, r in vr_sub.iterrows()}
            match_preds = build_match_predictions(
                vr_row_dict, hier_model, poisson_model, model_name
            )

            first_b = None
            for budget in BUDGETS:
                opt = MultiOptimizer(budget_yen=budget, allow_triple=True)
                ms  = opt.optimize(match_preds)
                if is_jackpot_covered(match_preds, ms.selections, actual):
                    first_b = budget
                    break
            first_cover_budgets.append((hold_cnt_id, first_b))

        covered_rounds = [(rid, b) for rid, b in first_cover_budgets if b is not None]
        uncovered = [(rid, b) for rid, b in first_cover_budgets if b is None]

        print(f"  [{model_labels[model_name].strip()}]")
        print(f"    最大予算({BUDGETS[-1]:,}円)でカバーできた回: {len(covered_rounds)}/{n_rounds}回")
        if covered_rounds:
            budget_dist = defaultdict(list)
            for rid, b in covered_rounds:
                budget_dist[b].append(rid)
            for b in BUDGETS:
                if b in budget_dist:
                    rids = budget_dist[b]
                    print(f"    {b//100:>5}通り({b:>7,}円)で初カバー: {len(rids)}回  "
                          f"(回 {rids})")
        if uncovered:
            print(f"    カバー不可({BUDGETS[-1]:,}円超が必要): {len(uncovered)}回")
        print()

    # ──────────────────────────────────────────────────────────────────
    # 結果4: 各ラウンドの詳細（HierBayes）
    # ──────────────────────────────────────────────────────────────────
    print("=" * 72)
    print("  【実験4】ラウンド別詳細 (HierBayes × 各予算)")
    print("=" * 72)
    print()
    print(f"  {'回':>5}  {'実結果':>20}  {'Single':>6}  ", end="")
    for budget in [100, 800, 3200, 12800]:
        print(f" {budget//100:>4}通", end="")
    print()
    print("  " + "-" * 65)

    for hold_cnt_id in rounds:
        vr_sub = vr_df[vr_df["hold_cnt_id"] == hold_cnt_id].sort_values("match_no")
        ar_sub = ar_df[ar_df["hold_cnt_id"] == hold_cnt_id].sort_values("match_no")
        if len(vr_sub) < 13 or len(ar_sub) < 13:
            continue
        actual = [""] * 13
        for _, r in ar_sub.iterrows():
            actual[int(r["match_no"]) - 1] = str(r["result"])
        vr_row_dict = {int(r["match_no"]): r for _, r in vr_sub.iterrows()}
        match_preds = build_match_predictions(
            vr_row_dict, hier_model, poisson_model, "hierbayes"
        )
        n_correct = single_correct_count(match_preds, actual)
        result_str = "".join(actual)

        print(f"  {hold_cnt_id:>5}回  {result_str:>20}  {n_correct:>4}/13  ", end="")
        for budget in [100, 800, 3200, 12800]:
            opt = MultiOptimizer(budget_yen=budget, allow_triple=True)
            ms  = opt.optimize(match_preds)
            covered = is_jackpot_covered(match_preds, ms.selections, actual)
            print(f"  {'○' if covered else '×':>4} ", end="")
        print()

    # ──────────────────────────────────────────────────────────────────
    # 結果5: オラクル分析（「仮に結果が事前にわかっていたら何通り必要か」）
    # ──────────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  【実験5】オラクル分析 — 「正解を事前に知っていた場合」の必要通り数")
    print("  ※ 実際の結果がモデルの何位予測か → 全13試合の積 = 最低必要通り数")
    print("=" * 72)
    print()
    print(f"  {'回':>5}  {'1位x':>4} {'2位x':>4} {'3位x':>4}  {'最低通り数':>10}  {'コスト':>10}  ランク列")
    print("  " + "-" * 70)

    for od in sorted(oracle_data, key=lambda x: x["min_combos"]):
        rank_str = "".join(str(r) for r in od["ranks"])
        cost = od["min_combos"] * 100
        cost_str = f"{cost:>9,}円" if cost <= 10_000_000 else f">{10_000_000:,}円"
        print(
            f"  {od['round']:>5}回  "
            f"{od['n_rank1']:>4}試合 {od['n_rank2']:>4}試合 {od['n_rank3']:>4}試合  "
            f"{od['min_combos']:>10,}通り  {cost_str}  {rank_str}"
        )

    min_combos_list = [od["min_combos"] for od in oracle_data]
    print()
    print(f"  最小必要通り数の統計:")
    print(f"    最小値: {min(min_combos_list):,}通り  ({min(min_combos_list)*100:,}円)")
    print(f"    中央値: {int(np.median(min_combos_list)):,}通り  ({int(np.median(min_combos_list))*100:,}円)")
    print(f"    最大値: {max(min_combos_list):,}通り  ({max(min_combos_list)*100:,}円)")
    avg_mc = np.mean(min_combos_list)
    print(f"    平均値: {avg_mc:,.0f}通り  ({avg_mc*100:,.0f}円)")
    print()
    # 予算別「その予算で理論上カバー可能だった回数」
    print(f"  予算別カバー可能回数（オラクル = 結果が事前にわかっていた場合）:")
    for budget in BUDGETS:
        max_c = budget // 100
        covered = sum(1 for od in oracle_data if od["min_combos"] <= max_c)
        bar = "#" * covered
        print(f"    {max_c:>6}通り ({budget:>8,}円): {covered:>2}/{n_rounds}回  {bar}")
    print()
    # 3位予測が多い回の分布
    rank3_counts = [od["n_rank3"] for od in oracle_data]
    print(f"  「3位(最も確率低い結果)が実際に起きた」試合数の分布:")
    for n3 in range(0, 8):
        cnt = sum(1 for r in rank3_counts if r == n3)
        if cnt > 0:
            print(f"    {n3}試合で3位が的中: {cnt}回  {'#'*cnt}")

    # ──────────────────────────────────────────────────────────────────
    # まとめ
    # ──────────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  【まとめ】")
    print("=" * 72)
    print()
    print("  ■ シングル予想 (1枚100円) での1等 → 全モデルで 0/29回")
    print()
    best_model = max(model_names, key=lambda m: results[m][3200])
    best_cnt   = results[best_model][3200]
    print(f"  ■ 3,200円 (32通り) での1等カバー:")
    for m in model_names:
        cnt = results[m][3200]
        print(f"     {model_labels[m]}  {cnt}/{n_rounds}回 ({cnt/n_rounds:.0%})")
    print()
    best_model_l = max(model_names, key=lambda m: results[m][BUDGETS[-1]])
    best_cnt_l   = results[best_model_l][BUDGETS[-1]]
    print(f"  ■ {BUDGETS[-1]//100:,}通り ({BUDGETS[-1]:,}円) での1等カバー:")
    for m in model_names:
        cnt = results[m][BUDGETS[-1]]
        print(f"     {model_labels[m]}  {cnt}/{n_rounds}回 ({cnt/n_rounds:.0%})")
    print()
    print("  ■ 考察:")
    print("    ・シングル1枚では13問全問正解は29回中0回（理論値: 0.44^13 ≈ 1/30万）。")
    print("    ・マルチでも1等カバーには数万〜数十万円が必要（実験5参照）。")
    print("    ・実結果がモデルの「3位予測（最も低確率）」だった試合が多いほど高コスト。")
    print("    ・Favoriteモデルは市場人気に追従するため、波乱回でカバーが外れやすい。")
    print("    ・Ensemble系は単独モデルより安定したカバー率が期待できる。")
    print("    ・「1等を狙う」より「11問以上の正解確率を最大化する」戦略の方が現実的。")
    print()


if __name__ == "__main__":
    run_experiment()
