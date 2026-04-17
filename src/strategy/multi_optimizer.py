# -*- coding: utf-8 -*-
"""
toto マルチ購入 最適化モジュール (Edge-Aware 版)
================================================

選択ロジック:
  1. ベース: 全試合 Simple (argmax P) を本命とする (1択)
  2. マルチ枠選定:
       - エッジ高 (best_edge >= EDGE_HIGH_THR) : モデルが市場より大幅に高く評価 → 不確実
       - エッジ低 (min_edge  <  EDGE_LOW_THR)  : 市場がある結果を過小評価      → 不確実
       ⇒ 上記いずれかに該当する試合を「エッジ不確実」とマーク
  3. アップグレード優先度:
       Priority 3 = エッジ不確実 かつ Simple確率 < PROB_THRESHOLD (最優先)
       Priority 2 = エッジ不確実 (確率は高め)
       Priority 1 = Simple確率  < PROB_THRESHOLD のみ (エッジ情報なし or 確率的に不安定)
       Priority 0 = それ以外 (順当試合 / 最後に効率で昇格)
  4. 予算配分:
       各優先グループ内を efficiency (log gain / log cost) でソートしながら
       予算上限まで貪欲に 1択→2択→3択 へ昇格させる。

コスト制約:
  100円 × ∏(各試合の選択数) ≤ 予算
"""

import math
from dataclasses import dataclass, field
from typing import Optional

# ── エッジ閾値 ─────────────────────────────────────────────────────────
EDGE_HIGH_THR  = 1.5    # P/Q >= 1.5 → 市場が過小評価 (エッジ高)
EDGE_LOW_THR   = 0.7    # P/Q <  0.7 → モデルが過小評価 (市場強気)
PROB_THRESHOLD = 0.50   # Simple の top1 確率がこれを下回ると確率不確実


@dataclass
class MatchPrediction:
    """1試合分の予測情報"""
    no:         int
    home:       str
    away:       str
    proba:      list[float]              # [P(1), P(0), P(2)]  ※合計 ≈ 1.0
    vote_rate:  Optional[list[float]] = None  # [VR1, VR0, VR2]  0–100 形式
    method:     str = "HierBayes"
    has_model_data: bool = True

    # ── 基本プロパティ ─────────────────────────────────────────────────

    @property
    def sorted_outcomes(self) -> list[tuple[str, float]]:
        """確率降順 (結果ラベル, 確率) リスト"""
        pairs = [("1", self.proba[0]), ("0", self.proba[1]), ("2", self.proba[2])]
        return sorted(pairs, key=lambda x: x[1], reverse=True)

    def covered_prob(self, k: int) -> float:
        """上位 k 択を選んだ場合のカバー確率"""
        return sum(p for _, p in self.sorted_outcomes[:k])

    def top_k_labels(self, k: int) -> list[str]:
        """上位 k 択のラベルリスト"""
        return [lbl for lbl, _ in self.sorted_outcomes[:k]]

    @property
    def simple_pred(self) -> str:
        """Simple 予測ラベル (argmax P)"""
        return self.sorted_outcomes[0][0]

    @property
    def top1_prob(self) -> float:
        return self.sorted_outcomes[0][1]

    # ── エッジ関連 ─────────────────────────────────────────────────────

    @property
    def edge(self) -> list[float]:
        """[edge_1, edge_0, edge_2]  = P / (VR/100)。投票率なし時は [1,1,1]"""
        if self.vote_rate is None:
            return [1.0, 1.0, 1.0]
        result = []
        for p, vr in zip(self.proba, self.vote_rate):
            q = max(vr / 100.0, 0.01)
            result.append(p / q)
        return result

    @property
    def best_edge(self) -> float:
        return max(self.edge)

    @property
    def min_edge(self) -> float:
        return min(self.edge)

    @property
    def is_edge_uncertain(self) -> bool:
        """エッジ不確実フラグ: 投票率があり、かつ高/低エッジ条件を満たす"""
        if self.vote_rate is None:
            return False
        return self.best_edge >= EDGE_HIGH_THR or self.min_edge < EDGE_LOW_THR

    @property
    def prob_uncertain(self) -> bool:
        """確率不確実フラグ: Simple の top1 確率が閾値未満"""
        return self.top1_prob < PROB_THRESHOLD

    @property
    def priority(self) -> int:
        """
        アップグレード優先度 (高いほど先にマルチ枠割り当て)
        3: エッジ不確実 + 確率不確実   → 最優先でダブル/トリプル
        2: エッジ不確実のみ            → 次点
        1: 確率不確実のみ              → エッジ情報なし/確率低め
        0: 順当試合                    → 最後に効率で判断
        """
        if self.is_edge_uncertain and self.prob_uncertain:
            return 3
        if self.is_edge_uncertain:
            return 2
        if self.prob_uncertain:
            return 1
        return 0

    def edge_label(self) -> str:
        """表示用エッジ情報文字列"""
        if self.vote_rate is None:
            return "(投票率なし)"
        e = self.edge
        best_i = e.index(max(e))
        lbl = ["1", "0", "2"][best_i]
        flag = ""
        if self.best_edge >= EDGE_HIGH_THR:
            flag = " [!高エッジ]"
        elif self.min_edge < EDGE_LOW_THR:
            flag = " [市場強気]"
        return f"e={self.best_edge:.2f}→{lbl}{flag}"


@dataclass
class MultiSelection:
    """マルチ購入の最適化結果"""
    matches:     list[MatchPrediction]
    selections:  list[int]       # 各試合の選択数 (1/2/3)
    budget_yen:  int

    @property
    def n_combinations(self) -> int:
        r = 1
        for k in self.selections:
            r *= k
        return r

    @property
    def cost_yen(self) -> int:
        return self.n_combinations * 100

    @property
    def p_all_correct(self) -> float:
        p = 1.0
        for m, k in zip(self.matches, self.selections):
            p *= m.covered_prob(k)
        return p

    @property
    def avg_coverage(self) -> float:
        covers = [m.covered_prob(k) for m, k in zip(self.matches, self.selections)]
        return sum(covers) / len(covers)

    def summary(self) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("  toto マルチ予想 最適化結果 (Edge-Aware)")
        lines.append("=" * 80)
        lines.append(
            f"  予算: {self.budget_yen:,}円  |  "
            f"使用: {self.cost_yen:,}円  |  "
            f"組み合わせ: {self.n_combinations}通り"
        )
        lines.append(
            f"  全的中確率(理論値): {self.p_all_correct * 100:.4f}%  |  "
            f"平均カバー率: {self.avg_coverage:.3f}"
        )
        lines.append("-" * 80)

        pri_labels = {3: "[3:最優先]", 2: "[2:エッジ]", 1: "[1:確率]", 0: "[0:順当]"}
        lines.append(
            f"  {'No':>2}  {'試合':<18}  {'択':>2}  {'選択':<9}  "
            f"{'カバー率':>6}  {'優先度':>9}  エッジ情報"
        )
        lines.append("-" * 80)

        for m, k in zip(self.matches, self.selections):
            labels    = m.top_k_labels(k)
            cov       = m.covered_prob(k)
            star      = " *" if k > 1 else "  "
            label_str = "/".join(labels)
            pri_str   = pri_labels[m.priority]
            lines.append(
                f"  {m.no:>2}. {m.home:<8} vs {m.away:<8}  "
                f"{k}択  [{label_str:<5}]{star}  "
                f"{cov:.3f}  {pri_str}  {m.edge_label()}"
            )

        lines.append("-" * 80)
        # 全組み合わせ
        combos = self._enumerate_combinations()
        lines.append(f"  全組み合わせ ({len(combos)}通り):")
        for i, combo in enumerate(combos, 1):
            lines.append(f"    [{i:>2}] {' '.join(combo)}")
        lines.append("=" * 80)
        lines.append("  * = 複数択 (マルチ) / 優先度説明:")
        lines.append("    3=エッジ不確実+確率不確実  2=エッジ不確実  1=確率低め  0=順当")
        return "\n".join(lines)

    def _enumerate_combinations(self) -> list[list[str]]:
        combos: list[list[str]] = [[]]
        for m, k in zip(self.matches, self.selections):
            labels = m.top_k_labels(k)
            combos = [c + [l] for c in combos for l in labels]
        return combos


class MultiOptimizer:
    """
    Edge-Aware グリーディー昇格法によるマルチ購入最適化

    Parameters
    ----------
    budget_yen : int
        購入予算 (円)
    allow_triple : bool
        3択 (トリプル) を許可するか
    """

    def __init__(self, budget_yen: int = 5000, allow_triple: bool = True):
        self.budget_yen      = budget_yen
        self.allow_triple    = allow_triple
        self.max_combinations = budget_yen // 100

    def optimize(self, matches: list[MatchPrediction]) -> MultiSelection:
        """
        Edge-Aware 優先度付きグリーディー昇格で最適なマルチ選択を返す。

        アルゴリズム:
          1. 全試合を 1択 で初期化 (Simple)
          2. 未アップグレード候補を (priority DESC, efficiency DESC) でソート
          3. 予算上限まで順にアップグレード (1→2, 2→3)
          4. 効率が同程度なら priority が高い試合を優先
        """
        n          = len(matches)
        selections = [1] * n
        cur_cost   = 1   # 組み合わせ数 (コスト = cur_cost * 100)

        while True:
            best = self._best_upgrade(matches, selections, cur_cost)
            if best is None:
                break
            idx, new_k = best
            old_k        = selections[idx]
            selections[idx] = new_k
            cur_cost     = cur_cost * new_k // old_k

        return MultiSelection(
            matches=matches,
            selections=selections,
            budget_yen=self.budget_yen,
        )

    def _best_upgrade(
        self,
        matches: list[MatchPrediction],
        selections: list[int],
        cur_cost: int,
    ) -> Optional[tuple[int, int]]:
        """
        次にアップグレードすべき (試合インデックス, 新しい択数) を返す。
        優先度が高いグループ内で efficiency 最大のものを選ぶ。
        """
        # 各試合の昇格候補を収集: (priority, efficiency, idx, new_k)
        candidates = []
        for i, m in enumerate(matches):
            cur_k  = selections[i]
            next_k = cur_k + 1
            if next_k > 3:
                continue
            if not self.allow_triple and next_k > 2:
                continue

            new_cost = cur_cost * next_k // cur_k
            if new_cost > self.max_combinations:
                continue

            p_before = m.covered_prob(cur_k)
            p_after  = m.covered_prob(next_k)
            if p_before <= 0 or p_after <= p_before:
                continue

            # efficiency = log カバー率増加 / log コスト増加
            eff = (math.log(p_after) - math.log(p_before)) / \
                  (math.log(next_k) - math.log(cur_k))

            candidates.append((m.priority, eff, i, next_k))

        if not candidates:
            return None

        # まず最高優先度グループに絞り込み
        max_priority = max(c[0] for c in candidates)
        top_group    = [c for c in candidates if c[0] == max_priority]

        # 同優先度内で efficiency 最大のものを選ぶ
        best = max(top_group, key=lambda c: c[1])
        return best[2], best[3]   # (idx, new_k)

    def optimize_scenarios(
        self, matches: list[MatchPrediction], budgets: list[int]
    ) -> list[tuple[int, MultiSelection]]:
        """複数予算シナリオを一括最適化"""
        results = []
        for b in sorted(budgets):
            opt = MultiOptimizer(budget_yen=b, allow_triple=self.allow_triple)
            results.append((b, opt.optimize(matches)))
        return results
