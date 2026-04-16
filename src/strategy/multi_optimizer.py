# -*- coding: utf-8 -*-
"""
toto マルチ購入 最適化モジュール

各試合で 1/2/3 択を選び、
  コスト = 100円 × ∏(選択数_i) ≤ 予算
の制約下で
  P(全的中) = ∏(選択試合のカバー確率_i)
を最大化する。

アルゴリズム: グリーディー昇格法
  全シングル → 効率の高い試合から順にダブル/トリプルへ昇格
"""

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MatchPrediction:
    """1試合分の予測情報"""
    no: int
    home: str
    away: str
    proba: list[float]          # [P(1), P(0), P(2)]
    method: str = "RF"

    @property
    def sorted_outcomes(self) -> list[tuple[str, float]]:
        """確率降順に並べた (結果ラベル, 確率) リスト"""
        pairs = [("1", self.proba[0]), ("0", self.proba[1]), ("2", self.proba[2])]
        return sorted(pairs, key=lambda x: x[1], reverse=True)

    def covered_prob(self, k: int) -> float:
        """上位k択を選んだ場合のカバー確率"""
        return sum(p for _, p in self.sorted_outcomes[:k])

    def top_k_labels(self, k: int) -> list[str]:
        """上位k択のラベルリスト"""
        return [label for label, _ in self.sorted_outcomes[:k]]


@dataclass
class MultiSelection:
    """マルチ購入の最適化結果"""
    matches: list[MatchPrediction]
    selections: list[int]           # 各試合の選択数 (1/2/3)
    budget_yen: int

    @property
    def n_combinations(self) -> int:
        result = 1
        for k in self.selections:
            result *= k
        return result

    @property
    def cost_yen(self) -> int:
        return self.n_combinations * 100

    @property
    def p_all_correct(self) -> float:
        """全的中確率"""
        p = 1.0
        for m, k in zip(self.matches, self.selections):
            p *= m.covered_prob(k)
        return p

    @property
    def expected_correct_per_match(self) -> list[float]:
        """各試合のカバー確率リスト"""
        return [m.covered_prob(k) for m, k in zip(self.matches, self.selections)]

    def summary(self) -> str:
        lines = []
        lines.append("=" * 72)
        lines.append(" toto マルチ予想 最適化結果")
        lines.append("=" * 72)
        lines.append(
            f" 予算: {self.budget_yen:,}円 | "
            f"使用: {self.cost_yen:,}円 | "
            f"組み合わせ数: {self.n_combinations}通り"
        )
        lines.append(f" 全的中確率(理論値): {self.p_all_correct * 100:.4f}%")
        lines.append("-" * 72)
        lines.append(
            f"{'No':>2}  {'ホーム':<9} {'':>2} {'アウェイ':<9}  "
            f"{'択':<2}  {'選択':<10}  カバー率  {'確率(1/0/2)'}"
        )
        lines.append("-" * 72)

        for m, k in zip(self.matches, self.selections):
            labels = m.top_k_labels(k)
            cov = m.covered_prob(k)
            star = " *" if k > 1 else "  "
            label_str = "/".join(labels)
            lines.append(
                f"{m.no:>2}. {m.home:<9} vs {m.away:<9}  "
                f"{k}択  [{label_str:<5}]{star}  "
                f"{cov:.2f}     "
                f"1:{m.proba[0]:.2f}/0:{m.proba[1]:.2f}/2:{m.proba[2]:.2f}"
                f"  ({m.method})"
            )

        lines.append("-" * 72)
        # 予想配列 (全通りを列挙)
        combo_list = self._enumerate_combinations()
        lines.append(f" 全組み合わせ ({len(combo_list)}通り):")
        for i, combo in enumerate(combo_list, 1):
            lines.append(f"  [{i:>2}] {' / '.join(combo)}")
        lines.append("=" * 72)
        lines.append(" * = 複数択選択 (マルチ)")
        lines.append(f" 各試合カバー率平均: {sum(self.expected_correct_per_match)/len(self.matches):.3f}")
        return "\n".join(lines)

    def _enumerate_combinations(self) -> list[list[str]]:
        """全組み合わせを列挙"""
        combos = [[]]
        for m, k in zip(self.matches, self.selections):
            labels = m.top_k_labels(k)
            combos = [c + [l] for c in combos for l in labels]
        return combos


class MultiOptimizer:
    """
    グリーディー昇格法によるマルチ購入最適化

    Parameters
    ----------
    budget_yen : int
        購入予算 (円)
    allow_triple : bool
        3択 (トリプル) を許可するか
    """

    def __init__(self, budget_yen: int = 5000, allow_triple: bool = True):
        self.budget_yen = budget_yen
        self.allow_triple = allow_triple
        self.max_combinations = budget_yen // 100

    def optimize(self, matches: list[MatchPrediction]) -> MultiSelection:
        """最適なマルチ選択を返す"""
        n = len(matches)
        selections = [1] * n          # 全シングルからスタート
        current_cost = 1              # 組み合わせ数

        while True:
            best_gain = -1.0
            best_idx = -1
            best_new_k = -1

            for i, m in enumerate(matches):
                cur_k = selections[i]
                next_k = cur_k + 1
                if next_k > 3:
                    continue
                if not self.allow_triple and next_k > 2:
                    continue

                # コスト確認: 昇格したら current_cost は (next_k/cur_k) 倍
                new_cost = current_cost * next_k // cur_k
                if new_cost > self.max_combinations:
                    continue

                # 効率 = カバー確率の対数増加量 / コスト対数増加量
                p_before = m.covered_prob(cur_k)
                p_after = m.covered_prob(next_k)
                if p_before <= 0 or p_after <= p_before:
                    continue

                log_gain = math.log(p_after) - math.log(p_before)
                log_cost = math.log(next_k) - math.log(cur_k)
                efficiency = log_gain / log_cost

                if efficiency > best_gain:
                    best_gain = efficiency
                    best_idx = i
                    best_new_k = next_k

            if best_idx == -1:
                break  # これ以上昇格できない

            # 昇格実行
            old_k = selections[best_idx]
            selections[best_idx] = best_new_k
            current_cost = current_cost * best_new_k // old_k

        return MultiSelection(
            matches=matches,
            selections=selections,
            budget_yen=self.budget_yen,
        )

    def optimize_multiple_scenarios(
        self, matches: list[MatchPrediction], n_scenarios: int = 3
    ) -> list[MultiSelection]:
        """
        予算を変えた複数シナリオで最適化
        例: 1000円 / 3000円 / 5000円
        """
        scenarios = []
        budgets = self._get_budget_breakpoints(n_scenarios)
        for b in budgets:
            opt = MultiOptimizer(budget_yen=b, allow_triple=self.allow_triple)
            scenarios.append(opt.optimize(matches))
        return scenarios

    def _get_budget_breakpoints(self, n: int) -> list[int]:
        """予算の主要区切り (100円単位で組み合わせ数が変化するポイント)"""
        breakpoints = []
        for power2 in range(0, 7):    # 1〜64通り
            for triple_factor in [1, 3]:
                cost = (2 ** power2) * triple_factor * 100
                if cost <= self.budget_yen:
                    breakpoints.append(cost)
        breakpoints = sorted(set(breakpoints))
        # 均等にn点選ぶ
        step = max(1, len(breakpoints) // n)
        return breakpoints[-n * step::step][:n] or [self.budget_yen]
