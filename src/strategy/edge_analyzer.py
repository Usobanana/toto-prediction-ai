# -*- coding: utf-8 -*-
"""
エッジ分析モジュール
====================
HierBayes k=5 の確率 P と toto 投票率 Q の比率を「エッジ」として定義し、
試合を以下に分類するレポートと収支シミュレーションを提供する。

  勝負レース (波乱狙い): edge >= 1.2  → モデルが市場を大きく上回ると判断
  順当レース           : edge <= 1.05 → モデルと市場がほぼ一致
  中立               : その他

エッジの意味:
  edge = P(model) / Q(market)
  例) P=0.45, Q=0.30 → edge=1.50 → モデルは市場の1.5倍の確率を付けている
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# ──────────────────────────────────────────────────────────────────────
# 日本語 → football-data.co.uk 英語 チーム名マッピング
# ──────────────────────────────────────────────────────────────────────
JP_EN_MAP: dict[str, Optional[str]] = {
    # J1
    "横浜FM":   "Yokohama F. Marinos",
    "川崎Ｆ":   "Kawasaki Frontale",
    "川崎F":    "Kawasaki Frontale",
    "鹿島":     "Kashima Antlers",
    "浦和":     "Urawa Reds",
    "Ｃ大阪":   "Cerezo Osaka",
    "C大阪":    "Cerezo Osaka",
    "Ｇ大阪":   "Gamba Osaka",
    "G大阪":    "Gamba Osaka",
    "広島":     "Sanfrecce Hiroshima",
    "神戸":     "Vissel Kobe",
    "名古屋":   "Nagoya Grampus",
    "FC東京":   "FC Tokyo",
    "京都":     "Kyoto",
    "柏":       "Kashiwa Reysol",
    "湘南":     "Shonan Bellmare",
    "横浜FC":   "Yokohama FC",
    "清水":     "Shimizu S-Pulse",
    "鳥栖":     "Sagan Tosu",
    "町田":     "Machida",
    "福岡":     "Avispa Fukuoka",
    "長崎":     "V-Varen Nagasaki",
    # J2（football-data.co.ukにあるもの）
    "新潟":     "Albirex Niigata",
    "甲府":     "Kofu",
    "仙台":     "Vegalta Sendai",
    "山形":     "Montedio Yamagata",
    "岡山":     "Okayama",
    "磐田":     "Iwata",
    "東京Ｖ":   "Verdy",
    "東京V":    "Verdy",
    "大宮":     "Omiya Ardija",
    "札幌":     "Hokkaido Consadole Sapporo",
    "大分":     "Oita Trinita",
    "熊本":     "Kumamoto",
    "徳島":     "Tokushima",
    "秋田":     None,  # データなし
    "水戸":     None,
    "千葉":     None,
    "群馬":     None,
    "藤枝":     None,
    "山口":     None,
    "栃木SC":   None,
    "栃木Ｃ":   None,
    "いわき":   None,
    "宮崎":     None,
    "富山":     None,
    "岐阜":     None,
    "愛媛":     None,
    "讃岐":     None,
    "琉球":     None,
    "鳥取":     None,
    "高知":     None,
    "相模原":   None,
    "松本":     None,
    # J3
    "今治":     None,
    "八戸":     None,
    "鹿児島":   None,
    "滋賀":     None,
    # totoで時々使われる略記
    "FC大阪":   None,
}

# 勝負レース / 順当レース の閾値
EDGE_UPSET_THRESHOLD  = 1.2   # これ以上 → 勝負レース
EDGE_SOLID_THRESHOLD  = 1.05  # これ以下 → 順当レース


@dataclass
class MatchEdge:
    """1試合分のエッジ分析結果"""
    hold_cnt_id:  int
    match_no:     int
    home_jp:      str
    away_jp:      str
    home_en:      Optional[str]
    away_en:      Optional[str]
    # 投票率 (0–100)
    vote_rate_1:  float
    vote_rate_0:  float
    vote_rate_2:  float
    # モデル確率 (0–1)
    prob_1:       float
    prob_0:       float
    prob_2:       float
    # エッジ = prob / (vote_rate/100)
    edge_1:       float
    edge_0:       float
    edge_2:       float
    # 推奨と分類
    pred_simple:  str     # argmax(prob)
    pred_edge:    str     # argmax(edge)
    best_edge:    float   # max(edge)
    category:     str     # 勝負レース / 順当レース / 中立
    actual_result: Optional[str] = None  # 実際の結果
    has_model_data: bool = True  # モデルデータがあるか


@dataclass
class RoundReport:
    """1ラウンド分のレポート"""
    hold_cnt_id: int
    matches:     list[MatchEdge] = field(default_factory=list)

    @property
    def upset_matches(self):
        return [m for m in self.matches if m.category == "勝負レース"]

    @property
    def solid_matches(self):
        return [m for m in self.matches if m.category == "順当レース"]

    @property
    def neutral_matches(self):
        return [m for m in self.matches if m.category == "中立"]

    def correct_simple(self):
        return sum(1 for m in self.matches
                   if m.actual_result and m.pred_simple == m.actual_result)

    def correct_edge(self):
        return sum(1 for m in self.matches
                   if m.actual_result and m.pred_edge == m.actual_result)

    def total_with_result(self):
        return sum(1 for m in self.matches if m.actual_result is not None)


class EdgeAnalyzer:
    """
    エッジ分析・レポート・収支シミュレーションを提供するクラス

    Parameters
    ----------
    hier_model : HierarchicalPoissonModel
        学習済みの階層ベイズモデル
    """

    def __init__(self, hier_model):
        self.model = hier_model

    def _get_proba(self, home_jp: str, away_jp: str) -> tuple[list[float], bool]:
        """
        日本語チーム名から確率を取得。
        Returns: ([p1, p0, p2], has_model_data)
        """
        home_en = JP_EN_MAP.get(home_jp)
        away_en = JP_EN_MAP.get(away_jp)

        att_h = self.model._get_attack(home_en or "")
        def_h = self.model._get_defense(home_en or "")
        att_a = self.model._get_attack(away_en or "")
        def_a = self.model._get_defense(away_en or "")

        lam_h = self.model._mu_home * att_h * def_a
        lam_a = self.model._mu_away * att_a * def_h
        p1, p0, p2 = self.model._score_probs(lam_h, lam_a)

        # 両チームともデータなしの場合はデフォルト確率
        has_data = (home_en in self.model._attack) or (away_en in self.model._attack)
        return [p1, p0, p2], has_data

    def analyze_match(
        self,
        hold_cnt_id: int,
        match_no: int,
        home_jp: str,
        away_jp: str,
        vote_rate_1: float,
        vote_rate_0: float,
        vote_rate_2: float,
        actual_result: Optional[str] = None,
    ) -> MatchEdge:
        """1試合分のエッジを計算してMatchEdgeを返す"""
        proba, has_data = self._get_proba(home_jp, away_jp)
        p1, p0, p2 = proba

        # 投票率を 0–1 に正規化
        vr1 = max(vote_rate_1 / 100.0, 0.01)
        vr0 = max(vote_rate_0 / 100.0, 0.01)
        vr2 = max(vote_rate_2 / 100.0, 0.01)

        e1 = p1 / vr1
        e0 = p0 / vr0
        e2 = p2 / vr2

        pred_simple = ["1", "0", "2"][np.argmax([p1, p0, p2])]
        pred_edge   = ["1", "0", "2"][np.argmax([e1, e0, e2])]
        best_edge   = max(e1, e0, e2)

        if best_edge >= EDGE_UPSET_THRESHOLD:
            category = "勝負レース"
        elif best_edge <= EDGE_SOLID_THRESHOLD:
            category = "順当レース"
        else:
            category = "中立"

        return MatchEdge(
            hold_cnt_id=hold_cnt_id, match_no=match_no,
            home_jp=home_jp, away_jp=away_jp,
            home_en=JP_EN_MAP.get(home_jp), away_en=JP_EN_MAP.get(away_jp),
            vote_rate_1=vote_rate_1, vote_rate_0=vote_rate_0, vote_rate_2=vote_rate_2,
            prob_1=p1, prob_0=p0, prob_2=p2,
            edge_1=e1, edge_0=e0, edge_2=e2,
            pred_simple=pred_simple, pred_edge=pred_edge,
            best_edge=best_edge, category=category,
            actual_result=actual_result, has_model_data=has_data,
        )

    def analyze_round(
        self,
        hold_cnt_id: int,
        vote_df: pd.DataFrame,
        result_df: Optional[pd.DataFrame] = None,
    ) -> RoundReport:
        """1ラウンド全試合を分析してRoundReportを返す"""
        vr = vote_df[vote_df["hold_cnt_id"] == hold_cnt_id].sort_values("match_no")
        report = RoundReport(hold_cnt_id=hold_cnt_id)

        for _, row in vr.iterrows():
            actual = None
            if result_df is not None:
                mask = (result_df["hold_cnt_id"] == hold_cnt_id) & \
                       (result_df["match_no"] == row["match_no"])
                hits = result_df.loc[mask, "result"]
                if len(hits) > 0:
                    actual = str(hits.iloc[0])

            me = self.analyze_match(
                hold_cnt_id=hold_cnt_id,
                match_no=int(row["match_no"]),
                home_jp=row["home_team"],
                away_jp=row["away_team"],
                vote_rate_1=row["vote_rate_1"],
                vote_rate_0=row["vote_rate_0"],
                vote_rate_2=row["vote_rate_2"],
                actual_result=actual,
            )
            report.matches.append(me)

        return report

    def print_round_report(self, report: RoundReport):
        """ラウンドレポートをターミナルに表示"""
        hid = report.hold_cnt_id
        print()
        print("=" * 78)
        print(f"  第{hid}回 toto エッジ分析レポート")
        print("=" * 78)
        print(f"  {'No':>2}  {'試合':<18}  {'投票率1/0/2':>14}  "
              f"{'モデルP1/0/2':>14}  {'Edge':>5}  {'分類':<8}  推奨  実結果")
        print("  " + "-" * 74)

        for m in report.matches:
            vr_str = f"{m.vote_rate_1:.0f}/{m.vote_rate_0:.0f}/{m.vote_rate_2:.0f}%"
            pr_str = f"{m.prob_1*100:.0f}/{m.prob_0*100:.0f}/{m.prob_2*100:.0f}%"
            cat_mark = {"勝負レース": "[!]", "順当レース": "[ ]", "中立": "[-]"}
            result_mark = ""
            if m.actual_result:
                if m.pred_edge == m.actual_result:
                    result_mark = "○"
                else:
                    result_mark = "×"

            label_map = {"1": "ホ勝", "0": "引分", "2": "ア勝"}
            print(
                f"  {m.match_no:>2}. {m.home_jp:<8} vs {m.away_jp:<8}  "
                f"{vr_str:>14}  {pr_str:>14}  {m.best_edge:>4.2f}  "
                f"{cat_mark[m.category]}{m.category:<6}  "
                f"{label_map.get(m.pred_edge,'?'):>4}  "
                f"{result_mark} {m.actual_result or '-'}"
            )

        print("  " + "-" * 74)
        print(f"  勝負レース: {len(report.upset_matches)}試合  "
              f"順当レース: {len(report.solid_matches)}試合  "
              f"中立: {len(report.neutral_matches)}試合")
        if report.total_with_result() > 0:
            print(f"  エッジ戦略 正答: {report.correct_edge()}/{report.total_with_result()}  "
                  f"シンプル戦略 正答: {report.correct_simple()}/{report.total_with_result()}")
