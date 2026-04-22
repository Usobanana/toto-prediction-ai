# -*- coding: utf-8 -*-
"""
toto 対戦データスクレイパー
===========================
store.toto-dream.com の対戦データページ (gameid ベース) から
通算対戦成績・最近5試合の H2H データを取得する。

URL パターン:
  https://store.toto-dream.com/dcs/subos/screen/pi03/spin002/
  PGSPIN00201InitRivalTeamData.form?gameid=<gameid>

取得データ (通算):
  gameid, home_team, away_team, date,
  h2h_home_win_home,   h2h_draw_home,   h2h_away_win_home,   (ホーム時)
  h2h_home_win_away,   h2h_draw_away,   h2h_away_win_away,   (アウェイ時)
  h2h_home_win_total,  h2h_draw_total,  h2h_away_win_total   (合計)

取得データ (最近5試合):
  gameid, match_no (1-5), date, venue, home, score_home, score_away, away

実行例:
  # 第1624回 (sales中) の H2H を取得
  python src/scraper/rival_team_scraper.py --hold-cnt-id 1624 --seed-gameid 54051

  # 既知の gameid を直接指定
  python src/scraper/rival_team_scraper.py --gameids 54051 54052 54053
"""

import re
import time
import csv
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

RIVAL_URL = (
    "https://store.toto-dream.com/dcs/subos/screen/pi03/spin002/"
    "PGSPIN00201InitRivalTeamData.form"
)
VOTE_RATE_URL = (
    "https://store.toto-dream.com/dcs/subos/screen/pi09/spin003/"
    "PGSPIN00301InitVoteRate.form"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9",
    "Referer": "https://store.toto-dream.com/",
}

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED_FILE = DATA_DIR / "rival_team_seed.json"
H2H_SUMMARY_PATH = DATA_DIR / "toto_h2h_summary.csv"
H2H_RECENT_PATH  = DATA_DIR / "toto_h2h_recent.csv"

# gameid 走査の最大オフセット (1回あたり13試合なので余裕を持たせる)
SCAN_RANGE = 40


# ─── ユーティリティ ───────────────────────────────────────────────────────────

def _normalize_team(name: str) -> str:
    """スペース・全角スペース・略称の正規化"""
    return re.sub(r"[\s　]+", "", name).strip()


def _teams_match(t1: str, t2: str) -> bool:
    """チーム名を緩くマッチング (全角半角・略称を考慮)"""
    a = _normalize_team(t1).replace("Ｆ", "F").replace("Ｃ", "C")
    b = _normalize_team(t2).replace("Ｆ", "F").replace("Ｃ", "C")
    return a == b or a.startswith(b) or b.startswith(a) or a in b or b in a


def _load_seed() -> int:
    """最後に確認した gameid を読み込む (なければ 54000 を返す)"""
    if SEED_FILE.exists():
        try:
            return int(json.loads(SEED_FILE.read_text(encoding="utf-8")).get("last_gameid", 54000))
        except Exception:
            pass
    return 54000


def _save_seed(gameid: int):
    SEED_FILE.write_text(json.dumps({"last_gameid": gameid}), encoding="utf-8")


# ─── メインスクレイパークラス ─────────────────────────────────────────────────

class RivalTeamScraper:
    def __init__(self, sleep_sec: float = 1.5):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.sleep_sec = sleep_sec

    def _get(self, url: str, params: dict = None) -> Optional[BeautifulSoup]:
        try:
            resp = self.session.get(url, params=params, timeout=20)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as e:
            logger.debug(f"GET失敗: {url} params={params} | {e}")
            return None

    # ── gameid ページのパース ──────────────────────────────────────────────

    def fetch_rival_data(self, gameid: int) -> Optional[dict]:
        """
        1 つの gameid からデータを取得してパース。
        失敗または対象ページでない場合は None を返す。
        """
        soup = self._get(RIVAL_URL, params={"gameid": gameid})
        if soup is None:
            return None

        # ページにコンテンツがあるか確認 (エラーページ判定)
        body_text = soup.get_text()
        if "システムでエラーが発生しました" in body_text or "通算対戦成績" not in body_text:
            return None

        result = {"gameid": gameid}

        # ── チーム名・試合日・会場 ───────────────────────────────────────
        home_team, away_team, match_date = self._parse_basic_info(soup)
        result["home_team"]  = home_team
        result["away_team"]  = away_team
        result["match_date"] = match_date

        # ── 通算対戦成績テーブル ──────────────────────────────────────────
        summary = self._parse_h2h_summary(soup)
        result.update(summary)

        # ── 最近5試合 ────────────────────────────────────────────────────
        result["recent_matches"] = self._parse_recent_matches(soup)

        return result

    def _get_all_cells(self, soup: BeautifulSoup) -> list[str]:
        """対戦データを含むテーブルの全tdテキストをフラットなリストで返す"""
        for table in soup.find_all("table"):
            cells = [td.get_text(strip=True) for td in table.find_all("td")]
            # "X勝ち" パターンが2つ以上あれば対戦データテーブルと判断
            kachi_count = sum(1 for c in cells if c.endswith("勝ち"))
            if kachi_count >= 2:
                return cells
        # fallback: 最初のテーブル
        if soup.find("table"):
            return [td.get_text(strip=True) for td in soup.find("table").find_all("td")]
        return []

    def _parse_basic_info(self, soup: BeautifulSoup) -> tuple[str, str, str]:
        """チーム名と試合日を抽出"""
        match_date = ""
        home_team  = ""
        away_team  = ""

        # 試合日: "YYYY年MM月DD日" パターン
        date_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", soup.get_text())
        if date_match:
            match_date = f"{date_match.group(1)}/{int(date_match.group(2)):02d}/{int(date_match.group(3)):02d}"

        # チーム名: "X勝ち" セルが home/away の名前を含む
        cells = self._get_all_cells(soup)
        for cell in cells:
            if cell.endswith("勝ち"):
                name = cell[:-2]  # "仙台勝ち" → "仙台"
                if not home_team:
                    home_team = name
                elif not away_team:
                    away_team = name
                    break

        return home_team, away_team, match_date

    def _parse_h2h_summary(self, soup: BeautifulSoup) -> dict:
        """
        通算対戦成績テーブルをパース。
        セル列例: ... 'ホーム時', '12勝', '8分', '4勝', 'アウェイ時',
                       'アウェイ時', '10勝', '8分', '5勝', 'ホーム時',
                       '合計', '22勝', '16分', '9勝', '合計' ...
        """
        result = {
            "h2h_home_win_home": None, "h2h_draw_home": None, "h2h_away_win_home": None,
            "h2h_home_win_away": None, "h2h_draw_away": None, "h2h_away_win_away": None,
            "h2h_home_win_total": None,"h2h_draw_total": None,"h2h_away_win_total": None,
        }

        cells = self._get_all_cells(soup)

        def _extract_num(s: str) -> Optional[int]:
            m = re.search(r"(\d+)", s)
            return int(m.group(1)) if m else None

        def _read_triple(cells, start) -> Optional[tuple[int, int, int]]:
            """cells[start:start+3] から (勝, 分, 勝) を読む"""
            if start + 2 >= len(cells):
                return None
            a = _extract_num(cells[start])
            b = _extract_num(cells[start + 1])
            c = _extract_num(cells[start + 2])
            if a is not None and b is not None and c is not None:
                return a, b, c
            return None

        # "ホーム時" を探してその直後3セルを読む
        home_sections = []
        away_sections = []
        total_sections = []

        i = 0
        while i < len(cells):
            cell = cells[i]
            if cell == "ホーム時":
                triple = _read_triple(cells, i + 1)
                if triple:
                    home_sections.append((i, triple))
            elif cell == "アウェイ時":
                triple = _read_triple(cells, i + 1)
                if triple:
                    away_sections.append((i, triple))
            elif cell == "合計":
                triple = _read_triple(cells, i + 1)
                if triple:
                    total_sections.append((i, triple))
            i += 1

        if home_sections:
            _, (w, d, l) = home_sections[0]
            result["h2h_home_win_home"] = w
            result["h2h_draw_home"]     = d
            result["h2h_away_win_home"] = l

        if away_sections:
            _, (w, d, l) = away_sections[0]
            result["h2h_home_win_away"] = w
            result["h2h_draw_away"]     = d
            result["h2h_away_win_away"] = l

        if total_sections:
            _, (w, d, l) = total_sections[0]
            result["h2h_home_win_total"] = w
            result["h2h_draw_total"]     = d
            result["h2h_away_win_total"] = l

        return result

    def _parse_recent_matches(self, soup: BeautifulSoup) -> list[dict]:
        """最近5試合の個別結果を抽出 (セルのフラットリストから YYYY/MM/DD とスコアを検索)"""
        matches = []
        cells = self._get_all_cells(soup)
        for cell in cells:
            date_m = re.search(r"(\d{4}/\d{2}/\d{2})", cell)
            score_m = re.search(r"(\d+)\s*[-－]\s*(\d+)", cell)
            if date_m and score_m:
                matches.append({
                    "date":       date_m.group(1),
                    "score_home": int(score_m.group(1)),
                    "score_away": int(score_m.group(2)),
                })
                if len(matches) >= 5:
                    break
        return matches

    # ── holdCntId から gameid を発見 ─────────────────────────────────────

    def get_round_teams(self, hold_cnt_id: int) -> list[tuple[str, str]]:
        """
        vote_rate ページから (home_team, away_team) のリストを取得。
        """
        soup = self._get(VOTE_RATE_URL, params={"commodityId": "01", "holdCntId": hold_cnt_id})
        if soup is None:
            return []

        teams = []
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 8:
                continue
            texts = [c.get_text(strip=True) for c in cells]
            match_no_str = texts[2] if len(texts) > 2 else ""
            if not re.fullmatch(r"\d{1,2}", match_no_str):
                continue
            match_no = int(match_no_str)
            if not (1 <= match_no <= 13):
                continue
            home = texts[3] if len(texts) > 3 else ""
            away = texts[7] if len(texts) > 7 else ""
            if home and away:
                teams.append((home, away))

        return teams

    def discover_gameids(
        self,
        hold_cnt_id: int,
        seed_gameid: Optional[int] = None,
    ) -> dict[tuple[str, str], int]:
        """
        holdCntId に属する試合の gameid を発見して返す。
        seed_gameid 付近を走査し、vote_rate ページの試合リストとマッチングする。

        Returns:
            { (home_team, away_team): gameid }
        """
        expected_teams = self.get_round_teams(hold_cnt_id)
        if not expected_teams:
            logger.error(f"第{hold_cnt_id}回の試合リスト取得失敗")
            return {}

        logger.info(f"第{hold_cnt_id}回: {len(expected_teams)}試合を期待")
        for h, a in expected_teams:
            logger.info(f"  {h} vs {a}")

        if seed_gameid is None:
            seed_gameid = _load_seed()

        # seed から -SCAN_RANGE ~ +SCAN_RANGE の範囲を走査
        found: dict[tuple[str, str], int] = {}
        start = max(1, seed_gameid - SCAN_RANGE)
        end   = seed_gameid + SCAN_RANGE

        logger.info(f"gameid {start}〜{end} を走査中...")
        for gid in range(start, end + 1):
            if len(found) == len(expected_teams):
                break

            data = self.fetch_rival_data(gid)
            time.sleep(self.sleep_sec)
            if data is None:
                continue

            h_page = data.get("home_team", "")
            a_page = data.get("away_team", "")
            if not h_page or not a_page:
                continue

            # expected_teams とマッチング
            for (exp_home, exp_away) in expected_teams:
                if (exp_home, exp_away) in found:
                    continue
                if _teams_match(h_page, exp_home) and _teams_match(a_page, exp_away):
                    found[(exp_home, exp_away)] = gid
                    logger.info(f"  マッチ: {exp_home} vs {exp_away} → gameid={gid}")
                    _save_seed(gid)
                    break

        logger.info(f"発見: {len(found)}/{len(expected_teams)} 試合")
        return found

    # ── 開催回まるごとスクレイプ ─────────────────────────────────────────

    def scrape_round(
        self,
        hold_cnt_id: int,
        seed_gameid: Optional[int] = None,
        summary_path: Path = H2H_SUMMARY_PATH,
        recent_path: Path  = H2H_RECENT_PATH,
    ) -> list[dict]:
        """
        holdCntId の全試合 H2H データを取得して CSV に追記保存する。
        """
        gameid_map = self.discover_gameids(hold_cnt_id, seed_gameid)
        if not gameid_map:
            return []

        summary_rows = []
        recent_rows  = []

        for (home, away), gid in gameid_map.items():
            data = self.fetch_rival_data(gid)
            time.sleep(self.sleep_sec)
            if data is None:
                logger.warning(f"gameid={gid} データ取得失敗")
                continue

            row = {
                "hold_cnt_id":         hold_cnt_id,
                "gameid":              gid,
                "home_team":           home,
                "away_team":           away,
                "match_date":          data.get("match_date", ""),
                "h2h_home_win_home":   data.get("h2h_home_win_home"),
                "h2h_draw_home":       data.get("h2h_draw_home"),
                "h2h_away_win_home":   data.get("h2h_away_win_home"),
                "h2h_home_win_away":   data.get("h2h_home_win_away"),
                "h2h_draw_away":       data.get("h2h_draw_away"),
                "h2h_away_win_away":   data.get("h2h_away_win_away"),
                "h2h_home_win_total":  data.get("h2h_home_win_total"),
                "h2h_draw_total":      data.get("h2h_draw_total"),
                "h2h_away_win_total":  data.get("h2h_away_win_total"),
            }
            summary_rows.append(row)

            for i, m in enumerate(data.get("recent_matches", []), 1):
                recent_rows.append({
                    "hold_cnt_id": hold_cnt_id,
                    "gameid":      gid,
                    "home_team":   home,
                    "away_team":   away,
                    "match_no":    i,
                    "date":        m["date"],
                    "score_home":  m["score_home"],
                    "score_away":  m["score_away"],
                })

        if summary_rows:
            _append_csv(summary_rows, summary_path, list(summary_rows[0].keys()))
            logger.info(f"通算成績: {len(summary_rows)}試合 → {summary_path}")
        if recent_rows:
            _append_csv(recent_rows, recent_path, list(recent_rows[0].keys()))
            logger.info(f"最近5試合: {len(recent_rows)}行 → {recent_path}")

        return summary_rows

    def scrape_gameids(
        self,
        gameids: list[int],
        hold_cnt_id: Optional[int] = None,
        summary_path: Path = H2H_SUMMARY_PATH,
        recent_path: Path  = H2H_RECENT_PATH,
    ) -> list[dict]:
        """
        gameid リストを直接指定してスクレイプ。
        """
        summary_rows = []
        recent_rows  = []

        for gid in gameids:
            data = self.fetch_rival_data(gid)
            time.sleep(self.sleep_sec)
            if data is None:
                logger.warning(f"gameid={gid} データ取得失敗")
                continue

            _save_seed(gid)
            row = {
                "hold_cnt_id":         hold_cnt_id,
                "gameid":              gid,
                "home_team":           data.get("home_team", ""),
                "away_team":           data.get("away_team", ""),
                "match_date":          data.get("match_date", ""),
                "h2h_home_win_home":   data.get("h2h_home_win_home"),
                "h2h_draw_home":       data.get("h2h_draw_home"),
                "h2h_away_win_home":   data.get("h2h_away_win_home"),
                "h2h_home_win_away":   data.get("h2h_home_win_away"),
                "h2h_draw_away":       data.get("h2h_draw_away"),
                "h2h_away_win_away":   data.get("h2h_away_win_away"),
                "h2h_home_win_total":  data.get("h2h_home_win_total"),
                "h2h_draw_total":      data.get("h2h_draw_total"),
                "h2h_away_win_total":  data.get("h2h_away_win_total"),
            }
            summary_rows.append(row)
            logger.info(f"gameid={gid}: {row['home_team']} vs {row['away_team']}")

            for i, m in enumerate(data.get("recent_matches", []), 1):
                recent_rows.append({
                    "hold_cnt_id": hold_cnt_id,
                    "gameid":      gid,
                    "home_team":   row["home_team"],
                    "away_team":   row["away_team"],
                    "match_no":    i,
                    "date":        m["date"],
                    "score_home":  m["score_home"],
                    "score_away":  m["score_away"],
                })

        if summary_rows:
            _append_csv(summary_rows, summary_path, list(summary_rows[0].keys()))
        if recent_rows:
            _append_csv(recent_rows, recent_path, list(recent_rows[0].keys()))

        return summary_rows


# ─── CSV ユーティリティ ───────────────────────────────────────────────────────

def _append_csv(rows: list[dict], path: Path, fieldnames: list[str]):
    """CSV に追記 (ファイルがなければヘッダー付きで新規作成)"""
    mode = "a" if path.exists() else "w"
    with open(path, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="toto対戦データスクレイパー")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--hold-cnt-id", type=int, help="開催回 (例: 1624)")
    grp.add_argument("--gameids", type=int, nargs="+", help="gameid を直接指定 (例: 54051 54052)")

    parser.add_argument("--seed-gameid", type=int, default=None,
                        help="gameid 走査の起点 (省略時は前回保存値を使用)")
    parser.add_argument("--sleep", type=float, default=1.5, help="待機秒数")
    parser.add_argument("--summary-out", type=str, default=str(H2H_SUMMARY_PATH))
    parser.add_argument("--recent-out",  type=str, default=str(H2H_RECENT_PATH))
    args = parser.parse_args()

    scraper = RivalTeamScraper(sleep_sec=args.sleep)
    summary_path = Path(args.summary_out)
    recent_path  = Path(args.recent_out)

    if args.hold_cnt_id:
        results = scraper.scrape_round(
            hold_cnt_id=args.hold_cnt_id,
            seed_gameid=args.seed_gameid,
            summary_path=summary_path,
            recent_path=recent_path,
        )
    else:
        results = scraper.scrape_gameids(
            gameids=args.gameids,
            summary_path=summary_path,
            recent_path=recent_path,
        )

    print(f"\n完了: {len(results)} 試合分のH2Hデータ取得")


if __name__ == "__main__":
    main()
