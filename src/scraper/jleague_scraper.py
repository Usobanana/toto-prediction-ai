"""
Jリーグ試合データ収集スクレイパー

ソース1: football-data.co.uk (メイン) - J1リーグ 2012-現在, ベッティングオッズ付き
ソース2: TheSportsDB API (補完用) - J1リーグ 2020-現在
"""

import time
import csv
import io
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9",
}

# football-data.co.uk - J1リーグ (H=ホーム勝, D=引き分け, A=アウェイ勝)
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/new/JPN.csv"

# TheSportsDB - J1リーグ ID
THESPORTSDB_LEAGUE_ID = 4633
THESPORTSDB_BASE = "https://www.thesportsdb.com/api/v1/json/3"

# J-League公式データサイト
JLEAGUE_DATA_BASE = "https://data.j-league.or.jp"


class FootballDataScraper:
    """
    football-data.co.uk から J1リーグデータを取得 (無料CSV)
    データ: 2012~現在, 列: Home, Away, HG, AG, Res, オッズ各種
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def fetch(self) -> list[dict]:
        """全シーズンデータを一括取得"""
        import pandas as pd

        resp = self.session.get(FOOTBALL_DATA_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        logger.info(f"football-data.co.uk: {len(df)}試合取得 (シーズン: {sorted(df['Season'].unique())})")
        return self._convert(df)

    def _convert(self, df) -> list[dict]:
        """H/D/A を 1/0/2 に変換し統一フォーマットで返す"""
        res_map = {"H": "1", "D": "0", "A": "2"}
        results = []
        for _, row in df.iterrows():
            if str(row.get("Res", "")) not in res_map:
                continue
            results.append({
                "date": str(row.get("Date", "")),
                "season": str(row.get("Season", "")),
                "home_team": str(row.get("Home", "")),
                "away_team": str(row.get("Away", "")),
                "home_score": int(row["HG"]) if str(row.get("HG", "")).isdigit() else None,
                "away_score": int(row["AG"]) if str(row.get("AG", "")).isdigit() else None,
                "result": res_map[row["Res"]],
                # ベッティングオッズ (特徴量として使用可能)
                "odds_home_avg": float(row["AvgCH"]) if str(row.get("AvgCH", "")).replace(".", "").isdigit() else None,
                "odds_draw_avg": float(row["AvgCD"]) if str(row.get("AvgCD", "")).replace(".", "").isdigit() else None,
                "odds_away_avg": float(row["AvgCA"]) if str(row.get("AvgCA", "")).replace(".", "").isdigit() else None,
            })
        logger.info(f"変換後: {len(results)}試合 (結果あり)")
        return results

    def save(self, data: list[dict], path: Path = None):
        if path is None:
            path = DATA_DIR / "jleague_results.csv"
        import csv
        if not data:
            return
        fieldnames = list(data[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        logger.info(f"保存完了: {path} ({len(data)}試合)")


class TheSportsDBScraper:
    """TheSportsDB 無料APIからJ1リーグデータを取得"""

    def __init__(self, sleep_sec: float = 1.0):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.sleep_sec = sleep_sec

    def _get_json(self, endpoint: str, params: dict = None) -> Optional[dict]:
        url = f"{THESPORTSDB_BASE}/{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"API失敗: {url} | {e}")
            return None

    def fetch_season_results(self, season: str) -> list[dict]:
        """指定シーズンの全試合結果を取得 (例: season='2023-2024' or '2023')"""
        data = self._get_json(
            "eventsseason.php",
            params={"id": THESPORTSDB_LEAGUE_ID, "s": season},
        )
        if not data or "events" not in data or not data["events"]:
            logger.warning(f"シーズン {season} のデータなし")
            return []

        results = []
        for ev in data["events"]:
            if ev.get("strStatus") not in ("Match Finished", "FT"):
                continue
            results.append(self._parse_event(ev))

        logger.info(f"シーズン {season}: {len(results)}試合取得")
        return results

    def _parse_event(self, ev: dict) -> dict:
        home_score = int(ev.get("intHomeScore") or 0)
        away_score = int(ev.get("intAwayScore") or 0)

        if home_score > away_score:
            result = "1"
        elif home_score == away_score:
            result = "0"
        else:
            result = "2"

        return {
            "event_id": ev.get("idEvent"),
            "date": ev.get("dateEvent"),
            "season": ev.get("strSeason"),
            "round": ev.get("intRound"),
            "home_team": ev.get("strHomeTeam"),
            "away_team": ev.get("strAwayTeam"),
            "home_score": home_score,
            "away_score": away_score,
            "result": result,
            "venue": ev.get("strVenue"),
        }

    def fetch_season_by_rounds(self, season: str, max_rounds: int = 38) -> list[dict]:
        """
        ラウンド別に取得 (無料APIはシーズン一括が15件制限のため)
        J1は通常34ラウンド、ACLなど含め38まで試みる
        """
        results = []
        empty_rounds = 0

        for r in range(1, max_rounds + 1):
            data = self._get_json(
                "eventsround.php",
                params={"id": THESPORTSDB_LEAGUE_ID, "r": r, "s": season},
            )
            if not data or not data.get("events"):
                empty_rounds += 1
                if empty_rounds >= 3:
                    break  # 3ラウンド連続で空なら終了
                continue

            empty_rounds = 0
            for ev in data["events"]:
                if ev.get("intHomeScore") is None or ev.get("intAwayScore") is None:
                    continue
                results.append(self._parse_event(ev))
            time.sleep(self.sleep_sec)

        logger.info(f"シーズン {season}: {len(results)}試合取得 (ラウンド別)")
        return results

    def fetch_all_seasons(self, seasons: list[str] = None) -> list[dict]:
        """複数シーズンのデータを一括取得"""
        if seasons is None:
            # 利用可能な主要シーズン
            seasons = ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]

        all_results = []
        for season in seasons:
            results = self.fetch_season_by_rounds(season)
            all_results.extend(results)

        return all_results

    def save(self, data: list[dict], path: Path = None):
        if path is None:
            path = DATA_DIR / "jleague_results.csv"
        if not data:
            logger.warning("保存データなし")
            return

        fieldnames = ["event_id", "date", "season", "round", "home_team",
                      "away_team", "home_score", "away_score", "result", "venue"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)
        logger.info(f"保存完了: {path} ({len(data)}試合)")


class JLeagueOfficialScraper:
    """data.j-league.or.jp からスクレイピング"""

    def __init__(self, sleep_sec: float = 2.0):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.sleep_sec = sleep_sec

    def _get(self, url: str, params: dict = None) -> Optional[BeautifulSoup]:
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            logger.error(f"GET失敗: {url} | {e}")
            return None

    def fetch_season_schedule(self, year: int, competition_id: str = "010") -> list[dict]:
        """
        試合日程・結果を取得
        competition_id: 010=J1, 020=J2, 030=J3
        """
        url = f"{JLEAGUE_DATA_BASE}/SFTP01/"
        # J-League公式サイトはJavaScriptが多用されているため、
        # 直接のスクレイピングが困難な場合はTheSportsDBにフォールバック
        soup = self._get(url)
        if soup is None:
            return []

        results = []
        tables = soup.find_all("table")
        for table in tables:
            for row in table.find_all("tr")[1:]:
                cols = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                if len(cols) >= 6:
                    results.append({
                        "date": cols[0] if cols else "",
                        "round": cols[1] if len(cols) > 1 else "",
                        "home_team": cols[2] if len(cols) > 2 else "",
                        "score": cols[3] if len(cols) > 3 else "",
                        "away_team": cols[4] if len(cols) > 4 else "",
                        "venue": cols[5] if len(cols) > 5 else "",
                    })

        return results


def collect_jleague_data(seasons: list[str] = None) -> list[dict]:
    """J1リーグデータを収集してCSVに保存"""
    scraper = TheSportsDBScraper()
    data = scraper.fetch_all_seasons(seasons)
    if data:
        scraper.save(data)
    return data


if __name__ == "__main__":
    data = collect_jleague_data()
    print(f"取得完了: {len(data)}試合")
    if data:
        print("サンプル:", json.dumps(data[0], ensure_ascii=False, indent=2))
