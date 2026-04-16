"""
toto-dream.com から過去のくじ結果を収集するスクレイパー

対象: toto (1/0/2 = ホーム勝/引き分け/アウェイ勝)
URL: store.toto-dream.com
取得データ: 開催回, 試合日, ホームチーム, アウェイチーム, ホームスコア, アウェイスコア, 結果(1/0/2)
"""

import time
import re
import csv
import json
import logging
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://store.toto-dream.com/dcs/subos/screen/pi04/spin011"
LIST_URL = f"{BASE_URL}/PGSPIN01101InitLotResultLsttoto.form"
DETAIL_URL = f"{BASE_URL}/PGSPIN01101LnkHoldCntLotResultLsttoto.form"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9",
}

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class TotoScraper:
    def __init__(self, sleep_sec: float = 1.5):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.sleep_sec = sleep_sec

    def _get(self, url: str, params: dict = None) -> Optional[BeautifulSoup]:
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as e:
            logger.error(f"GET失敗: {url} | {e}")
            return None

    def fetch_round_list(self) -> list[dict]:
        """全開催回の一覧を取得"""
        soup = self._get(LIST_URL, params={"popupDispDiv": "disp"})
        if soup is None:
            return []

        rounds = []
        # 開催回リンクを探す (holdCntId パラメータ)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            m = re.search(r"holdCntId=(\d+)", href)
            if m:
                hold_id = int(m.group(1))
                text = a.get_text(strip=True)
                rounds.append({"hold_cnt_id": hold_id, "label": text})

        rounds = sorted(rounds, key=lambda x: x["hold_cnt_id"])
        logger.info(f"開催回一覧: {len(rounds)}件 ({rounds[0]['hold_cnt_id']}〜{rounds[-1]['hold_cnt_id']})")
        return rounds

    def fetch_round_result(self, hold_cnt_id: int) -> list[dict]:
        """指定開催回の試合結果を取得"""
        soup = self._get(
            DETAIL_URL,
            params={"popupDispDiv": "disp", "holdCntId": hold_cnt_id},
        )
        if soup is None:
            return []

        results = []
        # テーブルから試合データを抽出
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cols = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                if len(cols) >= 5:
                    parsed = self._parse_row(cols, hold_cnt_id)
                    if parsed:
                        results.append(parsed)

        return results

    def _parse_row(self, cols: list[str], hold_cnt_id: int) -> Optional[dict]:
        """行データをパース。ホーム/アウェイ/スコア/結果を抽出"""
        # 典型的な列構成を探す: 試合番号, 日付, ホーム, スコア, アウェイ, 結果
        text = " ".join(cols)

        # スコア形式 "N対N" or "N-N" を含む行を対象
        score_pattern = re.search(r"(\d+)[対\-](\d+)", text)
        if not score_pattern:
            return None

        # 結果ラベル (1/0/2) を探す
        result = None
        for col in reversed(cols):
            if col in ("1", "0", "2"):
                result = col
                break
        if result is None:
            return None

        home_score = int(score_pattern.group(1))
        away_score = int(score_pattern.group(2))

        # チーム名候補 (スコアの前後)
        # cols から日付っぽい列とスコア列を除いた残りをチーム名候補に
        date_str = ""
        home_team = ""
        away_team = ""

        for i, col in enumerate(cols):
            if re.search(r"\d{4}[/\-年]\d{1,2}[/\-月]\d{1,2}", col):
                date_str = col

        score_idx = next(
            (i for i, c in enumerate(cols) if re.search(r"\d+[対\-]\d+", c)), None
        )
        if score_idx is not None and score_idx >= 1:
            home_team = cols[score_idx - 1]
            if score_idx + 1 < len(cols):
                away_team = cols[score_idx + 1]

        if not home_team or not away_team:
            return None

        # 結果の正規化: toto は 1=ホーム勝, 0=引き分け, 2=アウェイ勝
        if home_score > away_score:
            toto_result = "1"
        elif home_score == away_score:
            toto_result = "0"
        else:
            toto_result = "2"

        return {
            "hold_cnt_id": hold_cnt_id,
            "date": date_str,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "result": toto_result,
        }

    def scrape_all(
        self,
        start_id: Optional[int] = None,
        end_id: Optional[int] = None,
        output_path: Optional[Path] = None,
    ) -> list[dict]:
        """全開催回をスクレイプしてCSVに保存"""
        if output_path is None:
            output_path = DATA_DIR / "toto_results.csv"

        rounds = self.fetch_round_list()
        if not rounds:
            logger.error("開催回一覧の取得に失敗しました")
            return []

        if start_id:
            rounds = [r for r in rounds if r["hold_cnt_id"] >= start_id]
        if end_id:
            rounds = [r for r in rounds if r["hold_cnt_id"] <= end_id]

        all_results = []
        for i, r in enumerate(rounds):
            hid = r["hold_cnt_id"]
            logger.info(f"[{i+1}/{len(rounds)}] 第{hid}回 取得中...")
            matches = self.fetch_round_result(hid)
            all_results.extend(matches)
            time.sleep(self.sleep_sec)

        if all_results:
            self._save_csv(all_results, output_path)
            logger.info(f"保存完了: {output_path} ({len(all_results)}試合)")
        else:
            logger.warning("取得結果が0件です")

        return all_results

    def _save_csv(self, data: list[dict], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["hold_cnt_id", "date", "home_team", "away_team",
                      "home_score", "away_score", "result"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    scraper = TotoScraper(sleep_sec=1.5)
    # まず最新50回分だけ取得してデータ確認
    rounds = scraper.fetch_round_list()
    if rounds:
        recent = rounds[-50:]
        start = recent[0]["hold_cnt_id"]
        end = recent[-1]["hold_cnt_id"]
        scraper.scrape_all(start_id=start, end_id=end)
