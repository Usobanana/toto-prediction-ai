# -*- coding: utf-8 -*-
"""
toto 投票率スクレイパー
========================
store.toto-dream.com の「投票結果」ページから
各試合の1/0/2の投票率を取得する。

URL パターン:
  https://store.toto-dream.com/dcs/subos/screen/pi09/spin003/
  PGSPIN00301InitVoteRate.form?commodityId=01&holdCntId=<回>

取得データ:
  hold_cnt_id, match_no, home_team, away_team,
  vote_rate_1(%), vote_rate_0(%), vote_rate_2(%)

実行:
  python src/scraper/vote_rate_scraper.py --start 1580 --end 1621
  python src/scraper/vote_rate_scraper.py --last 50   # 直近50回
"""

import re
import time
import csv
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

BASE_URL = (
    "https://store.toto-dream.com/dcs/subos/screen/pi09/spin003/"
    "PGSPIN00301InitVoteRate.form"
)
LIST_URL = (
    "https://store.toto-dream.com/dcs/subos/screen/pi04/spin011/"
    "PGSPIN01101InitLotResultLsttoto.form"
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
OUTPUT_PATH = DATA_DIR / "toto_vote_rates.csv"


def fetch_vote_rates(hold_cnt_id: int, session: requests.Session) -> list[dict]:
    """1回分の投票率を取得して返す"""
    params = {"commodityId": "01", "holdCntId": str(hold_cnt_id)}
    try:
        resp = session.get(BASE_URL, params=params, timeout=20)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
    except requests.RequestException as e:
        logger.error(f"第{hold_cnt_id}回 取得失敗: {e}")
        return []

    return parse_vote_rates(resp.text, hold_cnt_id)


def parse_vote_rates(html: str, hold_cnt_id: int) -> list[dict]:
    """HTMLをパースして各試合の投票率を抽出

    各試合行の構造:
      [開催日, 開始時刻, 試合NO, ホーム, rate_1(%), rate_0(%), rate_2(%), アウェイ]
    """
    soup = BeautifulSoup(html, "lxml")
    records = []

    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 8:
            continue

        texts = [c.get_text(strip=True) for c in cells]

        # 試合NO列（index=2）: 1〜13 の整数
        match_no_str = texts[2]
        if not re.fullmatch(r"\d{1,2}", match_no_str):
            continue
        match_no = int(match_no_str)
        if not (1 <= match_no <= 13):
            continue

        # 投票率を抽出 (index 4,5,6)
        def extract_rate(text: str) -> Optional[float]:
            m = re.search(r"([\d.]+)%", text)
            return float(m.group(1)) if m else None

        rate_1 = extract_rate(texts[4])
        rate_0 = extract_rate(texts[5])
        rate_2 = extract_rate(texts[6])

        if rate_1 is None or rate_0 is None or rate_2 is None:
            continue

        home_team = texts[3]
        away_team = texts[7]

        records.append({
            "hold_cnt_id": hold_cnt_id,
            "match_no":    match_no,
            "home_team":   home_team,
            "away_team":   away_team,
            "vote_rate_1": rate_1,
            "vote_rate_0": rate_0,
            "vote_rate_2": rate_2,
        })

    return records


def get_available_rounds(session: requests.Session) -> list[int]:
    """利用可能な全開催回のIDリストを取得"""
    try:
        resp = session.get(
            LIST_URL, params={"popupDispDiv": "disp"}, timeout=20
        )
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
    except requests.RequestException as e:
        logger.error(f"開催回リスト取得失敗: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    ids = set()
    for a in soup.find_all("a", href=True):
        m = re.search(r"holdCntId=(\d+)", a["href"])
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def save_csv(records: list[dict], path: Path, append: bool = False):
    """CSVに保存（追記 or 上書き）"""
    mode = "a" if append and path.exists() else "w"
    fieldnames = [
        "hold_cnt_id", "match_no", "home_team", "away_team",
        "vote_rate_1", "vote_rate_0", "vote_rate_2",
    ]
    with open(path, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerows(records)
    logger.info(f"保存: {path} ({len(records)}件, mode={mode})")


def scrape_range(
    start_id: int,
    end_id: int,
    sleep_sec: float = 1.5,
    output: Path = OUTPUT_PATH,
) -> list[dict]:
    """指定範囲の開催回を巡回してスクレイプ"""
    session = requests.Session()
    session.headers.update(HEADERS)

    all_records = []
    ids = list(range(start_id, end_id + 1))
    total = len(ids)

    for i, hid in enumerate(ids, 1):
        logger.info(f"[{i}/{total}] 第{hid}回 取得中...")
        records = fetch_vote_rates(hid, session)
        if records:
            all_records.extend(records)
            logger.info(f"  -> {len(records)}試合 取得")
        else:
            logger.warning(f"  -> 0件 (非公開 or エラー)")
        time.sleep(sleep_sec)

    if all_records:
        save_csv(all_records, output)
        logger.info(f"\n完了: {len(all_records)}試合 -> {output}")
    return all_records


def main():
    parser = argparse.ArgumentParser(description="toto投票率スクレイパー")
    parser.add_argument("--start", type=int, help="開始回 (例: 1580)")
    parser.add_argument("--end",   type=int, help="終了回 (例: 1621)")
    parser.add_argument("--last",  type=int, help="直近N回 (例: 50)")
    parser.add_argument("--sleep", type=float, default=1.5, help="待機秒数")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.last:
        # 利用可能な最新回を取得して直近N回を対象
        logger.info("開催回リストを取得中...")
        session = requests.Session()
        session.headers.update(HEADERS)
        rounds = get_available_rounds(session)
        if not rounds:
            logger.error("開催回リストの取得に失敗")
            return
        target = rounds[-args.last:]
        start_id = target[0]
        end_id   = target[-1]
        logger.info(f"対象: 第{start_id}回 〜 第{end_id}回 ({len(target)}回)")
    elif args.start and args.end:
        start_id = args.start
        end_id   = args.end
    else:
        parser.print_help()
        return

    scrape_range(start_id, end_id, args.sleep, output_path)


if __name__ == "__main__":
    main()
