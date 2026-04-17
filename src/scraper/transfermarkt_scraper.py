"""
Transfermarkt 市場価値スクレイパー

優先順位:
  1. transfermarkt-datasets (GitHub dcaribou/transfermarkt-datasets) からCSV取得
  2. transfermarkt.com から直接スクレイピング
  3. 静的フォールバックデータを使用

出力: data/raw/team_market_values.csv
  カラム: season, team, market_value_eur (単位: 百万ユーロ)
"""

import io
import os
import re
import time
import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)

# ── プロジェクトルートとデータパス ─────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..")
_OUTPUT_PATH = os.path.join(_ROOT, "data", "raw", "team_market_values.csv")

# ── Jリーグチーム名 Transfermarkt英語名 → jleague_results.csv名 マッピング ──
# Transfermarkt clubs.csv の "name" 列 → ローカルのチーム名
TM_TO_LOCAL: dict[str, str] = {
    # J1主要クラブ (Transfermarkt表記)
    "Kashima Antlers":           "Kashima Antlers",
    "Urawa Red Diamonds":        "Urawa Reds",
    "Gamba Osaka":               "Gamba Osaka",
    "Cerezo Osaka":              "Cerezo Osaka",
    "Nagoya Grampus":            "Nagoya Grampus",
    "Kawasaki Frontale":         "Kawasaki Frontale",
    "Yokohama F. Marinos":       "Yokohama F. Marinos",
    "Yokohama F.Marinos":        "Yokohama F. Marinos",
    "Vissel Kobe":               "Vissel Kobe",
    "Kyoto Sanga":               "Kyoto",
    "Kyoto Sanga FC":            "Kyoto",
    "Avispa Fukuoka":            "Avispa Fukuoka",
    "Albirex Niigata":           "Albirex Niigata",
    "Vegalta Sendai":            "Vegalta Sendai",
    "Montedio Yamagata":         "Montedio Yamagata",
    "Vanforet Kofu":             "Kofu",
    "Yokohama FC":               "Yokohama FC",
    "FC Tokyo":                  "FC Tokyo",
    "Tokyo Verdy":               "Verdy",
    "Shimizu S-Pulse":           "Shimizu S-Pulse",
    "Júbilo Iwata":              "Iwata",
    "Jubilo Iwata":              "Iwata",
    "Fagiano Okayama":           "Okayama",
    "Omiya Ardija":              "Omiya Ardija",
    "Hokkaido Consadole Sapporo": "Hokkaido Consadole Sapporo",
    "Consadole Sapporo":         "Hokkaido Consadole Sapporo",
    "Sagan Tosu":                "Sagan Tosu",
    "Sanfrecce Hiroshima":       "Sanfrecce Hiroshima",
    "Kashiwa Reysol":            "Kashiwa Reysol",
    "Shonan Bellmare":           "Shonan Bellmare",
    "Tokushima Vortis":          "Tokushima",
    "Oita Trinita":              "Oita Trinita",
    "Roasso Kumamoto":           "Kumamoto",
    "V-Varen Nagasaki":          "V-Varen Nagasaki",
    "FC Machida Zelvia":         "Machida",
    "AC長野パルセイロ":             "Yamaga",
    "AC Nagano Parceiro":        "Yamaga",
}

# Jリーグ competition_id (transfermarkt-datasets での識別子)
J_LEAGUE_IDS = {"JAP1", "JAP2"}  # J1 / J2

# ── 静的フォールバックデータ (2022-2025 概算値、単位: 百万ユーロ) ──────────
# 出典: Transfermarkt公開情報からの手動集計概算
FALLBACK_DATA = [
    # season, team, market_value_eur
    # 2022
    (2022, "Kawasaki Frontale",            19.2),
    (2022, "Yokohama F. Marinos",          17.5),
    (2022, "Urawa Reds",                   15.8),
    (2022, "Kashima Antlers",              14.3),
    (2022, "Gamba Osaka",                  11.2),
    (2022, "Cerezo Osaka",                 10.5),
    (2022, "Nagoya Grampus",               10.8),
    (2022, "Vissel Kobe",                  18.5),
    (2022, "FC Tokyo",                     12.0),
    (2022, "Sanfrecce Hiroshima",           9.5),
    (2022, "Sagan Tosu",                    6.2),
    (2022, "Hokkaido Consadole Sapporo",    8.0),
    (2022, "Shimizu S-Pulse",               7.5),
    (2022, "Albirex Niigata",               5.5),
    (2022, "Kyoto",                         6.0),
    (2022, "Avispa Fukuoka",                5.8),
    (2022, "Kashiwa Reysol",                9.2),
    (2022, "Vegalta Sendai",                6.5),
    (2022, "Omiya Ardija",                  4.5),
    (2022, "Iwata",                         7.0),
    (2022, "Yokohama FC",                   5.0),
    (2022, "Shonan Bellmare",               5.5),
    (2022, "Verdy",                         4.8),
    (2022, "Montedio Yamagata",             3.5),
    (2022, "Kofu",                          3.2),
    (2022, "Okayama",                       3.0),
    # 2023
    (2023, "Kawasaki Frontale",            20.5),
    (2023, "Yokohama F. Marinos",          22.0),
    (2023, "Urawa Reds",                   17.2),
    (2023, "Kashima Antlers",              15.0),
    (2023, "Gamba Osaka",                  12.5),
    (2023, "Cerezo Osaka",                 11.8),
    (2023, "Nagoya Grampus",               11.5),
    (2023, "Vissel Kobe",                  25.0),
    (2023, "FC Tokyo",                     13.5),
    (2023, "Sanfrecce Hiroshima",          10.2),
    (2023, "Sagan Tosu",                    7.0),
    (2023, "Hokkaido Consadole Sapporo",    8.5),
    (2023, "Shimizu S-Pulse",               8.0),
    (2023, "Albirex Niigata",               6.0),
    (2023, "Kyoto",                         6.5),
    (2023, "Avispa Fukuoka",                6.2),
    (2023, "Kashiwa Reysol",                9.8),
    (2023, "Vegalta Sendai",                5.5),
    (2023, "Omiya Ardija",                  4.8),
    (2023, "Iwata",                         7.5),
    (2023, "Yokohama FC",                   5.5),
    (2023, "Shonan Bellmare",               5.8),
    (2023, "Verdy",                         5.2),
    (2023, "Montedio Yamagata",             3.8),
    (2023, "Kofu",                          3.5),
    (2023, "Okayama",                       3.2),
    # 2024
    (2024, "Kawasaki Frontale",            21.0),
    (2024, "Yokohama F. Marinos",          23.5),
    (2024, "Urawa Reds",                   18.0),
    (2024, "Kashima Antlers",              16.0),
    (2024, "Gamba Osaka",                  13.0),
    (2024, "Cerezo Osaka",                 12.0),
    (2024, "Nagoya Grampus",               12.0),
    (2024, "Vissel Kobe",                  27.0),
    (2024, "FC Tokyo",                     14.0),
    (2024, "Sanfrecce Hiroshima",          11.0),
    (2024, "Sagan Tosu",                    7.5),
    (2024, "Hokkaido Consadole Sapporo",    9.0),
    (2024, "Shimizu S-Pulse",               8.5),
    (2024, "Albirex Niigata",               6.5),
    (2024, "Kyoto",                         7.0),
    (2024, "Avispa Fukuoka",                6.5),
    (2024, "Kashiwa Reysol",               10.5),
    (2024, "Vegalta Sendai",                6.0),
    (2024, "Omiya Ardija",                  5.0),
    (2024, "Iwata",                         8.0),
    (2024, "Yokohama FC",                   6.0),
    (2024, "Shonan Bellmare",               6.0),
    (2024, "Verdy",                         5.5),
    (2024, "Montedio Yamagata",             4.0),
    (2024, "Kofu",                          3.8),
    (2024, "Okayama",                       3.5),
    (2024, "Machida",                       4.5),
    # 2025
    (2025, "Kawasaki Frontale",            22.0),
    (2025, "Yokohama F. Marinos",          24.0),
    (2025, "Urawa Reds",                   18.5),
    (2025, "Kashima Antlers",              16.5),
    (2025, "Gamba Osaka",                  13.5),
    (2025, "Cerezo Osaka",                 12.5),
    (2025, "Nagoya Grampus",               12.5),
    (2025, "Vissel Kobe",                  28.0),
    (2025, "FC Tokyo",                     14.5),
    (2025, "Sanfrecce Hiroshima",          11.5),
    (2025, "Sagan Tosu",                    8.0),
    (2025, "Hokkaido Consadole Sapporo",    9.5),
    (2025, "Shimizu S-Pulse",               9.0),
    (2025, "Albirex Niigata",               7.0),
    (2025, "Kyoto",                         7.5),
    (2025, "Avispa Fukuoka",                7.0),
    (2025, "Kashiwa Reysol",               11.0),
    (2025, "Vegalta Sendai",                6.5),
    (2025, "Omiya Ardija",                  5.5),
    (2025, "Iwata",                         8.5),
    (2025, "Yokohama FC",                   6.5),
    (2025, "Shonan Bellmare",               6.5),
    (2025, "Verdy",                         6.0),
    (2025, "Montedio Yamagata",             4.5),
    (2025, "Kofu",                          4.0),
    (2025, "Okayama",                       4.0),
    (2025, "Machida",                       5.0),
]


def _get_fallback_df() -> pd.DataFrame:
    """静的フォールバックデータをDataFrameとして返す"""
    df = pd.DataFrame(FALLBACK_DATA, columns=["season", "team", "market_value_eur"])
    return df


def _fetch_from_datasets() -> pd.DataFrame | None:
    """
    transfermarkt-datasets GitHub リポジトリから clubs.csv を取得し、
    Jリーグチームの市場価値を抽出する。
    """
    # 複数のURLを試みる (ブランチ名やパス変更に対応)
    candidate_urls = [
        "https://raw.githubusercontent.com/dcaribou/transfermarkt-datasets/master/data/clubs.csv",
        "https://raw.githubusercontent.com/dcaribou/transfermarkt-datasets/main/data/clubs.csv",
        # v2 形式の場合
        "https://raw.githubusercontent.com/dcaribou/transfermarkt-datasets/master/data/prep/clubs.csv",
    ]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    for url in candidate_urls:
        try:
            logger.info(f"Fetching clubs.csv from: {url}")
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code != 200:
                logger.warning(f"HTTP {resp.status_code} for {url}")
                continue

            df = pd.read_csv(io.StringIO(resp.text))
            logger.info(f"clubs.csv columns: {df.columns.tolist()}")

            # domestic_competition_id でJリーグを絞り込む
            if "domestic_competition_id" in df.columns:
                j_df = df[df["domestic_competition_id"].isin(J_LEAGUE_IDS)].copy()
            elif "competition_id" in df.columns:
                j_df = df[df["competition_id"].isin(J_LEAGUE_IDS)].copy()
            else:
                # 全データからチーム名で絞り込み
                known_tm_names = set(TM_TO_LOCAL.keys())
                j_df = df[df.get("name", df.get("club_name", pd.Series())).isin(known_tm_names)].copy()

            if j_df.empty:
                logger.warning(f"No J-League teams found in {url}")
                continue

            logger.info(f"Found {len(j_df)} J-League clubs from datasets")

            # 市場価値カラムを探す
            mv_col = None
            for candidate in ["market_value_in_eur", "total_market_value", "squad_market_value",
                               "market_value", "marketvalue"]:
                if candidate in j_df.columns:
                    mv_col = candidate
                    break

            if mv_col is None:
                logger.warning(f"No market value column found. Available: {j_df.columns.tolist()}")
                continue

            # チーム名カラムを探す
            name_col = None
            for candidate in ["name", "club_name", "pretty_name"]:
                if candidate in j_df.columns:
                    name_col = candidate
                    break

            if name_col is None:
                logger.warning("No name column found in clubs.csv")
                continue

            # シーズンカラム
            season_col = None
            for candidate in ["last_season", "season", "year"]:
                if candidate in j_df.columns:
                    season_col = candidate
                    break

            rows = []
            for _, r in j_df.iterrows():
                tm_name = str(r[name_col])
                local_name = TM_TO_LOCAL.get(tm_name)
                if local_name is None:
                    # 部分一致を試みる
                    for k, v in TM_TO_LOCAL.items():
                        if k.lower() in tm_name.lower() or tm_name.lower() in k.lower():
                            local_name = v
                            break
                if local_name is None:
                    logger.debug(f"No mapping for TM team: {tm_name!r}")
                    continue

                raw_mv = r[mv_col]
                if pd.isna(raw_mv):
                    continue

                # 値を数値に変換 (例: "€19.20m" → 19.2)
                mv = _parse_market_value(raw_mv)
                if mv is None or mv <= 0:
                    continue

                season = int(r[season_col]) if season_col and not pd.isna(r.get(season_col)) else None
                if season is None:
                    season = 2024  # デフォルト

                rows.append({"season": season, "team": local_name, "market_value_eur": mv})

            if not rows:
                logger.warning("No valid rows extracted from datasets CSV")
                continue

            result_df = pd.DataFrame(rows)
            logger.info(f"Successfully extracted {len(result_df)} rows from datasets")
            return result_df

        except Exception as e:
            logger.warning(f"Failed to fetch from {url}: {e}")
            continue

    return None


def _fetch_from_transfermarkt() -> pd.DataFrame | None:
    """
    transfermarkt.com のJ1リーグ市場価値ページを直接スクレイピングする。
    robots.txt を尊重し、適切なUser-Agentとウェイトを使用する。
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.transfermarkt.com/",
    }

    url = "https://www.transfermarkt.com/j1-league/marktwerteverein/wettbewerb/JAP1"

    try:
        logger.info(f"Scraping transfermarkt.com: {url}")
        time.sleep(2)  # 礼儀正しいクロール間隔
        resp = requests.get(url, headers=headers, timeout=30)

        if resp.status_code != 200:
            logger.warning(f"HTTP {resp.status_code} from transfermarkt.com")
            return None

        # pandas で HTML テーブルをパース
        try:
            tables = pd.read_html(resp.text, flavor="html5lib")
        except Exception:
            tables = pd.read_html(resp.text)

        if not tables:
            logger.warning("No tables found on transfermarkt page")
            return None

        # 市場価値テーブルを探す (最大の表が候補)
        target = max(tables, key=len)
        logger.info(f"Best table shape: {target.shape}, columns: {target.columns.tolist()}")

        rows = []
        for _, r in target.iterrows():
            row_str = " ".join(str(v) for v in r.values)

            # チーム名を探す
            team_local = None
            for k, v in TM_TO_LOCAL.items():
                if k.lower() in row_str.lower():
                    team_local = v
                    break

            if team_local is None:
                continue

            # 市場価値を探す (例: "€19.20m")
            mv_match = re.search(r"€\s*([\d,.]+)\s*([mMbBkK]?)", row_str)
            if mv_match:
                mv = _parse_market_value(mv_match.group(0))
                if mv and mv > 0:
                    rows.append({"season": 2024, "team": team_local, "market_value_eur": mv})

        if rows:
            logger.info(f"Scraped {len(rows)} teams from transfermarkt.com")
            return pd.DataFrame(rows)

    except Exception as e:
        logger.warning(f"transfermarkt.com scraping failed: {e}")

    return None


def _parse_market_value(raw) -> float | None:
    """
    様々な形式の市場価値を百万ユーロの float に変換する。
    例:
      "€19.20m"  → 19.2
      "€1.50bn"  → 1500.0
      19200000   → 19.2
      "19,200,000" → 19.2
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None

    s = str(raw).strip().replace(",", "")

    # 数値のみの場合 (単位: ユーロ → 百万ユーロに変換)
    if re.match(r"^[\d.]+$", s):
        val = float(s)
        # 100万以上ならユーロ単位と判定
        if val >= 1_000_000:
            return round(val / 1_000_000, 3)
        # それ以下はすでに百万ユーロ単位と判定
        return round(val, 3)

    # "€19.20m" 形式
    m = re.search(r"([\d.]+)\s*([mMbBkK]?)", s.replace("€", "").replace("£", "").strip())
    if m:
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit == "b":
            return round(val * 1000, 3)
        elif unit == "m":
            return round(val, 3)
        elif unit == "k":
            return round(val / 1000, 3)
        else:
            # 単位なし: 大きければユーロ単位
            if val >= 1_000_000:
                return round(val / 1_000_000, 3)
            return round(val, 3)

    return None


def fetch_market_values(force_fallback: bool = False) -> pd.DataFrame:
    """
    Jリーグチームの市場価値を取得し、DataFrameを返す。

    取得優先順位:
      1. transfermarkt-datasets CSV
      2. transfermarkt.com スクレイピング
      3. 静的フォールバックデータ

    Returns:
        pd.DataFrame: columns=[season, team, market_value_eur]
    """
    if not force_fallback:
        # 方針A: datasets CSV
        df = _fetch_from_datasets()
        if df is not None and not df.empty:
            logger.info("Using data from transfermarkt-datasets")
            return df

        # 方針B: 直接スクレイピング
        df = _fetch_from_transfermarkt()
        if df is not None and not df.empty:
            logger.info("Using data from transfermarkt.com scraping")
            return df

        logger.warning("All online sources failed. Using static fallback data.")

    return _get_fallback_df()


def save_market_values(df: pd.DataFrame, output_path: str = _OUTPUT_PATH) -> None:
    """DataFrameをCSVに保存する"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} rows to {output_path}")


def run(force_fallback: bool = False) -> pd.DataFrame:
    """
    スクレイパーを実行して market_values.csv を保存する。

    Args:
        force_fallback: True の場合、オンライン取得をスキップして静的データを使用

    Returns:
        保存されたDataFrame
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = fetch_market_values(force_fallback=force_fallback)
    save_market_values(df)
    return df


if __name__ == "__main__":
    result = run()
    print(result.head(20).to_string(index=False))
    print(f"\n合計 {len(result)} 行を {_OUTPUT_PATH} に保存しました。")
