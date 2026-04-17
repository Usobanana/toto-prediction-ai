# -*- coding: utf-8 -*-
"""
Open-Meteo 過去気象データ取得
==============================
無料・APIキー不要。スタジアム座標と試合日から天気特徴量を取得してCSVに保存。

使い方:
  python -m src.scraper.weather_fetcher

出力:
  data/raw/match_weather.csv  (date, home_team, temp_avg, precip_sum, wind_max)
"""

import io, sys, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# 試合は通常14:00〜19:00帯。その時間帯のデータを平均する。
MATCH_HOURS = list(range(12, 20))   # 12〜19時


def _load_stadium_coords() -> dict[str, tuple[float, float]]:
    _dir = os.path.dirname(os.path.abspath(__file__))
    _csv = os.path.join(_dir, "..", "..", "data", "stadium_coords.csv")
    df = pd.read_csv(_csv)
    return {row["team"]: (float(row["lat"]), float(row["lon"])) for _, row in df.iterrows()}


def fetch_weather(lat: float, lon: float, date_str: str) -> dict | None:
    """
    指定座標・日付の気象データを取得。
    戻り値: {"temp_avg": float, "precip_sum": float, "wind_max": float} or None
    """
    if not HAS_REQUESTS:
        return None
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": "Asia/Tokyo",
    }
    try:
        resp = requests.get(ARCHIVE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        times  = hourly.get("time", [])
        temps  = hourly.get("temperature_2m", [])
        precip = hourly.get("precipitation", [])
        winds  = hourly.get("wind_speed_10m", [])

        # 試合時間帯 (12〜19時) のデータのみ抽出
        match_temps, match_precip, match_winds = [], [], []
        for i, t in enumerate(times):
            hour = int(t[11:13])
            if hour in MATCH_HOURS:
                if i < len(temps)  and temps[i]  is not None: match_temps.append(temps[i])
                if i < len(precip) and precip[i] is not None: match_precip.append(precip[i])
                if i < len(winds)  and winds[i]  is not None: match_winds.append(winds[i])

        return {
            "temp_avg":   float(np.mean(match_temps))  if match_temps  else np.nan,
            "precip_sum": float(np.sum(match_precip))  if match_precip else np.nan,
            "wind_max":   float(np.max(match_winds))   if match_winds  else np.nan,
        }
    except Exception as e:
        print(f"  [WARN] weather fetch failed ({lat},{lon},{date_str}): {e}")
        return None


def build_weather_dataset(
    results_csv: str = "data/raw/jleague_results.csv",
    out_csv: str     = "data/raw/match_weather.csv",
    sleep_sec: float = 0.3,
) -> pd.DataFrame:
    """
    試合結果CSVを読み込み、各試合のホームスタジアム天気を取得してCSV保存。
    既に取得済みの試合はスキップ (増分取得)。
    """
    coords = _load_stadium_coords()

    df = pd.read_csv(results_csv)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # 既存データを読み込み
    if os.path.exists(out_csv):
        existing = pd.read_csv(out_csv, parse_dates=["date"])
        done_keys = set(
            zip(existing["date"].dt.strftime("%Y-%m-%d"), existing["home_team"])
        )
        rows = existing.to_dict("records")
        print(f"  既存: {len(existing)}件を読み込み。増分取得します。")
    else:
        done_keys = set()
        rows = []

    total   = len(df)
    skipped = 0
    fetched = 0
    errors  = 0

    for i, row in df.iterrows():
        home      = row["home_team"]
        date_str  = row["date"].strftime("%Y-%m-%d")
        key       = (date_str, home)

        if key in done_keys:
            skipped += 1
            continue

        coord = coords.get(home)
        if coord is None:
            errors += 1
            print(f"  [SKIP] 座標なし: {home}")
            continue

        print(f"  [{i+1}/{total}] {date_str} {home}... ", end="", flush=True)
        w = fetch_weather(coord[0], coord[1], date_str)
        if w:
            rows.append({
                "date":       date_str,
                "home_team":  home,
                "away_team":  row["away_team"],
                "temp_avg":   round(w["temp_avg"], 1),
                "precip_sum": round(w["precip_sum"], 1),
                "wind_max":   round(w["wind_max"], 1),
            })
            done_keys.add(key)
            fetched += 1
            print(f"気温{w['temp_avg']:.1f}℃ 降水{w['precip_sum']:.1f}mm 風{w['wind_max']:.1f}m/s")
        else:
            errors += 1
            print("失敗")

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date").reset_index(drop=True)
        out.to_csv(out_csv, index=False)
        print(f"\n  保存: {out_csv}  (全{len(out)}件 / 今回取得{fetched}件 / スキップ{skipped}件)")
    return out


if __name__ == "__main__":
    build_weather_dataset()
