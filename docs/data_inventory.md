# データ収集一覧

**最終更新:** 2026-04-21

---

## 1. 収集済みデータファイル

| ファイル | 件数 | 用途 |
|---|---|---|
| `data/raw/jleague_results.csv` | 4,523試合 | J-League試合結果・オッズ |
| `data/raw/match_weather.csv` | 1,133試合 | 試合時の気象データ |
| `data/raw/team_market_values.csv` | 69行 | チーム市場価値（€M） |
| `data/raw/toto_actual_results.csv` | 377回 | toto公式抽選結果 |
| `data/raw/toto_vote_rates.csv` | 377回 | toto投票割合 |
| `data/raw/stadium_coords.csv` | 20+チーム | スタジアムGPS座標 |
| `data/processed/features.parquet` | 4,523試合 | 学習用特徴量（50+項目） |

---

## 2. データソース別詳細

### A. J-League試合結果
**スクレイパー:** `src/scraper/jleague_scraper.py`

#### プライマリ: football-data.co.uk
- URL: `https://www.football-data.co.uk/new/JPN.csv`
- 取得項目:
  - `date` — 試合日
  - `season` — シーズン年（2012年〜現在）
  - `home_team` / `away_team` — チーム名
  - `home_score` / `away_score` — 得点
  - `result` — 結果（1=ホーム勝, 0=引分, 2=アウェイ勝）
  - `odds_home_avg` / `odds_draw_avg` / `odds_away_avg` — 平均オッズ

#### バックアップ: TheSportsDB API
- URL: `https://www.thesportsdb.com/api/v1/json/3`（リーグID: 4633）
- 追加項目: `event_id`, `venue`, `round`

---

### B. 気象データ
**スクレイパー:** `src/scraper/weather_fetcher.py`

**ソース:** Open-Meteo Archive API（無料・API Key不要）
- URL: `https://archive-api.open-meteo.com/v1/archive`
- 取得項目:
  - `temp_avg` — 試合時間帯の平均気温（°C）
  - `precip_sum` — 降水量合計（mm）
  - `wind_max` — 最大風速（m/s）
- 試合時間帯: 12:00〜19:00 JST
- スタジアム座標は `stadium_coords.csv` を使用

---

### C. チーム市場価値
**スクレイパー:** `src/scraper/transfermarkt_scraper.py`

3段階フォールバック:

| 優先度 | ソース | 方法 |
|---|---|---|
| 1 | GitHub (dcaribou/transfermarkt-datasets) | CSVダウンロード |
| 2 | transfermarkt.com | HTMLスクレイピング |
| 3 | スタティックデータ | ハードコード値（2022-2025） |

- 取得項目: `season`, `team`, `market_value_eur`（総スカッド価値・€M）

---

### D. toto公式抽選結果
**スクレイパー:** `src/scraper/toto_scraper.py`

**ソース:** store.toto-dream.com
- 取得項目:
  - `hold_cnt_id` — 抽選回
  - `match_no` — カード番号（1〜13）
  - `date` — 試合日
  - `home_team` / `away_team` — チーム名（日本語）
  - `home_score` / `away_score` — 得点
  - `result` — 結果

---

### E. toto投票割合
**スクレイパー:** `src/scraper/vote_rate_scraper.py`

**ソース:** store.toto-dream.com
- 取得項目:
  - `hold_cnt_id` — 抽選回
  - `match_no` — カード番号
  - `vote_rate_1` — ホーム勝ち票率（%）
  - `vote_rate_0` — 引き分け票率（%）
  - `vote_rate_2` — アウェイ勝ち票率（%）

---

### F. スタジアム座標
**ソース:** 静的データ（手動収集）

- 取得項目: `team`, `stadium`, `lat`, `lon`
- 20+クラブのGPS座標を収録
- 用途: 気象データ取得・アウェイ移動距離計算

---

## 3. 学習用特徴量（50+項目）

### Eloレーティング・チーム強度
| 特徴量 | 説明 |
|---|---|
| `home_elo` / `away_elo` | Eloレーティング（K=38, 初期値1500） |
| `elo_diff` / `elo_diff_abs` | Elo差 |
| `elo_prob_home` | EloモデルによるHP勝率 |
| `market_value_home` / `away` | スカッド市場価値（€M） |
| `market_value_log_ratio` | 市場価値の対数比 |

### 直近フォーム
| 特徴量 | 説明 |
|---|---|
| `home/away_form_win_rate` | 直近5試合の勝率 |
| `home/away_form_win_rate_3` | 直近3試合の勝率 |
| `home/away_form_draw_rate` | 直近5試合の引分率 |
| `home/away_form_goals_for_avg` | 直近5試合の平均得点 |
| `home/away_form_goals_against_avg` | 直近5試合の平均失点 |
| `form_momentum_home/away` | 3試合vs5試合勝率のトレンド |

### ホーム/アウェイ別成績
| 特徴量 | 説明 |
|---|---|
| `home_home_win_rate` | ホームでの勝率 |
| `away_away_win_rate` | アウェイでの勝率 |
| `home_home_goals_for/against_avg` | ホームでの平均得失点 |
| `away_away_goals_for/against_avg` | アウェイでの平均得失点 |

### 直接対決（H2H）
| 特徴量 | 説明 |
|---|---|
| `h2h_home_win_rate` | 直近5回の対戦勝率 |
| `h2h_draw_rate` | 直近5回の引分率 |
| `h2h_count` | 対戦回数 |

### シーズン順位
| 特徴量 | 説明 |
|---|---|
| `standings_pts_home/away` | 試合前の勝点 |
| `standings_pts_diff` | 勝点差 |
| `standings_rank_home/away` | 順位 |
| `standings_ppg_home/away` | 1試合あたり勝点 |
| `relgap_home/away` | 降格圏との勝点差 |

### 気象・環境
| 特徴量 | 説明 |
|---|---|
| `temp_avg` | 試合時間帯の平均気温（°C） |
| `precip_sum` | 降水量（mm） |
| `wind_max` | 最大風速（m/s） |
| `away_travel_km` | アウェイチームの移動距離（km） |
| `rest_days_home/away` | 前試合からの休息日数 |
| `fatigue_home/away` | 直近14日の試合数 |

### 引き分け特化
| 特徴量 | 説明 |
|---|---|
| `both_draw_rate` | 両チームの平均引分率 |
| `both_venue_draw_rate` | ホーム/アウェイ別引分率の平均 |
| `attack_balance` | 攻撃力の均衡度（0=均衡, 1=差あり） |

### オッズ
| 特徴量 | 説明 |
|---|---|
| `odds_home/draw/away_avg` | 平均ブックメーカーオッズ |
| `implied_prob_home/draw/away` | オッズから算出した勝率 |
| `odds_overround` | ブックメーカーマージン |
| `elo_odds_diff` | EloモデルとオッズのHP勝率差 |

### 時系列
| 特徴量 | 説明 |
|---|---|
| `season_progress` | シーズン進行率（0〜1） |
| `games_played` | チームの累計試合数 |

---

## 4. 外部API一覧

| API | 認証 | 用途 |
|---|---|---|
| football-data.co.uk | 不要 | J-League試合結果 |
| TheSportsDB | 不要（レート制限あり） | 試合結果バックアップ |
| Open-Meteo Archive | 不要（無料） | 過去気象データ |
| transfermarkt.com | 不要（スクレイピング） | 市場価値 |
| GitHub Raw (Transfermarkt datasets) | 不要 | 市場価値バックアップ |
| store.toto-dream.com | 不要（スクレイピング） | toto結果・投票割合 |

---

## 5. データカバレッジ

| データ | 期間 | 件数 |
|---|---|---|
| J-League試合結果 | 2012年〜現在 | 4,523試合 |
| 気象データ | 部分的（約25%） | 1,133試合 |
| チーム市場価値 | 2022〜2025年 | 27〜30チーム/年 |
| toto抽選結果 | 約7年分 | 377回 |
| toto投票割合 | 同上 | 377回 |

---

## 6. データ収集フロー

```
collect_data()
  │
  ├─ 1. FootballDataScraper → jleague_results.csv
  │       └─ 試合結果・オッズ
  │
  ├─ 2. build_weather_dataset() → match_weather.csv
  │       └─ stadium_coords.csv + Open-Meteo API（差分取得）
  │
  ├─ 3. fetch_market_values() → team_market_values.csv
  │       └─ Transfermarkt-datasets → 直接スクレイピング → スタティック
  │
  ├─ 4. TotoScraper → toto_actual_results.csv
  │       └─ toto-dream.com 抽選結果
  │
  └─ 5. VoteRateScraper → toto_vote_rates.csv
          └─ toto-dream.com 投票割合

preprocess()
  └─ FeatureBuilder.build() → features.parquet
          └─ 50+特徴量を時系列リーク防止で計算
```
