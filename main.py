"""
toto予想AI - メイン実行スクリプト

使い方:
  python main.py collect          # データ収集 (J1リーグ + toto結果)
  python main.py backtest         # バックテスト実行 (全モデル)
  python main.py predict          # 次回toto予想
  python main.py all              # 上記すべて実行
"""

import sys
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path("reports")

for d in [RAW_DIR, PROCESSED_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 1. データ収集
# ─────────────────────────────────────────────

def collect_data():
    """J1リーグ試合データを football-data.co.uk から収集"""
    from src.scraper.jleague_scraper import FootballDataScraper

    logger.info("=== データ収集開始 (football-data.co.uk) ===")
    scraper = FootballDataScraper()
    data = scraper.fetch()

    if not data:
        logger.error("データ取得失敗")
        return None

    output = RAW_DIR / "jleague_results.csv"
    scraper.save(data, output)
    logger.info(f"収集完了: {len(data)}試合 → {output}")
    return output


# ─────────────────────────────────────────────
# 2. データ前処理
# ─────────────────────────────────────────────

def preprocess(raw_path: Path = None) -> pd.DataFrame:
    """生データを読み込み、特徴量を生成"""
    from src.features.feature_builder import FeatureBuilder

    if raw_path is None:
        raw_path = RAW_DIR / "jleague_results.csv"

    if not raw_path.exists():
        logger.error(f"データファイルが見つかりません: {raw_path}")
        logger.info("先に `python main.py collect` を実行してください")
        return pd.DataFrame()

    df = pd.read_csv(raw_path)
    logger.info(f"読み込み: {len(df)}試合 ({raw_path})")

    # 基本フィルタ
    df = df.dropna(subset=["home_team", "away_team", "result"])
    df = df[df["result"].astype(str).isin(["1", "0", "2"])]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(f"前処理後: {len(df)}試合")

    # 特徴量生成
    builder = FeatureBuilder(form_window=5)
    feature_df = builder.build(df)

    out_path = PROCESSED_DIR / "features.parquet"
    feature_df.to_parquet(out_path, index=False)
    logger.info(f"特徴量保存: {out_path}")

    return feature_df


# ─────────────────────────────────────────────
# 3. バックテスト
# ─────────────────────────────────────────────

def run_backtest(feature_df: pd.DataFrame = None):
    """全モデルをバックテスト"""
    from src.models.baseline import MostFrequentModel, HomeWinModel, TeamWinRateModel, OddsModel
    from src.models.ml_models import (
        LogisticRegressionModel, RandomForestModel, XGBoostModel, LightGBMModel
    )
    from src.features.feature_builder import get_feature_columns
    from src.evaluation.backtest import Backtester, save_results, print_comparison_table

    if feature_df is None:
        feat_path = PROCESSED_DIR / "features.parquet"
        if not feat_path.exists():
            logger.info("特徴量ファイルなし → 前処理を実行します")
            feature_df = preprocess()
        else:
            feature_df = pd.read_parquet(feat_path)

    if feature_df is None or len(feature_df) == 0:
        logger.error("特徴量データなし。処理を中断します。")
        return

    # オッズあり/なし両方の特徴量
    feature_cols_no_odds = get_feature_columns(include_odds=False)
    feature_cols_with_odds = get_feature_columns(include_odds=True)

    available_no_odds = [c for c in feature_cols_no_odds if c in feature_df.columns]
    available_with_odds = [c for c in feature_cols_with_odds if c in feature_df.columns]
    logger.info(f"特徴量(オッズなし): {len(available_no_odds)}個, (オッズあり): {len(available_with_odds)}個")

    # オッズなしモデル
    models_no_odds = [
        MostFrequentModel(),
        HomeWinModel(),
        TeamWinRateModel(),
        LogisticRegressionModel(),
        RandomForestModel(),
    ]

    # XGBoost/LightGBM は利用可能な場合のみ
    try:
        models_no_odds.append(XGBoostModel())
    except ImportError:
        logger.warning("XGBoost未インストール → スキップ")

    try:
        models_no_odds.append(LightGBMModel())
    except ImportError:
        logger.warning("LightGBM未インストール → スキップ")

    backtester = Backtester(n_splits=5, min_train_size=100)

    # オッズなしモデルのバックテスト
    logger.info("\n=== オッズなし特徴量でバックテスト ===")
    results = backtester.run_all(models_no_odds, feature_df, available_no_odds)

    # オッズあり (オッズモデル + RFにオッズ特徴量追加)
    if len(available_with_odds) > len(available_no_odds):
        logger.info("\n=== オッズあり特徴量でバックテスト ===")
        # オッズがある行のみ
        odds_df = feature_df.dropna(subset=["odds_home_avg"])
        if len(odds_df) > 200:
            odds_models = [
                OddsModel(),
                RandomForestModel(),
            ]
            odds_models[1].name = "rf_with_odds"
            odds_results = backtester.run_all(odds_models, odds_df, available_with_odds)
            results.update(odds_results)

    save_results(results)
    print_comparison_table(results)

    return results


# ─────────────────────────────────────────────
# 4. 次回toto予想
# ─────────────────────────────────────────────

def predict_next(matches: list[dict] = None):
    """
    次回totoの試合を予想する

    Parameters
    ----------
    matches : [{"home_team": "...", "away_team": "..."}, ...]
        予想したい試合一覧。None の場合はサンプルを使用。
    """
    from src.models.ml_models import RandomForestModel
    from src.features.feature_builder import FeatureBuilder, get_feature_columns
    import numpy as np

    feat_path = PROCESSED_DIR / "features.parquet"
    raw_path = RAW_DIR / "jleague_results.csv"

    if not feat_path.exists() or not raw_path.exists():
        logger.error("データ未収集。`python main.py collect` を実行してください")
        return

    # 全履歴で学習
    feature_df = pd.read_parquet(feat_path)
    feature_cols = [c for c in get_feature_columns() if c in feature_df.columns]

    X_all = feature_df[feature_cols].fillna(0)
    y_all = feature_df["result"].astype(str)

    model = RandomForestModel()
    model.fit(X_all, y_all)

    # 予想対象の試合
    if matches is None:
        # サンプル試合 (実際の次回開催カードに差し替えること)
        matches = [
            {"home_team": "Vissel Kobe", "away_team": "Kashima Antlers"},
            {"home_team": "Kawasaki Frontale", "away_team": "Yokohama F.Marinos"},
            {"home_team": "Gamba Osaka", "away_team": "Urawa Red Diamonds"},
        ]

    # 直近Eloを再計算して予想用特徴量を作成
    raw_df = pd.read_csv(raw_path)
    raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
    raw_df = raw_df.dropna(subset=["date"])
    raw_df = raw_df.sort_values("date").reset_index(drop=True)

    builder = FeatureBuilder(form_window=5)
    hist_df = builder.build(raw_df)

    # 各チームの最新特徴量を取得 (最後の出場試合の特徴量を代用)
    print("\n=== 次回toto予想 ===")
    print(f"{'試合':<45} {'予想':>6} {'確率(1/0/2)'}")
    print("-" * 70)

    for m in matches:
        home = m["home_team"]
        away = m["away_team"]

        # ホームチームの最新フォーム特徴量を取得
        home_rows = hist_df[hist_df["home_team"] == home]
        away_rows = hist_df[hist_df["away_team"] == away]

        if home_rows.empty or away_rows.empty:
            print(f"{home} vs {away:<20} データ不足のため予想不可")
            continue

        # 最新行から特徴量を取得して予想用dfを作成
        feat_row = home_rows.iloc[-1][feature_cols].copy()
        feat_df_pred = pd.DataFrame([feat_row])

        pred = model.predict(feat_df_pred)[0]
        proba = model.predict_proba(feat_df_pred)[0]

        label = {"1": "ホーム勝", "0": "引き分け", "2": "アウェイ勝"}.get(pred, pred)
        print(
            f"{home} vs {away:<20} "
            f"【{label}】  "
            f"1:{proba[0]:.2f} / 0:{proba[1]:.2f} / 2:{proba[2]:.2f}"
        )


# ─────────────────────────────────────────────
# エントリーポイント
# ─────────────────────────────────────────────

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"

    if cmd == "collect":
        collect_data()

    elif cmd == "preprocess":
        preprocess()

    elif cmd == "backtest":
        run_backtest()

    elif cmd == "predict":
        predict_next()

    elif cmd == "all":
        logger.info("=== 全処理実行 ===")
        raw_path = collect_data()
        feature_df = preprocess(raw_path)
        if feature_df is not None and len(feature_df) > 0:
            run_backtest(feature_df)
            predict_next()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
