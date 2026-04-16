"""
バックテストフレームワーク

・時系列スプリット: 過去データで学習→未来データで評価 (リーク防止)
・各モデルの正答率, クラス別精度, toto全的中率を計算
・結果をCSV/JSONで保存
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class BacktestResult:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.fold_results: list[dict] = []

    def add_fold(self, fold: int, y_true, y_pred, split_date: str = ""):
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true, y_pred, labels=["1", "0", "2"],
            target_names=["ホーム勝(1)", "引き分け(0)", "アウェイ勝(2)"],
            output_dict=True,
            zero_division=0,
        )
        self.fold_results.append({
            "fold": fold,
            "split_date": split_date,
            "accuracy": acc,
            "n_samples": len(y_true),
            "class_report": report,
            "y_true": list(y_true),
            "y_pred": list(y_pred),
        })

    @property
    def mean_accuracy(self) -> float:
        if not self.fold_results:
            return 0.0
        return np.mean([f["accuracy"] for f in self.fold_results])

    def summary(self) -> dict:
        accs = [f["accuracy"] for f in self.fold_results]
        return {
            "model": self.model_name,
            "n_folds": len(self.fold_results),
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "min_accuracy": float(np.min(accs)),
            "max_accuracy": float(np.max(accs)),
            "fold_details": [
                {"fold": f["fold"], "accuracy": f["accuracy"],
                 "n_samples": f["n_samples"], "split_date": f["split_date"]}
                for f in self.fold_results
            ],
        }

    def print_summary(self):
        s = self.summary()
        print(f"\n{'='*60}")
        print(f"モデル: {s['model']}")
        print(f"{'='*60}")
        print(f"平均正答率: {s['mean_accuracy']:.3f} ± {s['std_accuracy']:.3f}")
        print(f"最小/最大:  {s['min_accuracy']:.3f} / {s['max_accuracy']:.3f}")
        print(f"評価フォルド数: {s['n_folds']}")
        for fd in s["fold_details"]:
            print(f"  Fold {fd['fold']+1}: acc={fd['accuracy']:.3f} (n={fd['n_samples']}, from={fd['split_date']})")


class Backtester:
    """
    時系列ウォークフォワードバックテスト

    Parameters
    ----------
    n_splits : int
        時系列分割数
    min_train_size : int
        最低限必要な訓練サンプル数
    """

    def __init__(self, n_splits: int = 5, min_train_size: int = 200):
        self.n_splits = n_splits
        self.min_train_size = min_train_size

    def run(
        self,
        model: BaseModel,
        feature_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> BacktestResult:
        """
        Parameters
        ----------
        feature_df : 特徴量付きDataFrame (date, result 列を含む)
        feature_cols : 学習に使う特徴量列
        """
        df = feature_df.dropna(subset=["result"]).copy()
        df = df.sort_values("date").reset_index(drop=True)

        X = df[feature_cols + ["home_team", "away_team"]].copy()
        y = df["result"].astype(str)

        result = BacktestResult(model.name)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            if len(train_idx) < self.min_train_size:
                logger.info(f"Fold {fold+1}: 訓練データ不足({len(train_idx)}件) → スキップ")
                continue

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            split_date = str(df["date"].iloc[test_idx[0]])[:10]

            logger.info(
                f"[{model.name}] Fold {fold+1}: "
                f"train={len(train_idx)}, test={len(test_idx)}, split={split_date}"
            )

            # チーム列を含む拡張Xを渡す (TeamWinRateModelなど用)
            train_cols = feature_cols + [c for c in ["home_team", "away_team"] if c in X_train.columns]
            model.fit(X_train[train_cols], y_train)
            y_pred = model.predict(X_test[train_cols])

            result.add_fold(fold, y_test.tolist(), y_pred.tolist(), split_date)

        return result

    def run_all(
        self,
        models: list[BaseModel],
        feature_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> dict[str, BacktestResult]:
        results = {}
        for model in models:
            logger.info(f"\n▶ バックテスト開始: {model.name}")
            res = self.run(model, feature_df, feature_cols)
            res.print_summary()
            results[model.name] = res
        return results


def save_results(results: dict[str, BacktestResult], path: Optional[Path] = None):
    if path is None:
        path = REPORTS_DIR / "backtest_results.json"
    summaries = {name: r.summary() for name, r in results.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    logger.info(f"結果保存: {path}")


def print_comparison_table(results: dict[str, BacktestResult]):
    """モデル比較テーブルを表示"""
    rows = []
    for name, r in results.items():
        s = r.summary()
        rows.append({
            "モデル": s["model"],
            "平均正答率": f"{s['mean_accuracy']:.4f}",
            "標準偏差": f"{s['std_accuracy']:.4f}",
            "最大": f"{s['max_accuracy']:.4f}",
        })

    df = pd.DataFrame(rows).sort_values("平均正答率", ascending=False)
    print("\n" + "="*60)
    print("【モデル比較】")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
