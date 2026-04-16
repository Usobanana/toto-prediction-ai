# -*- coding: utf-8 -*-
"""
generate_pages.py
=================
GitHub Pages 用データを更新するスクリプト。

使い方:
  # 結果を記録する (第1622回の結果が「1,0,2,1,1,2,1,2,2,1,1,2,2」の場合)
  python generate_pages.py record 1622 1,0,2,1,1,2,1,2,2,1,1,2,2

  # 新しいラウンドの予想を追加する
  python generate_pages.py add-round

  # データファイルの確認
  python generate_pages.py status
"""
import sys
import json
from pathlib import Path
from datetime import date

ROUNDS_JSON = Path("docs/data/rounds.json")


def load_data():
    with open(ROUNDS_JSON, encoding="utf-8") as f:
        return json.load(f)


def save_data(data):
    with open(ROUNDS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[保存完了] {ROUNDS_JSON}")


def record_result(round_no: int, results: list[str]):
    """実際の結果を記録し、正答数を計算する"""
    data = load_data()

    target = None
    for r in data["rounds"]:
        if r["round"] == round_no:
            target = r
            break

    if target is None:
        print(f"[エラー] 第{round_no}回のデータが見つかりません")
        sys.exit(1)

    if len(results) != len(target["matches"]):
        print(f"[エラー] 結果の数が一致しません: {len(results)} != {len(target['matches'])}")
        sys.exit(1)

    # 結果を記録
    correct = 0
    print(f"\n第{round_no}回 振り返り")
    print("=" * 60)
    print(f"{'No':>2}  {'試合':<20}  {'予想':^6}  {'結果':^6}  {'判定'}")
    print("-" * 60)

    for i, (match, result) in enumerate(zip(target["matches"], results)):
        match["result"] = result
        pred = match["single"]
        ok = pred == result
        if ok:
            correct += 1

        pred_labels = {"1": "ホーム勝", "0": "引き分け", "2": "アウェイ勝"}
        mark = "○" if ok else "×"
        print(
            f"{match['no']:>2}. {match['home']} vs {match['away']:<12}"
            f"  {pred_labels.get(pred, pred):^8}  {pred_labels.get(result, result):^8}  {mark}"
        )

    target["correct"] = correct
    target["total"] = len(target["matches"])
    target["accuracy"] = round(correct / len(target["matches"]), 4)
    target["status"] = "done"

    print("-" * 60)
    acc_pct = round(correct / len(target["matches"]) * 100, 1)
    print(f"  正答数: {correct} / {len(target['matches'])} ({acc_pct}%)")
    print()

    save_data(data)
    print("[次のステップ] git add docs/ && git commit -m 'result: 第{round_no}回結果記録' && git push")


def show_status():
    """現在の状態を表示"""
    data = load_data()
    print("\n=== toto予想AI - データ状態 ===")
    print(f"ラウンド数: {len(data['rounds'])}")
    print()
    for r in data["rounds"]:
        status = r["status"]
        if status == "done":
            pct = round(r["correct"] / r["total"] * 100, 1)
            print(f"  第{r['round']}回 [{status}] {r['correct']}/{r['total']} ({pct}%) - {r['match_date']}")
        else:
            print(f"  第{r['round']}回 [{status}] 予想済み・結果待ち - {r['match_date']}")
    print()
    print(f"GitHub Pages URL: https://usobanana.github.io/toto-prediction-ai/")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "record":
        if len(sys.argv) < 4:
            print("使い方: python generate_pages.py record <ラウンド番号> <結果（カンマ区切り）>")
            print("例: python generate_pages.py record 1622 1,0,2,1,1,2,1,2,2,1,1,2,2")
            sys.exit(1)
        round_no = int(sys.argv[2])
        results = sys.argv[3].split(",")
        record_result(round_no, results)

    elif cmd == "status":
        show_status()

    else:
        print(f"不明なコマンド: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
