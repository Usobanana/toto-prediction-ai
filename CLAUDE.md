# toto-prediction-ai プロジェクトルール

## ブランチ
- **常に `main` ブランチで作業する**（worktree は使わない）

## toto 予想ワークフロー

### 予想実行時の必須手順

第N回の予想を行うときは、以下を必ず実行すること：

1. **スクリプト作成**: `predict_XXXX.py`（シングル）と `predict_XXXX_multi.py`（マルチ）
2. **MD保存**: 予想結果を `predictions/` フォルダに保存
   - `predictions/XXXX_single.md` — シングル予想（1試合1択）
   - `predictions/XXXX_multi.md` — マルチ予想（複数択・組み合わせ）

### MD ファイルの必須記載項目

**シングル予想 (`XXXX_single.md`)**
- 開催回・予想日・モデル名・バックテスト正答率
- 全13試合の予想（試合番号 / ホーム / アウェイ / 予想 / 確率1/0/2 / 推定方法）
- 予想配列 `[N/N/N/...]`
- toto公式対戦データ取得分（gameid・H2H成績）

**マルチ予想 (`XXXX_multi.md`)**
- 各試合の確率・不確実フラグ
- 予算別プラン比較（1,000 / 3,200 / 6,400円）
- 推奨プラン（3,200円）の選択詳細と全組み合わせリスト
- 複数択にした理由

### ファイル命名規則
```
predictions/
  1624_single.md
  1624_multi.md
  1625_single.md
  1625_multi.md
  ...
```

## データ収集ルール

- toto公式対戦データ（rival_team_scraper）は予想時に都度取得する
  ```bash
  python src/scraper/rival_team_scraper.py --hold-cnt-id XXXX --seed-gameid XXXXX
  ```
- 投票率データ（vote_rates）は販売終了後に取得できる。取得後はマルチ予想を再実行するとエッジ連動選定になる

## モデル

- **シングル予想**: RandomForest + Optuna（61特徴量、バックテスト 48.15%）
- **マルチ予想**: 同上（Edge-Aware 最適化）
- 前回スクリプト（`predict_XXXX.py`）を雛形として次回分を作成する
