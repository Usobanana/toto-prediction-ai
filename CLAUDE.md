# toto-prediction-ai プロジェクトルール

## ブランチ
- **常に `main` ブランチで作業する**（worktree は使わない）

## toto 予想ワークフロー

### 予想実行時の必須手順

第N回の予想を行うときは、以下を必ず実行すること：

1. **最新情報収集**: `predictions/XXXX_info.md` を作成
   - SP toto-dream.com の「その他データ」を全13試合分 WebFetch で取得
     ```
     URL: https://sp.toto-dream.com/dcs/subos/screen/si01/ssin025/PGSSIN02501RivalTeamTotoSP.form?holdCntId=XXXX&commodityId=01&gameId=YYYYY
     ```
   - 各試合のリーグ（J1/J2/J3）・直近フォーム・今季H2H・投票率を記載
   - `knowledge/toto_knowledge.md` のルールを適用してモデル予想に対する評価を行う

2. **スクリプト作成**: `predict_XXXX.py`（シングル）と `predict_XXXX_multi.py`（マルチ）

3. **MD保存**: 予想結果を `predictions/` フォルダに保存
   - `predictions/XXXX_info.md` — 最新情報・フォーム評価
   - `predictions/XXXX_single.md` — シングル予想（1試合1択）
   - `predictions/XXXX_multi.md` — マルチ予想（手動調整済み・組み合わせ）

4. **結果確認後**: `predictions/XXXX_review.md` を作成して振り返り
   - `knowledge/toto_knowledge.md` にナレッジを追記

### MD ファイルの必須記載項目

**最新情報 (`XXXX_info.md`)**
- 全13試合のリーグ区分（J1/J2/J3）
- 各チームの直近フォーム（5試合）
- 今シーズンH2H（あれば）
- モデル vs 最新情報の総評（✅/⚠️/🚨）

**シングル予想 (`XXXX_single.md`)**
- 開催回・予想日・モデル名・バックテスト正答率
- 全13試合の予想（試合番号 / ホーム / アウェイ / 予想 / 確率1/0/2 / 推定方法）
- 予想配列 `[N/N/N/...]`
- toto公式対戦データ取得分（gameid・H2H成績）

**マルチ予想 (`XXXX_multi.md`)**
- 各試合の確率・投票率・エッジ・フォーム評価
- 手動調整箇所（🔄マーク）と理由
- 予算別プラン比較（1,000 / 3,200 / 6,400円）
- 推奨プラン（3,200円）の選択詳細と全組み合わせリスト

**振り返り (`XXXX_review.md`)**
- 全13試合の正誤
- 手動調整した試合の検証（修正が正しかったか）
- エッジ試合の結果（高エッジで当たった/外れた）
- `knowledge/toto_knowledge.md` への追記内容

### ファイル命名規則
```
predictions/
  1624_info.md      ← 最新情報
  1624_single.md    ← シングル予想
  1624_multi.md     ← マルチ予想
  1624_review.md    ← 結果後の振り返り
  1625_info.md
  ...

knowledge/
  toto_knowledge.md ← 蓄積ナレッジ（振り返りから更新）
```

## データ収集ルール

- toto公式対戦データ（rival_team_scraper）は予想時に都度取得する
  ```bash
  python src/scraper/rival_team_scraper.py --hold-cnt-id XXXX --seed-gameid XXXXX
  ```
- 投票率データ（vote_rates）は販売終了後に取得できる。取得後はマルチ予想を再実行するとエッジ連動選定になる

### 「最新情報に更新して」と言われたとき

以下を順番に実行すること：

1. **リアルタイム投票率を取得**（販売中でも取得可能）
   ```
   URL: https://sp.toto-dream.com/dcs/subos/screen/si01/ssin025/PGSSIN02501ForwardVotetotoSP.form?holdCntId=XXXX&commodityId=01&gameAssortment=9&fromId=SSIN026
   ```
   - WebFetch でページを取得し、13試合の投票率（VR1/VR0/VR2）を抽出
   - `data/raw/toto_vote_rates.csv` に保存（既存の同回データがあれば上書き）

2. **マルチ予想を再実行**
   ```bash
   python predict_XXXX_multi.py
   ```

3. **`predictions/XXXX_multi.md` を更新**
   - 投票率・エッジ値を全試合分記載
   - 予想日を更新
   - 複数択の変更があれば組み合わせリストも更新

## モデル

- **シングル予想**: RandomForest + Optuna（67特徴量、バックテスト 48.15%）
- **マルチ予想**: 同上（Edge-Aware 最適化）
- 前回スクリプト（`predict_XXXX.py`）を雛形として次回分を作成する
