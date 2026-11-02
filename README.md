# Kaggle Playground S6E10：航空会社の顧客満足度予測

There is an English version after this Japanese version.

Kaggle Playground Series Season 6 Episode 10 への取り組みです。
乗客の属性・フライト情報・サービス評価から、乗客が満足したかどうか（`satisfaction`）を予測する二値分類コンペです。

- コンペページ：https://www.kaggle.com/competitions/playground-series-s6e10
- 評価指標：ROC AUC
- 期間：2026年10月（締切 10/31）

## 最終結果

| 枠 | 中身 | CV | Public | Private |
|---|---|---|---|---|
| **CV ベース** | Trust the CV（253モデルのスタッキング）を自分で再実行・再現 | 0.96229 | 0.96181 | **0.96238** |
| PLB ベース | Public LB に合わせて作られた公開ブレンド | なし | 0.96225 | 0.96223 |

- 最終順位：**221位**（Private 0.96238）
- Public では PLB ベースの枠が上だったが、**Private では CV ベースの枠が上回った**。
  Public の6万行に合わせ込んだ分は Private に持ち越されず、CV だけで選んだ提出が最も安定していた。

## データ

- train：約70万行 / test：約30万行（本物のアンケートデータから生成された合成データ）
- 特徴量：数値（年齢・飛行距離・出発/到着の遅延）、サービス評価13項目（0〜5）、カテゴリ（性別・顧客タイプ・旅行目的・クラス）
- 欠損は `Arrival Delay in Minutes` のみ（約0.04%）

※ コンペのルールに従い、データファイルはリポジトリに含めていません。

## アプローチ

- **検証**：StratifiedKFold（5分割、`random_state=42`）。すべての判断を CV の AUC で行った。
  全モデルで同じ分割を使い、oof 同士を公平に比較・ブレンド・スタッキングできるようにした。
- **モデル**：LightGBM、CatBoost、XGBoost、ロジスティック回帰、Kumo-Tabular（Kaggle GPU）
- **特徴量**：Flight Distance の頻度エンコーディングと、fold 内で作るターゲットエンコーディング
- **アンサンブル**：重み付き平均、シード平均、ロジット空間での L2 ロジスティック回帰によるスタッキング
- **公開データの利用**：Chris Deotte の TFM の oof、公開の CV スタッキング（181・201・253モデル）を、自分の CV で比較したうえで使用

## 実験記録

| # | 内容 | CV AUC | Public LB |
|---|---|---|---|
| 1 | LightGBM ベースライン | 0.95886 | 0.95827 |
| 2 | ＋ サービス評価の平均・最小値 | 0.95863 | - |
| 3 | ＋ 元データを学習データに追加 | 0.95865 | - |
| 4 | CatBoost（GPU）ベースライン | 0.95846 | - |
| 5 | ブレンド：LightGBM ＋ CatBoost | 0.95895 | 0.95836 |
| 6 | ロジスティック回帰 | 0.94868 | - |
| 7 | ブレンド：LightGBM ＋ CatBoost ＋ ロジスティック回帰 | 0.95904 | 0.95847 |
| 8 | ブレンド：チューニング後 LightGBM ＋ CatBoost | 0.95916 | 0.95857 |
| 9 | CatBoost のチューニング | 0.95852 | - |
| 10 | ブレンド：チューニング後 LightGBM ＋ チューニング後 CatBoost | 0.95910 | - |
| 11 | **LightGBM ＋ Flight Distance エンコーディング** | **0.96054** | - |
| 12 | ブレンド：FD入り LightGBM ＋ CatBoost | 0.96058 | 0.96017 |
| 13 | CatBoost ＋ Flight Distance をカテゴリ化 | 0.96031 | - |
| 14 | ブレンド：FD入り LightGBM ＋ FD入り CatBoost | 0.96072 | 0.96033 |
| 15 | 自分の8モデルのスタッキング | 0.96071 | - |
| 16 | シード平均（LightGBM・CatBoost 各3シード）のブレンド | 0.96078 | 0.96035 |
| 17 | XGBoost ＋ Flight Distance エンコーディング | 0.96031 | - |
| 18 | スタッキング：自分のモデル ＋ 公開 TFM 11種 | 0.96149 | 0.96105 |
| 19 | スタッキング：自分の全モデル ＋ 公開 TFM | 0.96166 | - |
| 20 | 線形モデル ＋ Flight Distance の One-Hot | 0.95066 | - |
| 21 | Kumo-Tabular small ＋ 路線特徴量（Kaggle GPU、お手本10万行） | 0.95907 | - |
| 22 | XGBoost（目的関数をペアの順番に変更） | 0.95868 | - |
| 23 | LightGBM ＋ 疑似ラベル（fold ごとに作成） | 0.96053 | - |
| 24 | 公開 181モデルのスタッキング（参考） | 0.96217 | - |
| 25 | スタッキング：181モデル ＋ 自分の FD モデル | 0.96216 | 0.96176 |
| 26 | スタッキング：181モデル ＋ 201モデル ＋ 自分のモデル | 0.96219 | 0.96179 |
| 27 | **Trust the CV（253モデル）を自分で再実行・再現** | **0.96229** | **0.96181** |

## わかったこと

### 検証
- 開発中、CV と Public LB はずっと連動していた（Public は CV より 0.0004〜0.0006 低い）。
- 最終的に、Private は CV ベースの提出で **CV より約0.0001 高く**出た。
  Public の6万行は全員共通で、たまたま予測しにくい乗客が多めだったと考えられる。
- CV が 0.0001 違う2つのスタッキングが、Public では 0.00001 しか違わないこともあった。
  Public の6万行では細かい差を見分けられないので、判断は CV で行うべき。

### 一番効いたこと：Flight Distance は「路線の ID」
- Flight Distance は約70万行で 3,474 種類しかなく、1つの値が平均約200回出てくる。
  近い距離同士でも満足率が似ておらず、**数値ではなく路線の ID のように振る舞う**。
- 頻度エンコーディングと、fold 内のターゲットエンコーディング（スムージング a = 20）を足すと、
  LightGBM の CV が **0.95903 → 0.96054（+0.0015）**。全 fold で改善し、このプロジェクトの改善の大半を占めた。
- リーク対策：ターゲットエンコーディングは各 fold の学習用データだけから作り、
  さらに内側の5分割で、自分の答えが自分のエンコーディングに入らないようにした。CV と LB の差が変わらなかったことで、リークが無いことを確認した。
- CatBoost でも、Flight Distance をカテゴリとして渡すと同様に効いた（0.95846 → 0.96031）。

### なぜ Flight Distance だけが効いたのか
- **体系的な調査**：各列をグループの ID とみなしてエンコーディングすると、
  Flight Distance は「数値としての単体 AUC 0.649 → グループとして 0.743」と大きく上がった。
  他の列にこの性質は無く、Flight Distance と他の列の組み合わせも、木のモデルがすでに表現できる範囲だった。
- **ヒストグラムの区切り（max_bin）の実験**（fold 0、学習率 0.1）：

  | 特徴量 | max_bin=255（デフォルト） | max_bin=4096 |
  |---|---|---|
  | 元の列だけ | 0.95930 | 0.96003（+0.00073） |
  | ＋ FD の頻度・ターゲットエンコーディング | **0.96079** | 0.96062 |

  LightGBM はデフォルトで数値を最大255区間に区切るため、約14種類の距離が1つの区間にまとめられ、
  **路線の区別が失われていた**。区切りを細かくすると一部取り戻せるが、
  1つの路線を切り出すには2回の分岐が必要で、78枚の葉では3,500路線を切り分けきれない。
  エンコーディングは、全路線の満足しやすさを1つの列にまとめて渡せるので、より強く効いた。
- **上位者から学んだ本当の理由**：このデータは GPT-2 で生成されていて、
  3桁の距離は GPT-2 にとって **1つの単語（トークン）** だった。
  生成モデルが「ラベルの後に、この距離という単語を書く確率」を学んでいたため、
  距離の値ごとに満足しやすさがバラバラになっていた。

### 効かなかったこと
- **サービス評価の集約**（平均・最小値、0を「該当なし」として除いたものも含む）、**他の数値列のエンコーディング**（年齢・遅延）、
  **Flight Distance と他の列の組み合わせ**は、どれも CV を改善しなかった。
  70万行あると、木のモデルはこれらの関係を通常の分岐ですでに学習できており、似た列を足しても列の抽出（`colsample_bytree`）が薄まるだけだった。
- **元データ**：行として足しても、元データだけで学習したモデル（teacher）の予測を特徴量にしても、効かなかった。
  teacher 単体はコンペの train で AUC 0.953 と、本物のデータの知識自体は通用していたが、合成データの train が大きく、すでにその情報を含んでいた。
  （※ これは「自分たちが試したやり方では」効かなかった、という意味。）

### モデルとアンサンブル
- **チューニング**（Optuna、fold 0 で30回探索 → 5fold で確認）で LightGBM は 0.95886 → 0.95903（5fold 中4つで改善）。
  CatBoost のチューニングは単体では少し上がったが、LightGBM と似た予測になり、ブレンドでは逆に下がったので不採用。
- **LightGBM と CatBoost のブレンド**は、木の作り方（葉ごと vs 対称木）が違い、間違え方が異なるので安定して効いた。
- **ロジスティック回帰**（0.94868）は、ブレンドの重みが0になった。
- **自分の8モデルのスタッキング**は、重み付き平均と同じ CV になった（0.96071 vs 0.96072）。
  学習した係数の合計がほぼ1で、メタモデルが重み付き平均を再発見していた。
- **シード平均**（各3シード）は小さいが確実に効いた：LightGBM 0.96054 → 0.96064、CatBoost 0.96031 → 0.96043。

### スタッキングでのモデルの価値は「相手が持っていない情報」で決まる

| 足したもの | TFM 11種（生の列のみ）に対して | 大きな公開スタッキングに対して |
|---|---|---|
| FD入りの木のモデル | **＋0.0003** | ≈ 0 |
| FD入りの Kumo-Tabular | ＋0.00007 | ≈ 0 |
| 目的関数をペアの順番にした XGBoost | ＋0.00008 | ≈ 0 |
| fold ごとの疑似ラベルの LightGBM | ＋0.00025 | ≈ 0 |

- 生の21列しか見ていない TFM に対しては、路線の情報や疑似ラベルが新しい情報になった。
  路線系のモデルをすでに多数含む公開スタッキングに対しては、どれも新しい情報にならなかった。
- FD入りの XGBoost は、路線の情報を持つ木のモデルが自分だけのときは係数 +0.30 と大きく使われたが、
  FD入りの LightGBM・CatBoost がそろうと重みが0になった。
- **単体の強さではなく、他のモデルが持っていない情報**が、スタッキングでの価値を決める。

### 公開スタッキング同士の組み合わせ
- 181モデルと201モデルのスタッキングは、単体ではどちらも CV 0.96217 だが、順位の相関が 0.998 とやや低く、組み合わせると 0.96219 に上がった。
- Trust the CV（253モデル、CV 0.96229）は、181・201モデルの材料をほぼ含んだうえで新しいモデルとメタモデルの工夫を足したもので、
  181・201モデルを混ぜると、混ぜた割合に応じて単調に下がった。材料の多いスタッキングが来ると、少ないものは薄めるだけになる。
- Trust the CV の再実行では、5つの fold を並列で計算すると、4コアの CPU で処理同士がコアを奪い合って極端に遅くなった。
  1fold ずつ順番に計算し、収束条件を `tol=1e-4` に戻すことで、約1時間止まっていた計算が約15分で終わった（CV はほぼ同じ 0.962279）。

## コンペ終了後に公開された上位解法から

上位の解法は、そろって **このデータは GPT-2 が書いた** ことに気づき、**生成の仕組みを真似たモデル**を作っていた。

### 1位（Chris Deotte）：ラベルを先に置いた GPT-2
- データは **GReaT**（GPT-2 に本物の行を「`column is value`」の文章として学習させ、1トークンずつ新しい行を書かせる方法）で作られており、
  **ラベルが最初に書かれ、その後に21個の特徴量が書かれていた**ことを特定した。
- GPT-2 を元データ → コンペのデータの順にファインチューニングし、
  「満足/不満」それぞれの後にこの行が書かれる確率を計算して、ベイズの定理で分類する：

  $$\log\frac{P(\text{満足}\mid x)}{P(\text{不満}\mid x)} = \log\frac{P(\text{満足})}{P(\text{不満})} + \log P(x\mid\text{満足}) - \log P(x\mid\text{不満})$$

- 列の順番は30通りで計算し、log-sum-exp でまとめる。
- この1モデルで OOF 0.9624。公開の245モデルのスタッキングに足すと +0.0005 上がり、最終スタッキングでは GPT 系のモデルが出力の約80%を担っていた。

### 4位（Ole-Jakob）：任意順の自己回帰 Transformer（AO-ARM）
- 1位と同じ「ラベルごとに、特徴量が生成される確率を比べる」生成モデルを、**自作の Transformer をゼロから学習**して実現（OOF 0.9620）。
- **公開の oof を一切使わず**、189個の自作モデルだけで Private 0.96304。
- 上位との「説明できない差」から、**「行がどう生成されたか」という未使用の情報があるはず**と仮説を立てたことが、この方向に進むきっかけだった。
- Claude と Codex にエージェントとして実験を回させ、189モデルの大半を作らせていた。

### 7位（学生2人のチーム）：シードを重ねた生成 Transformer
- Public 318位から、**5fold CV だけで選んで** Private 7位に上がった。
- 各列の値を1トークンにした小さな Transformer（6層・幅256）で $P(x, y)$ を学習し、ベイズの定理で分類（GenPure）。36エポックの長い学習とシード平均で、1シード 0.9612。
- 混ぜる候補の採用基準を**結果を見る前に固定し**、一度も緩めなかった。
- **CV の順番が Private の順番を完全に当てた**一方、Public は最後の改善を見分けられず、Public の最高ファイルは Private で弱かった。

## 振り返り

- **自分たちの到達点**：公開されている CV スタッキングの最先端（Private 0.9624 前後）までは到達した。
  上位との差は、生成モデルという**新しい種類の情報**を作れたかどうかだった。
- **自分たちの発見が捉えていたもの**：Flight Distance が路線の ID のように振る舞うという発見は、
  「データが GPT-2 のトークンで書かれていた」ことの影を正しく捉えていた。
  そこから「なぜそうなるのか」「データはどう作られたのか」まで踏み込めていれば、上位と同じ答えに近づけた可能性がある。
- **次に生かすこと**
  1. 説明できない差や不思議なパターンを見つけたら、**データはどう作られたのか**を考える。
  2. 値の種類が多い数値列があれば、`max_bin` を上げて fold 0 で比べる。上がれば、その列は値そのもの（ID）に意味があるサイン。
  3. 失敗したやり方は保留にしても、アイデア全体は捨てない。
  4. 判断は CV で行い、採用基準は結果を見る前に決めておく。
  5. Public LB を見て何度も選んだ改善は、Private には持ち越されないことが多い。

## 実行方法

1. コンペページから `train.csv`・`test.csv`・`sample_submission.csv` をダウンロードし、このフォルダに置く。
2. `model.ipynb` の一番上の再開用セルを実行してから、各モデルのセルを実行する。予測は `preds/` に保存される。
3. 公開データを使うスタッキングを再現する場合は、Chris Deotte の TFM の oof（`tfs/`）と公開スタッキングの oof（`atlas/`）を別途ダウンロードして配置する。

## 環境

- Python 3.14
- pandas, numpy, scipy, scikit-learn, lightgbm, catboost, xgboost, optuna, matplotlib, seaborn
- GPU：CatBoost はローカルの NVIDIA GPU、Kumo-Tabular は Kaggle Notebook の T4

## クレジット

- 公開 TFM の oof：[Chris Deotte](https://www.kaggle.com/datasets/cdeotte/s6e10-tfm-oof-and-test-predictions)
- Flight Distance を路線として扱う発想、FE の知見：Discussion の各投稿
- CV スタッキング：kratosyan、goodpjw2008、Mayank Raj（181モデル）、Yeashu Semwal（Trust the CV）ほか、oof を公開した皆さん
- 上位解法：Chris Deotte（1位）、Ole-Jakob（4位）、Team UIT - Loc & An（7位）


# Kaggle Playground S6E10: Predicting Airline Satisfaction

My solution for Kaggle Playground Series Season 6 Episode 10.
The task is binary classification: predicting whether an airline passenger was satisfied (`satisfaction`),
based on passenger attributes, flight information, and service ratings.

- Competition: https://www.kaggle.com/competitions/playground-series-s6e10
- Metric: ROC AUC
- Period: October 2026 (deadline: Oct 31)

## Final Results

| Slot | What it is | CV | Public | Private |
|---|---|---|---|---|
| **CV-based** | Trust the CV (253-member stack), re-run and reproduced myself | 0.96229 | 0.96181 | **0.96238** |
| Public-LB-based | A public blend built by tuning against the public leaderboard | none | 0.96225 | 0.96223 |

- Final rank: **221st** (Private 0.96238)
- The public-LB-based slot scored higher on the public leaderboard, but **the CV-based slot won on the private leaderboard**.
  Gains fitted to the 60k public rows did not carry over to the private set; the submission chosen by CV alone was the most reliable.

## Data

- Train: ~700k rows / Test: ~300k rows (synthetic data generated from a real survey dataset)
- Features: numerical (age, flight distance, departure/arrival delays), 13 service ratings (0–5),
  and categorical (gender, customer type, type of travel, class)
- Missing values only in `Arrival Delay in Minutes` (~0.04%)

*Data files are not included in this repository, in accordance with the competition rules.*

## Approach

- **Validation:** StratifiedKFold (5 folds, `random_state=42`). Every decision was made by CV AUC.
  All models share the same split, so their OOF predictions can be compared, blended, and stacked fairly.
- **Models:** LightGBM, CatBoost, XGBoost, logistic regression, Kumo-Tabular (Kaggle GPU)
- **Features:** frequency encoding and in-fold target encoding of Flight Distance
- **Ensembling:** weighted averaging, seed averaging, and L2 logistic-regression stacking in logit space
- **Use of public resources:** Chris Deotte's TFM OOF predictions and public CV stacks (181, 201, and 253 members),
  each compared against my own models using my own CV before being used

## Experiment Log

| # | Description | CV AUC | Public LB |
|---|---|---|---|
| 1 | LightGBM baseline | 0.95886 | 0.95827 |
| 2 | + service-rating mean & min | 0.95863 | - |
| 3 | + original dataset added to training | 0.95865 | - |
| 4 | CatBoost (GPU) baseline | 0.95846 | - |
| 5 | Blend: LightGBM + CatBoost | 0.95895 | 0.95836 |
| 6 | Logistic regression | 0.94868 | - |
| 7 | Blend: LightGBM + CatBoost + logistic regression | 0.95904 | 0.95847 |
| 8 | Blend: tuned LightGBM + CatBoost | 0.95916 | 0.95857 |
| 9 | Tuned CatBoost | 0.95852 | - |
| 10 | Blend: tuned LightGBM + tuned CatBoost | 0.95910 | - |
| 11 | **LightGBM + Flight Distance encoding** | **0.96054** | - |
| 12 | Blend: FD-encoded LightGBM + CatBoost | 0.96058 | 0.96017 |
| 13 | CatBoost + Flight Distance as a categorical feature | 0.96031 | - |
| 14 | Blend: FD-encoded LightGBM + FD-encoded CatBoost | 0.96072 | 0.96033 |
| 15 | Stacking my 8 models | 0.96071 | - |
| 16 | Blend of seed-averaged models (LightGBM & CatBoost, 3 seeds each) | 0.96078 | 0.96035 |
| 17 | XGBoost + Flight Distance encoding | 0.96031 | - |
| 18 | Stacking: my models + 11 public TFMs | 0.96149 | 0.96105 |
| 19 | Stacking: all my models + public TFMs | 0.96166 | - |
| 20 | Linear model + one-hot Flight Distance | 0.95066 | - |
| 21 | Kumo-Tabular small + route features (Kaggle GPU, 100k context rows) | 0.95907 | - |
| 22 | XGBoost with a pairwise-ranking objective | 0.95868 | - |
| 23 | LightGBM + pseudo-labels (generated per fold) | 0.96053 | - |
| 24 | Public 181-member stack (reference) | 0.96217 | - |
| 25 | Stacking: 181-member stack + my FD models | 0.96216 | 0.96176 |
| 26 | Stacking: 181-member + 201-member stacks + my models | 0.96219 | 0.96179 |
| 27 | **Trust the CV (253 members), re-run and reproduced myself** | **0.96229** | **0.96181** |

## Findings

### Validation
- During development, CV and the public LB moved together (public was consistently 0.0004–0.0006 below CV).
- In the end, the CV-based submission scored **about 0.0001 above CV on the private set**.
  The 60k public rows are shared by everyone and likely happened to contain harder-to-predict passengers.
- Two stacks whose CV differed by 0.0001 differed by only 0.00001 on the public LB.
  A 60k-row public split cannot resolve such small differences, so decisions should be made by CV.

### What worked most: Flight Distance as a "route ID"
- Flight Distance has only 3,474 unique values across ~700k rows (~200 repeats per value).
  Nearby distances do not share similar satisfaction rates; **the values behave like route IDs, not numbers**.
- Adding a frequency encoding and an in-fold target encoding (smoothing a = 20) raised LightGBM CV from
  **0.95903 → 0.96054 (+0.0015)**, with gains on every fold. This single change accounts for most of the project's improvement.
- Leakage control: the target encoding was built only from the training part of each fold, with an inner 5-fold split
  so that no row's own label enters its own encoding. The CV–LB gap stayed unchanged, confirming there was no leakage.
- CatBoost benefited similarly when Flight Distance was passed as a categorical feature (0.95846 → 0.96031).

### Why only Flight Distance worked
- **Systematic scan:** treating each column as a group ID and target-encoding it, Flight Distance jumped from
  a univariate AUC of 0.649 (as a number) to 0.743 (as a group). No other column showed this property,
  and combinations of Flight Distance with other columns were already within what trees can represent.
- **Histogram binning (`max_bin`) experiment** (fold 0, learning rate 0.1):

  | Features | max_bin=255 (default) | max_bin=4096 |
  |---|---|---|
  | Raw columns only | 0.95930 | 0.96003 (+0.00073) |
  | + FD frequency & target encoding | **0.96079** | 0.96062 |

  LightGBM bins each numeric feature into at most 255 buckets by default, so ~14 distances share each bucket and
  **route identity is lost**. Finer bins recover part of it, but isolating one route takes two splits, and a tree with
  78 leaves cannot carve out 3,500 routes. The encoding hands every route's propensity to the tree as a single column,
  so it works better.
- **The real reason (revealed after the competition):** this data was generated by GPT-2, and a 3-digit distance is
  **a single token (word)** for GPT-2. The generator learned "the probability of writing this distance token after the label",
  so satisfaction rates vary value by value.

### What did not work
- **Service-rating aggregates** (mean/min, including versions that treat 0 as "not applicable"),
  **encodings of other numeric columns** (age, delays), and **combinations of Flight Distance with other columns**
  all failed to improve CV. With ~700k rows, gradient-boosted trees already learn these relationships through ordinary splits;
  extra correlated columns only dilute feature sampling (`colsample_bytree`).
- **Original dataset:** neither appending its rows nor using a "teacher" model trained on it as a feature helped.
  The teacher alone reaches AUC 0.953 on the competition train set, so the real-data knowledge does transfer,
  but the large synthetic training set already contains it.
  (This means it did not work *in the ways I tried*.)

### Models and ensembling
- **Tuning** (Optuna, 30 trials on fold 0, verified on all 5 folds) improved LightGBM from 0.95886 to 0.95903 (gains on 4 of 5 folds).
  Tuning CatBoost helped it alone but made it more similar to LightGBM, lowering the blend, so it was not adopted.
- **Blending LightGBM and CatBoost** helped consistently: their tree-growing strategies (leaf-wise vs. symmetric trees) lead to different errors.
- **Logistic regression** (0.94868) received a blend weight of 0.
- **Stacking my 8 models** gave the same CV as a weighted average (0.96071 vs. 0.96072).
  The learned coefficients summed to ~1, i.e. the meta-model rediscovered the weighted average.
- **Seed averaging** (3 seeds each) gave small but reliable gains: LightGBM 0.96054 → 0.96064, CatBoost 0.96031 → 0.96043.

### In stacking, a model's value is information the others lack

| Added model | Gain vs. 11 TFMs (raw columns only) | Gain vs. large public stacks |
|---|---|---|
| FD-encoded tree models | **+0.0003** | ≈ 0 |
| FD-aware Kumo-Tabular | +0.00007 | ≈ 0 |
| XGBoost with a pairwise-ranking objective | +0.00008 | ≈ 0 |
| LightGBM with per-fold pseudo-labels | +0.00025 | ≈ 0 |

- Against TFMs that only see the 21 raw columns, route information and pseudo-labels were new information.
  Against public stacks that already contain many route-aware models, none of them added anything.
- FD-encoded XGBoost received a large coefficient (+0.30) when it was the only route-aware tree model in the stack,
  but a weight of 0 once FD-encoded LightGBM and CatBoost were present.
- **What earns weight in a stack is information the other members lack, not standalone strength.**

### Combining public stacks
- The 181- and 201-member stacks both score CV 0.96217 alone, but their rank correlation is only 0.998,
  and combining them raised CV to 0.96219.
- Trust the CV (253 members, CV 0.96229) contains almost all members of the 181- and 201-member stacks plus new members
  and stacker improvements. Mixing in the smaller stacks lowered CV monotonically with their weight:
  once a stack with more members arrives, the smaller ones only dilute it.
- When re-running Trust the CV, computing the 5 folds in parallel made the 4-core CPU thrash, slowing things to a crawl.
  Running one fold at a time with `tol=1e-4` finished in ~15 minutes a job that had stalled for an hour (CV essentially unchanged: 0.962279).

## Lessons from the top solutions (published after the competition)

The top solutions all realized that **this data was written by GPT-2** and built **models that imitate how the data was generated**.

### 1st place (Chris Deotte): GPT-2 with the target first
- Identified that the data was generated with **GReaT** (fine-tuning GPT-2 on real rows written as "`column is value`" text,
  then sampling new rows token by token), and that **the label was written first, followed by the 21 features**.
- Fine-tuned GPT-2 on the original data and then on the competition data, computed the probability of the row being written
  after each label, and classified with Bayes' rule:

  $$\log\frac{P(\text{satisfied}\mid x)}{P(\text{dissatisfied}\mid x)} = \log\frac{P(\text{satisfied})}{P(\text{dissatisfied})} + \log P(x\mid\text{satisfied}) - \log P(x\mid\text{dissatisfied})$$

- Scored 30 column orders and combined them with log-sum-exp.
- This single model reached OOF 0.9624. Adding it to the 245 public pairs gained +0.0005, and in the final stack
  the GPT-based models carried about 80% of the output.

### 4th place (Ole-Jakob): any-order autoregressive transformer (AO-ARM)
- The same generative idea (compare how likely the features are under each label), implemented with
  **a custom transformer trained from scratch** (OOF 0.9620).
- Used **no public OOF files at all**: 189 self-built models reached private 0.96304.
- An unexplained gap to the top led to the hypothesis that **unused information, such as how the rows were generated**, existed,
  which set the search in this direction.
- Claude and Codex ran experiments as agents and built most of the 189 models.

### 7th place (a two-student team): a seed-bagged generative transformer
- Rose from 318th on the public LB to 7th on the private LB **by selecting with 5-fold CV alone**.
- Learned P(x, y) with a small transformer (6 layers, width 256) where each column value is one token, and classified
  with Bayes' rule (GenPure). Long training (36 epochs) and seed bagging gave 0.9612 per seed.
- **Fixed the acceptance criteria for blend candidates before seeing any results** and never loosened them.
- **CV ordered the private scores exactly**, while the public LB could not see the final improvements,
  and their best public file was weak on private.

## Reflection

- **Where I got to:** I reached the frontier of publicly available CV stacks (private ≈ 0.9624).
  The gap to the top came down to whether one could build a **new kind of information**: a generative model.
- **What my discovery was actually capturing:** the finding that Flight Distance behaves like a route ID was a correct shadow of
  "the data was written in GPT-2 tokens". Had I pushed further to ask *why* this happens and *how the data was generated*,
  I might have reached the same answer as the top teams.
- **What I will carry forward**
  1. When I see an unexplained gap or a strange pattern, ask **"how was this data generated?"**
  2. If a numeric column has many unique values, compare `max_bin` settings on fold 0. A gain is a sign that the values act as IDs.
  3. Park the specific recipe that failed, not the whole idea.
  4. Make decisions by CV, and fix acceptance criteria before seeing results.
  5. Improvements selected repeatedly against the public LB often do not carry over to the private set.

## How to Run

1. Download `train.csv`, `test.csv`, and `sample_submission.csv` from the competition page and place them in this directory.
2. Run the setup cell at the top of `model.ipynb`, then the cells for each model. Predictions are saved to `preds/`.
3. To reproduce the stacks that use public resources, download Chris Deotte's TFM OOF predictions (`tfs/`) and
   the public stack OOF files (`atlas/`) separately.

## Environment

- Python 3.14
- pandas, numpy, scipy, scikit-learn, lightgbm, catboost, xgboost, optuna, matplotlib, seaborn
- GPU: local NVIDIA GPU for CatBoost; Kaggle Notebook T4 for Kumo-Tabular

## Credits

- Public TFM OOF predictions: [Chris Deotte](https://www.kaggle.com/datasets/cdeotte/s6e10-tfm-oof-and-test-predictions)
- The idea of treating Flight Distance as a route and related feature-engineering insights: Discussion posts by the community
- CV stacks: kratosyan, goodpjw2008, Mayank Raj (181 members), Yeashu Semwal (Trust the CV), and everyone who shared OOF predictions
- Top solutions: Chris Deotte (1st), Ole-Jakob (4th), Team UIT - Loc & An (7th)