#  実装レポート: [20 Newsgroups]


## 概要
本ドキュメントは、20 Newsgroupsに対して、自作LDAモデル(SVI)を適用してみるものです。
Variational Inference: A Review for Statisticiansの論文で適用例があり、それを参考に作っていきます

---
## 1. 使用するデータセット：「20 Newsgroups」

今回使用するのは、機械学習のテキスト分類やクラスタリングで最もよく使われる標準データセット「**20 Newsgroups**」です。(これは以前と同じです)

* **概要**: 1990年代のインターネット掲示板（ニュースグループ）の投稿を集めたデータセットです。
* **データ数**: 約18,000件のテキスト文書。
* **トピック（正解ラベル）**: 宇宙（space）、スポーツ（野球・ホッケー）、宗教、IT（Mac・Windows）、政治など、20の明確なカテゴリに分かれています。
* **なぜLDAに向いているか**: 最初から「明らかに違う話題」が混ざっていることが分かっているため、LDAが正しくサイコロ（トピック）を分離できたかどうかの「答え合わせ」が非常に直感的に行えるからです。
---

## 2. モデル式、添え字の説明

Bleiらの論文「Variational Inference: A Review for Statisticians (2017)」における潜在的ディリクレ配分法（LDA）の確率的グラフィカルモデル表現、および定義される添え字とパラメータは以下の通りです。

### 添え字の定義
- $d \in \{1, \dots, D\}$ : 文書（Document）のインデックス（全文書数 $D$）
- $n \in \{1, \dots, N_d\}$ : 文書 $d$ 内の単語（Token）のインデックス（文書 $d$ の総単語数 $N_d$）
- $k \in \{1, \dots, K\}$ : 潜在トピック（Topic）のインデックス（総トピック数 $K$）
- $v \in \{1, \dots, V\}$ : 語彙（Vocabulary）のインデックス（総語彙数 $V$）

### 確率変数とパラメータ
- $w_{d,n}$ : 観測変数。文書 $d$ の $n$ 番目の単語（1-of-V形式のベクトル、または語彙インデックス）。
- $z_{d,n}$ : 潜在変数。文書 $d$ の $n$ 番目の単語に割り当てられた潜在トピック（1-of-K形式のベクトル）。
- $\theta_d$ : 潜在変数。文書 $d$ のトピック混合比率（$K$ 次元シンプレックス上の確率ベクトル）。
- $\beta_k$ : 潜在変数（グローバル変数）。トピック $k$ における語彙分布（$V$ 次元シンプレックス上の確率ベクトル）。
- $\alpha$ : ハイパーパラメータ。文書のトピック分布 $\theta_d$ に対するディリクレ分布の事前パラメータ（$K$ 次元）。
- $\eta$ : ハイパーパラメータ。トピックの語彙分布 $\beta_k$ に対するディリクレ分布の事前パラメータ（$V$ 次元）。

### 生成プロセスと同時確率分布
LDAにおける完全データ同時確率分布は、以下のように定式化されます。

$$p(w, z, \theta, \beta | \alpha, \eta) = \prod_{k=1}^K p(\beta_k | \eta) \prod_{d=1}^D \left( p(\theta_d | \alpha) \prod_{n=1}^{N_d} p(z_{d,n} | \theta_d) p(w_{d,n} | \beta_{z_{d,n}}) \right)$$

ここで、各条件付き確率分布は以下のように定義されます。
- $\beta_k \sim \text{Dirichlet}(\eta)$
- $\theta_d \sim \text{Dirichlet}(\alpha)$
- $z_{d,n} \sim \text{Multinomial}(\theta_d)$
- $w_{d,n} \sim \text{Multinomial}(\beta_{z_{d,n}})$

---

## 3. パラメータの更新方法

真の事後分布 $p(z, \theta, \beta | w)$ は分母の周辺尤度（Evidence）の計算が困難（Intractable）であるため、変分推論を用いて近似します。平均場近似（Mean-Field Approximation）に基づき、以下の完全に独立な変分分布 $q$ を導入します。

$$q(z, \theta, \beta | \phi, \gamma, \lambda) = \prod_{k=1}^K q(\beta_k | \lambda_k) \prod_{d=1}^D \left( q(\theta_d | \gamma_d) \prod_{n=1}^{N_d} q(z_{d,n} | \phi_{d,n}) \right)$$

ここで、各変分パラメータは以下の通りです。
- $\lambda_k$ : トピック $k$ の語彙分布に関する変分ディリクレパラメータ（$V$ 次元）。（scikit-learnの `model.components_` に相当）
- $\gamma_d$ : 文書 $d$ のトピック比率に関する変分ディリクレパラメータ（$K$ 次元）。（`topic_distribution` に相当）
- $\phi_{d,n}$ : 文書 $d$ の $n$ 番目の単語のトピック割り当てに関する変分多項分布パラメータ（$K$ 次元）。

### 3.1 CAVI (Coordinate Ascent Variational Inference) による局所更新
各文書 $d$ に対して、グローバルパラメータ $\lambda$ を固定した状態で、局所変分パラメータ $\phi_{d,n}$ と $\gamma_d$ が収束するまで交互に更新を行います。

1. **トピック割り当て $\phi_{d,n}$ の更新**:
   各単語 $n$ およびトピック $k$ について、以下を計算し、トピック方向に合計が1になるよう正規化します。
   $$\phi_{d,n,k} \propto \exp \left( \Psi(\gamma_{d,k}) - \Psi\left(\sum_{j=1}^K \gamma_{d,j}\right) + \Psi(\lambda_{k, w_{d,n}}) - \Psi\left(\sum_{u=1}^V \lambda_{k,u}\right) \right)$$
   ※ $\Psi(\cdot)$ はディグマ関数（Digamma function）であり、ディリクレ分布の十分統計量の期待値 $\mathbb{E}[\log \cdot]$ を計算するために用いられます。

2. **文書トピック比率 $\gamma_d$ の更新**:
   $$\gamma_{d,k} = \alpha_k + \sum_{n=1}^{N_d} \phi_{d,n,k}$$

### 3.2 SVI (Stochastic Variational Inference) によるグローバル更新
確率的変分推論（SVI）では、全文書を一括処理する代わりに、ミニバッチ単位でデータをサンプリングし、自然勾配（Natural Gradient）を用いてグローバルパラメータ $\lambda$ を効率的に更新します。

1. **ミニバッチのサンプリング**:
   全文書集合 $D$ からサイズ $B$ のミニバッチ $S_t$ をランダムに抽出します。
   
2. **ローカルパラメータの収束**:
   サンプリングされた各文書 $d \in S_t$ について、上記のCAVIステップを繰り返し、$\phi_{d,n}$ と $\gamma_d$ を最適化します。

3. **中間グローバルパラメータ $\hat{\lambda}$ の計算**:
   ミニバッチ内のデータをもとに、全文書数 $D$ にスケールアップした暫定的な最適値を計算します。
   $$\hat{\lambda}_{k,v} = \eta + \frac{D}{B} \sum_{d \in S_t} \sum_{n=1}^{N_d} \phi_{d,n,k} \mathbb{I}(w_{d,n} = v)$$
   ※ $\mathbb{I}(\cdot)$ は指示関数（Indicator function）であり、文書 $d$ の $n$ 番目の単語の語彙インデックスが $v$ と一致するときに1、それ以外は0をとります。

4. **グローバルパラメータ $\lambda$ のステップ更新**:
   過去の重みと、現在のミニバッチから得られた新知識 $\hat{\lambda}$ をステップサイズ（学習率） $\rho_t$ で加重平均します。
   $$\lambda_{k,v}^{(t)} = (1 - \rho_t) \lambda_{k,v}^{(t-1)} + \rho_t \hat{\lambda}_{k,v}$$
   ここで、ステップサイズ $\rho_t$ は以下の減衰式で定義されます。
   $$\rho_t = (t + \tau)^{-\kappa}$$
   - $t$ : 現在のイテレーション回数
   - $\tau \ge 0$ : 初期ステップの学習速度を制御するディレイパラメータ
   - $\kappa \in (0.5, 1]$ : 学習率の減衰係数（scikit-learnの `learning_decay` パラメータに対応）


