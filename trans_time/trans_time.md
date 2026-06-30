# Transformerによる時系列予測

## 概要
本ドキュメントは、Transformerによる時系列予測に関して、3つの手法で学習してみます。
論文は以下を参考にしています。
Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting

Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
---

## 1. データファイル:

---

## 2. Informer: Long-Sequence Time-Series Forecasting

本章では、時系列予測においてTransformerの限界（$O(L^2)$の計算量）を突破し、長期間予測（LSTF）を可能にした画期的なモデル「Informer」の構造とメカニズムについて、数式と具体例を交えて解説する。

### 2.1 エンコーディングに関して

時系列データは、自然言語のように「単語の意味」を持たないため、モデルに「値の大きさ」「配列の順番」「現実世界のカレンダー」という3つの情報を同時に教え込む必要がある。Informerでは、入力系列 $\mathcal{X}^t \in \mathbb{R}^{L_x \times d_x}$ に対して、以下の3つの埋め込み（Embedding）を行い、最終的な入力ベクトル $\mathcal{X}_{feed}$ を生成する。

1. **Value Embedding (値の埋め込み):**
   生の数値データを1次元畳み込み層（Conv1d）に通し、Transformerが扱える次元数 $d_{model}$ に変換する。(xw+bのようにして拡張),αをdmodelの平方根として調整。
2. **Positional Encoding (位置エンコーディング):**
   データが配列のどこにあるかを示す固定の波長ベクトル。
   $$PE_{(pos, 2j)} = \sin(pos / 10000^{2j/d_{model}})$$

   $$PE_{(pos, 2j+1)} = \cos(pos / 10000^{2j/d_{model}})$$
3. **Temporal Encoding (時間特徴エンコーディング):**
   月、日、曜日、時間などのカレンダー情報を、学習可能なカテゴリカル埋め込みとしてベクトル化する。
   $$TE_{(pos)} = \sum_{i \in \{month, day, week, hour\}} \text{Embedding}_i(stamp_i)$$

これらを要素ごとに足し合わせたものがエンコーダへの入力となる。
$$\mathcal{X}_{feed} = αVE + PE + TE$$

**【数値例】**
「6月28日 午前9時」の株価データ `[100, 124, 135]` が入力されたとする（$pos=0,1,2$）。
$pos=0$ の「100」という値は、Value層で512次元のベクトルに変換される。そこに、$pos=0$ に対応する固定の $\sin, \cos$ ベクトルが足され、さらに「6月」「28日」「日曜日」「9時」という4つのカテゴリに対応する学習済みのベクトルが足し合わされ、最終的に512次元ベクトルが完成する。

### 2.2 エンコーダ層に関して

エンコーダの役割は、過去の長い系列から「重要な文脈（Memory）」を抽出することである。Informerはここで**ProbSparse Self-Attention**と**Distilling（蒸留）**という2つの大改造を行っている。

**① ProbSparse Self-Attention（Qの剪定）**
通常のAttentionは、Query($Q$)とKey($K$)のすべての組み合わせの内積を計算するため、計算量が $O(L^2)$ となる。

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

Informerは、各 $q_i$ がどれだけ「重要（アクティブ）か」を、一様分布とのKL情報量に基づく指標 $\bar{M}(q_i, K)$ で評価する。ここで取り出すのは、一部のデータのみで図る(keyを選定)

$$\bar{M}(q_i, K) = \ln \sum_{j=1}^{L_K} e^{\frac{q_i k_j^T}{\sqrt{d_k}}} - \frac{1}{L_K} \sum_{j=1}^{L_K} \frac{q_i k_j^T}{\sqrt{d_k}}$$

この指標が高い（他のデータに対して強い関心を持つ）上位 $u$ 個のQueryだけを選抜して $\bar{Q}$ とし、計算を行う。(選ばれQueryに全keyを掛ける)。つまり選ばれなかったQの大半は学習はされない。選ばれないところは平均値で埋める。

$$\text{ProbAttention}(Q, K, V) = \text{softmax}\left(\frac{\bar{Q} K^T}{\sqrt{d_k}}\right) V$$

選抜数 $u$ は $u = c \cdot \ln L_Q$ （$c$ は定数）に制限されるため、計算量は $O(L \ln L)$ へと劇的に削減される。

**② Distilling（蒸留操作）**
Attention層の直後に、1次元のMax Pooling層を挟むことで、時間軸方向のデータサイズを半分に圧縮する。FFNした後に。
$$X_{j+1}^t = \text{MaxPool}\left( \text{ELU}\left( \text{Conv1d}( [X_j^t]_{AB} ) \right) \right)$$

**【数値例】**
$L=1000$ のデータを入れた場合、通常のAttentionは $1000 \times 1000 = 1,000,000$ 回の計算が必要になる。しかしProbSparse Attentionでは、たとえば $u = 5 \ln(1000) \approx 34$ 個の優秀なQueryだけが選ばれるため、$34 \times 1000 = 34,000$ 回の計算で済む。さらにDistillingによって、次の層に渡るデータ数は $500 \to 250$ と半減していくため、メモリがパンクしない。

### 2.3 デコーダ層に関して

デコーダは、RNNのような自己回帰（1歩ずつ予測する手法）の「遅さ」と「誤差の蓄積」を防ぐため、**Generative Style Decoder（一発出しデコーダ）**を採用している。

デコーダへの入力 X_deは、既知の直近の過去データ X_token（長さ $L_token）と、予測したい未来の枠を表すゼロ行列 X_0（長さ L_y）を結合したプレースホルダーである。

例えば、過去5日と未来30日の場合、入力時は(35,512)のようになる。ここで、次のステップで使われるのはQ(35,512)で、エンコーダのK,Vと大きさが違うことは注意

デコーダ内部では、まず未来の情報をカンニングしないためのMasked Multi-head Attentionが行われ、その後、エンコーダが抽出した過去の圧縮記憶（$K, V$）と、自身の持つプレースホルダー（$Q$）をぶつけるCross-Attentionが行われる。
最後に全結合層（Linear）を1回だけ通し、最終的な予測値 $Y_{predict}$ を出力する。

**【数値例】**
過去5日分のデータから、未来30日分を予測したいとする。
デコーダには `[過去5日分のデータ] + [0.0が30個並んだ配列]` が入力される。Cross-Attentionによって「このゼロの枠には、エンコーダのどの記憶を当てはめるべきか」が計算され、最後のDNN層を通った瞬間に、30個のゼロが一気に「30日分の具体的な予測数値」へと書き換わって出力される。

### 2.4 特徴（まとめ）

Informerが時系列モデリングの歴史にもたらした主要な特徴は以下の4点に集約される。ここには書いていないが、大前提としてかなり大きなデータセットが必要である。

1. **$O(L \ln L)$ への計算量削減:** ProbSparse AttentionとDistillingの組み合わせにより、長期間の過去データをメモリを枯渇させることなく入力できるようになった。
2. **時間的セマンティクスの完全な統合:** Temporal Encodingにより、現実世界のカレンダー情報（曜日や時間帯による周期性）をディープラーニングの重み空間にマッピングすることに成功した。
3. **誤差蓄積の回避と高速推論:** Generative Style Decoderにより、長期間の未来予測を一回の順伝播（Forward pass）で完結させ、推論速度と安定性を飛躍的に向上させた。
3. **TRANSFORMER(言語モデルとの比較):** Pre-trainingができないので、実装には時間が大幅にかかる 
---

## 3. Autoformer:

本章では、Informerの半年後に考案された、Autoformer:に関するモデルのアルゴリズムを解説する

概要としては、モデルの計算を減らすことに注力していたが、アプローチを変えてそもそも、トレンドと季節に分けて、推定していけばいいという考えのもの

## 3.1 Autoformer エンコーダのアーキテクチャ

Autoformerのエンコーダの主目的は、過去の時系列データ $\mathcal{X}$ に内在する**「真の周期性（季節成分）」を抽出し、ランダムなノイズを極限まで平滑化すること**である。標準的なTransformerのSelf-Attentionメカニズム（計算量 $\mathcal{O}(L^2)$）を廃止し、信号処理に基づく **Auto-Correlation（自己相関）** と **Series Decomposition（時系列分解）** を組み合わせることで、計算量を $\mathcal{O}(L \log L)$ に抑えつつ、時系列特有の文脈を学習する。

### 3.1.1 エンコーダ $l$ 層の全体方程式
第 $l$ 層の内部処理は以下の2つの方程式で定式化される。

$$
\mathcal{S}_{en}^{l, 1}, \_ = \text{SeriesDecomp}\left( \text{Auto-Correlation}(\mathcal{X}_{en}^{l-1}, \mathcal{X}_{en}^{l-1}, \mathcal{X}_{en}^{l-1}) + \mathcal{X}_{en}^{l-1} \right)
$$

$$
\mathcal{X}_{en}^{l}, \_ = \text{SeriesDecomp}\left( \text{FeedForward}(\mathcal{S}_{en}^{l, 1}) + \mathcal{S}_{en}^{l, 1} \right)
$$

ここで、$S_en$ は中間表現としての季節成分を表し、抽出されたトレンド成分はエンコーダにおいては不要であるため `_` として破棄される。また、各ブロックの加算（+）は残差接続（Residual Connection）を示している。

---

### 3.1.2 Auto-Correlation（自己相関）メカニズム

このブロックは、波形をシフトさせながら自己類似度を測り、周期性を増幅させる役割を担う。処理は以下の3つのステップに分解される。

#### ① 特徴量テンソルの生成
入力 $\mathcal{X}_{en}^{l-1}$ に対して、学習可能な重み行列 $W_Q, W_K, W_V$ を用いて Query、Key、Value を生成する。これらはすべて同じ入力から派生した同じサイズのテンソルである。

$$
Q = \mathcal{X}_{en}^{l-1} W_Q, \quad K = \mathcal{X}_{en}^{l-1} W_K, \quad V = \mathcal{X}_{en}^{l-1} W_V
$$

#### ② ウィーナー＝ヒンチン定理による周期の発見
時間領域における遅延（ラグ） $\tau$ ごとの自己相関 $\mathcal{R}_{QK}(\tau)$ を、高速フーリエ変換（FFT）を用いて計算する。周波数領域での共役複素数の積（畳み込み定理の応用）により、すべてのズレ幅のスコアを $\mathcal{O}(L \log L)$ で一括計算する。

$$
\mathcal{S}_{QK} = \text{FFT}(Q) \odot \text{FFT}(K)^*
$$
$$
\mathcal{R}_{QK} = \text{iFFT}(\mathcal{S}_{QK})
$$

ここで、$\text{FFT}(\cdot)$ は高速フーリエ変換、* は複素共役、dot は要素ごとのアダマール積、iFFT$ は逆高速フーリエ変換である。得られた $\mathcal{R}_{QK}$ は各ズレ幅 $\tau \in \{1, 2, \dots, L\}$ における周期性の強さ（スコア）の配列となる。

#### ③ Time Delay Aggregation（遅延合成）
計算されたスコア $\mathcal{R}_{QK}$ をもとに、上位 $k$ 個（$k = c \log L$）のラグ $\tau_1, \tau_2, \dots, \tau_k$ を選出する。選ばれたラグの分だけ実体データ $V$ を時間軸に沿ってシフト（Roll）させ、Softmaxで正規化した重みを用いて加重平均をとる。

$$
\hat{\mathcal{R}}_{QK}(\tau_i) = \text{Softmax}(\mathcal{R}_{QK}(\tau_i))
$$
$$
\text{Auto-Correlation}(Q, K, V) = \sum_{i=1}^{k} \hat{\mathcal{R}}_{QK}(\tau_i) \odot \text{Roll}(V, \tau_i)
$$

この操作により、ランダムなノイズが相殺され、真の周期パターンのみが抽出・強化されたテンソル（$\text{new\_V}$）が出力される。

---

### 3.1.3 Series Decomposition（時系列分解）

Transformerにおける Layer Normalization の代替として機能する、時系列特有の正規化・分解ブロックである。入力テンソル $\mathcal{X}$ を、移動平均（Moving Average）を用いてトレンド成分 $\mathcal{X}_t$ と季節成分 $\mathcal{X}_s$ に分離する。

窓幅 $k$ の平均化プーリング（$\text{AvgPool}$）を用いて、波形の局所的なベースライン（トレンド）を抽出する。

$$
\mathcal{X}_t = \text{AvgPool}(\mathcal{X})
$$

その後、元の波形からトレンド成分を減算することで、ベースラインが $0$ に補正された純粋な季節成分（ギザギザの波形）を得る。

$$
\mathcal{X}_s = \mathcal{X} - \mathcal{X}_t
$$
$$
\text{SeriesDecomp}(\mathcal{X}) = (\mathcal{X}_s, \mathcal{X}_t)
$$

エンコーダにおいては、出力 $(\mathcal{X}_s, \mathcal{X}_t)$ のうち、トレンド成分 $\mathcal{X}_t$ は未来予測に不要なノイズとして破棄され、季節成分 $\mathcal{X}_s$ のみが次へと伝播する。

---

### 3.1.4 Feed-Forward Network (FFN) と最終出力

自己相関と1回目の分解を経た季節成分 $\mathcal{S}_{en}^{l, 1}$ は、位置ごとの特徴表現を深めるために、2層の全結合層と活性化関数からなる Feed-Forward Network に入力される。

$$
\text{FeedForward}(x) = \text{Activation}(x W_1 + b_1) W_2 + b_2
$$

FFNを通過したのち、再び残差接続と SeriesDecomp を経ることで、FFNによる非線形変換で生じた微小なトレンド歪みを完全に除去する。

以上の $l$ 層における一連の処理を $N$ 回繰り返すことで、エンコーダは最終出力 を生成する。このテンソルは、過去の入力時系列から一切のトレンドとノイズが削ぎ落とされた**「純度100%の過去の周期記憶」**として、デコーダの Cross Auto-Correlation 層へと引き継がれる。



































参考
Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting,Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang

Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting,Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long