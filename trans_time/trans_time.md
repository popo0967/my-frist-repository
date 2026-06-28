# Transformerによる時系列予測

## 概要
本ドキュメントは、Transformerによる時系列予測に関して、3つの手法で学習してみます。
論文は以下を参考にしています
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

Informerは、各 $q_i$ がどれだけ「重要（アクティブ）か」を、一様分布とのKL情報量に基づく指標 $\bar{M}(q_i, K)$ で評価する。

$$\bar{M}(q_i, K) = \ln \sum_{j=1}^{L_K} e^{\frac{q_i k_j^T}{\sqrt{d_k}}} - \frac{1}{L_K} \sum_{j=1}^{L_K} \frac{q_i k_j^T}{\sqrt{d_k}}$$

この指標が高い（他のデータに対して強い関心を持つ）上位 $u$ 個のQueryだけを選抜して $\bar{Q}$ とし、計算を行う。

$$\text{ProbAttention}(Q, K, V) = \text{softmax}\left(\frac{\bar{Q} K^T}{\sqrt{d_k}}\right) V$$

選抜数 $u$ は $u = c \cdot \ln L_Q$ （$c$ は定数）に制限されるため、計算量は $O(L \ln L)$ へと劇的に削減される。

**② Distilling（蒸留操作）**
Attention層の直後に、1次元のMax Pooling層を挟むことで、時間軸方向のデータサイズを半分に圧縮する。
$$X_{j+1}^t = \text{MaxPool}\left( \text{ELU}\left( \text{Conv1d}( [X_j^t]_{AB} ) \right) \right)$$

**【数値例】**
$L=1000$ のデータを入れた場合、通常のAttentionは $1000 \times 1000 = 1,000,000$ 回の計算が必要になる。しかしProbSparse Attentionでは、たとえば $u = 5 \ln(1000) \approx 34$ 個の優秀なQueryだけが選ばれるため、$34 \times 1000 = 34,000$ 回の計算で済む。さらにDistillingによって、次の層に渡るデータ数は $500 \to 250$ と半減していくため、メモリがパンクしない。

### 2.3 デコーダ層に関して

デコーダは、RNNのような自己回帰（1歩ずつ予測する手法）の「遅さ」と「誤差の蓄積」を防ぐため、**Generative Style Decoder（一発出しデコーダ）**を採用している。

デコーダへの入力 $\mathcal{X}_{de}$ は、既知の直近の過去データ $X_{token}$（長さ $L_{token}$）と、予測したい未来の枠を表すゼロ行列 $X_0$（長さ $L_y$）を結合したプレースホルダーである。
$$\mathcal{X}_{de} = \text{Concat}(X_{token}, X_0) \in \mathbb{R}^{(L_{token} + L_y) \times d_{model}}$$

デコーダ内部では、まず未来の情報をカンニングしないためのMasked Multi-head Attentionが行われ、その後、エンコーダが抽出した過去の圧縮記憶（$K, V$）と、自身の持つプレースホルダー（$Q$）をぶつけるCross-Attentionが行われる。
最後に全結合層（Linear）を1回だけ通し、最終的な予測値 $Y_{predict}$ を出力する。

**【数値例】**
過去5日分のデータから、未来30日分を予測したいとする。
デコーダには `[過去5日分のデータ] + [0.0が30個並んだ配列]` が入力される。Cross-Attentionによって「このゼロの枠には、エンコーダのどの記憶を当てはめるべきか」が計算され、最後のDNN層を通った瞬間に、30個のゼロが一気に「30日分の具体的な予測数値」へと書き換わって出力される。

### 2.4 特徴（まとめ）

Informerが時系列モデリングの歴史にもたらした主要な特徴は以下の3点に集約される。

1. **$O(L \ln L)$ への計算量削減:** ProbSparse AttentionとDistillingの組み合わせにより、長期間の過去データをメモリを枯渇させることなく入力できるようになった。
2. **時間的セマンティクスの完全な統合:** Temporal Encodingにより、現実世界のカレンダー情報（曜日や時間帯による周期性）をディープラーニングの重み空間にマッピングすることに成功した。
3. **誤差蓄積の回避と高速推論:** Generative Style Decoderにより、長期間の未来予測を一回の順伝播（Forward pass）で完結させ、推論速度と安定性を飛躍的に向上させた。