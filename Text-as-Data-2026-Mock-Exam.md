# Text as Data — Spring 2026
## Mock Exam

**Total: 45 points | 60 minutes**

**As a rough guide, one credit point corresponds to at least one sentence in your answer.**

*This mock exam covers one theoretical question and one code-reading question per topic. No programming required — for code questions, read the snippet and answer in writing.*

---

### Question 1 — Tokenization (4 points)

**a.** Consider the word **"preprocessing"**. A word-level tokenizer that has never seen this word assigns it a single `[UNK]` token. A BPE tokenizer trained on the same corpus produces the sequence `["pre", "##process", "##ing"]`. Explain *why* BPE handles this unseen word more informatively than the word-level tokenizer, and what downstream modeling advantage this provides. [2 points]

---

**b.** *(Code)* Look at the following snippet from the notebook. `integers` is a list of token IDs and `context_size = 4`.

```python
for i in range(1, context_size + 1):
    context = integers[:i]
    desired = integers[i]
    print(context, "---->", desired)
```

Explain what this loop is doing. Why is `desired` the token at position `i` rather than `i - 1`? How many `(context, target)` pairs does a document of $T$ tokens produce under this scheme? [2 points]

---

### Question 2 — TF-IDF and Vector Space (4 points)

**a.** A corpus contains five documents. Term **"epoch"** appears in all five. Term **"backpropagation"** appears in only one document (Document 1), where it occurs 10 times. Without computing exact numbers, which term has a higher TF-IDF score in Document 1? Justify your answer by reasoning through both the TF and IDF components. [2 points]

---

**b.** Write the formula for cosine similarity between two vectors **u** and **v**. Why is cosine similarity preferred over Euclidean distance when comparing document vectors of different lengths? Illustrate with a one-sentence concrete example. [2 points]

---

### Question 3 — Dimensionality Reduction and Topic Modeling (4 points)

**a.** Explain the generative assumption behind Latent Dirichlet Allocation (LDA). What does LDA assume about how a document is produced? What is a "topic" in LDA, and how are words generated from topics? [3 points]

---

**b.** *(Code)* From the notebook:

```python
sil_scores = []
for n in range(2, 20):
    km = KMeans(n_clusters=n)
    km.fit(X_tfidf)
    sil_scores.append(silhouette_score(X_tfidf, km.labels_))
```

What does the silhouette score measure for a given clustering? How would you use `sil_scores` to choose the best number of clusters? [1 point]

---

### Question 4 — Word Embeddings (4 points)

**a.** In a trained embedding space, the following vector arithmetic holds approximately:

$$\text{"king"} - \text{"man"} + \text{"woman"} \approx \text{"queen"}$$

Explain geometrically what this means. What property of the embedding space makes this arithmetic work? [2 points]

---

**b.** *(Code)* From the notebook:

```python
from gensim.models import Word2Vec

w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)
w2v.wv.most_similar('neural')
```

What does `sg=1` select? What does `window=5` control during training? What does `most_similar('neural')` return — what is it measuring and how? [2 points]

---

### Question 5 — The Self-Attention Mechanism (5 points)

**a.** Write the formula for scaled dot-product attention. Why is the scaling factor $1/\sqrt{d_k}$ necessary? What goes wrong at the softmax step if it is omitted when $d_k$ is large? [3 points]

---

**b.** *(Code)* The following is taken from the notebook. `inputs` has shape `[6, 3]` (6 tokens, embedding dimension 3) and `d_out = 2`.

```python
W_query = torch.nn.Parameter(torch.rand(d_in, d_out))   # d_in=3, d_out=2
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out))
W_value = torch.nn.Parameter(torch.rand(d_in, d_out))

query_2       = x_2 @ W_query          # x_2 is one token vector, shape [3]
keys          = inputs @ W_key
attn_scores_2 = query_2 @ keys.T / d_out**0.5
```

State the shape of each of the following: `W_query`, `query_2`, `keys`, `attn_scores_2`. [2 points]

---

### Question 6 — Multi-Head Attention and Transformer Blocks (4 points)

**a.** What is the difference between single-head and multi-head attention? What additional representational capacity do multiple heads provide? Using the sentence **"The model failed because the data was noisy"**, give a concrete example of what two different attention heads might attend to differently. [3 points]

---

**b.** *(Code)* What does the following code implement? Describe what the matrix `masked` looks like for a 4-token sequence, and explain what happens to the softmax weights for the masked positions.

```python
mask   = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
```

[1 point]

---

### Question 7 — Positional Embeddings and Encoder vs. Decoder (5 points)

**a.** Compare BERT (encoder-only) and GPT-2 (decoder-only): what type of attention masking does each use and why, what is each model's pre-training objective, and for which types of tasks is each architecture better suited? Give one example task for each. [3 points]

---

**b.** *(Code)* From the notebook:

```python
token_embeddings = token_embedding_layer(inputs)        # inputs shape: [8, 4]
pos_embeddings   = pos_embedding_layer(torch.arange(4)) # shape: [4, 256]
input_embeddings = token_embeddings + pos_embeddings    # shape: [8, 4, 256]
```

What does each dimension of the output shape `[8, 4, 256]` represent? What information does adding `pos_embeddings` to `token_embeddings` inject into the representation? [2 points]

---

### Question 8 — Language Model Training and Decoding (5 points)

**a.** Describe the next-token prediction training objective for GPT-2: what is the input, what is predicted, which loss is minimized, and how many training targets does a sequence of $T$ tokens produce? [3 points]

---

**b.** *(Code)* The `GPTDatasetV1` class creates training pairs with a sliding window:

```python
for i in range(0, len(token_ids) - max_length, stride):
    input_chunk  = token_ids[i : i + max_length]
    target_chunk = token_ids[i + 1 : i + max_length + 1]
```

What is `target_chunk` relative to `input_chunk` — how do they differ? What is the effect of using `stride < max_length` versus `stride = max_length` on the number of training examples and on data redundancy? [2 points]

---

### Question 9 — Fine-Tuning and RLHF (5 points)

**a.** Explain the three stages of the RLHF pipeline: (i) SFT — what data and objective; (ii) Reward Model — how it is trained from human feedback and what a pairwise preference loss computes; (iii) PPO — what the policy learns and what signal drives updates. [4 points]

---

**b.** *(Code)* What is wrong with the following RLHF setup — identify the single methodological flaw:

```python
policy    = GPT2LMHeadModel.from_pretrained('gpt2')
ref_model = policy   # reference model set to the same object as policy
optimizer.step()     # updates policy weights in-place
```

[1 point]

---

### Question 10 — Retrieval-Augmented Generation (5 points)

**a.** Compare sparse retrieval (TF-IDF / BM25) and dense retrieval (embedding-based): how does each score relevance, and what is one concrete advantage of each? [3 points]

---

**b.** *(Code)* From the notebook:

```python
tfidf        = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(corpus)
query_vec    = tfidf.transform([query])
scores       = cosine_similarity(query_vec, tfidf_matrix)
top_k_idx    = scores.argsort()[0][-3:][::-1]
```

If `corpus` has $N$ documents and vocabulary size $V$, what is the shape of `tfidf_matrix`? What does `top_k_idx` contain at the end? What is the key conceptual difference between this retrieval approach and a dense retriever that uses a neural embedding model? [2 points]

---

*End of Mock Exam — 45 points total*

*Topics: Tokenization (4) · TF-IDF & Vector Space (4) · Topic Modeling (4) · Word Embeddings (4) · Self-Attention (5) · Multi-Head Attention & Transformer Blocks (4) · Positional Embeddings & Encoder vs Decoder (5) · LM Training & Decoding (5) · Fine-Tuning & RLHF (5) · RAG (5)*
