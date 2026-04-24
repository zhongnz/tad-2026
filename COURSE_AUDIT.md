# Text as Data (2026) — Deep Course Audit + Learning Playbook

This audit is designed to help you **master both the slides and the code notebooks** as one coherent system.

---

## 1) What this repo contains (curriculum map)

### Core lecture arc (slides)
The slide deck sequence is a clean progression from foundations to modern LLM workflows:

1. `slides/01_course-intro.pdf` — intro and framing
2. `slides/02_tokens.pdf` — tokenization and linguistic units
3. `slides/03_distance.pdf` — vector spaces, similarity, dimensionality
4. `slides/04_ML-with-text.pdf` — supervised learning with text features
5. `slides/05_word_embeddings.pdf` — dense semantics
6. `slides/06_sequences_attention.pdf` — sequence models and attention mechanics
7. `slides/07_transformers_encoders.pdf` — transformer encoders/BERT-style ideas
8. `slides/08_decoder_LLMs.pdf` — decoder-only LLMs/GPT framing
9. `slides/09_IFT_RLHF.pdf` — alignment, instruction tuning, RLHF
10. `slides/10_reasoning_verifiers.pdf` — reasoning and verification ideas
11. `slides/11_RAG_agents.pdf` — retrieval + agents stack
12. `slides/12_LLMs_research.pdf` — research applications
13. `slides/13_LLMs_society.pdf` — societal/policy implications

Additional domain talk:
- `slides/12_StateLaws_Pres.pdf` (policy/state-law presentation)

### Code labs (notebooks)
The notebook line is mostly week-aligned and conceptually consistent with slides:

- Week 1: `notebooks/01_intro-text.ipynb`, `01_intro_text_lab.ipynb`
- Week 2: `notebooks/02_tokens.ipynb`
- Week 3: `notebooks/03_unsupervised.ipynb`, `03_unsupervised_lab.ipynb`
- Week 4: `notebooks/04_ML_DL.ipynb`
- Week 5: `notebooks/05_Embeddings.ipynb`
- Week 6: `notebooks/06_Attention.ipynb`
- Week 7: `notebooks/07_Attention_Apps.ipynb`
- Week 8: `notebooks/08_GPT_Imp_Train.ipynb`
- Week 9: `notebooks/09_Decoding_Finetuning.ipynb`
- Week 10: `notebooks/10_RLHF.ipynb`
- Week 11: `notebooks/11_RAG.ipynb`

### Practice + assessment assets
- Homework notebooks: `homeworks/homework_01.ipynb` … `homeworks/homework-10.ipynb`
- Worked homework solutions: `homeworks/solutions/*.ipynb`
- Exam prep: `exam_review/Text-as-Data-2026-Practice-Exam.pdf` and solution guide HTML

---

## 2) Dependency / tooling profile (what skills the course actually exercises)

Across notebooks, the most frequent libraries include:

- Core ML/data: `sklearn`, `numpy`, `pandas`, `matplotlib`
- NLP classical + linguistic: `nltk`, `spacy`, `gensim`
- Deep learning: `torch`
- LLM ecosystem: `transformers`, `tiktoken`, `trl`, `datasets`, `llama_index`
- Topic/discovery apps: `bertopic`, `deeplatent`

Interpretation: this is intentionally a **hybrid curriculum**:
1) classical NLP/ML foundations, then 2) modern transformer/LLM engineering.

---

## 3) Conceptual dependency graph (how to learn in the right order)

If you want high retention, learn this as a graph, not as isolated weeks.

### Layer A — Representation foundations
- Raw text cleaning, parsing, tokenization
- Bag-of-words / n-grams / tf-idf
- Similarity metrics (esp. cosine) + dimensionality reduction intuition

### Layer B — Supervised text prediction
- Feature pipelines (vectorization + model)
- Baselines (linear models, boosting) and evaluation discipline

### Layer C — Dense semantics and neural sequence ideas
- Embeddings and distributional meaning
- Attention as differentiable retrieval
- Encoder/decoder transformer architecture choices

### Layer D — LLM operation and adaptation
- GPT pretraining mechanics
- Decoding strategies and their behavioral tradeoffs
- Fine-tuning (classification + instruction)
- RLHF and alignment objectives (reward + KL)

### Layer E — Retrieval + tool use + deployment constraints
- RAG chunking/indexing/retrieval design
- Agentic orchestration and reasoning scaffolds
- Reliability/societal implications

This dependency path mirrors the repo’s progression and minimizes confusion when advanced topics appear.

---

## 4) Week-by-week mastery checklist (slides + notebook + homework loop)

For each week:

1. **Slides first (45–75 min):**
   - Extract 5 key claims
   - Write 2 “why this matters” bullets
2. **Notebook pass 1 (read-only, 30–45 min):**
   - Identify inputs, transformations, outputs
3. **Notebook pass 2 (active, 45–90 min):**
   - Re-run all cells
   - Change one hyperparameter + one preprocessing choice
4. **Homework attempt (time-boxed):**
   - Solve before reading solutions
5. **Post-mortem (15 min):**
   - Record 3 mistakes and corrected rule

If done weekly, this creates a compounding feedback loop from concept → implementation → evaluation.

---

## 5) High-yield code comprehension targets by module

### Weeks 1–3 (classic NLP)
- Understand exactly how tokenization changes downstream feature spaces.
- Be able to explain when cosine similarity fails (sparsity/domain drift).
- For clustering/topic work, justify parameter choices qualitatively.

### Weeks 4–5 (ML + embeddings)
- Distinguish feature engineering gains vs model-class gains.
- Explain why embedding-based similarity can outperform sparse vectors in semantic tasks.
- Know where leakage happens in vectorizer/model splits.

### Weeks 6–8 (attention + GPT internals)
- Be able to derive attention dimensions and tensor shapes from memory.
- Explain residual path + layer norm + MLP block roles in transformers.
- Contrast encoder attention masks vs causal decoder masks.

### Weeks 9–11 (decoding, RLHF, RAG)
- Describe top-k/top-p/temperature tradeoffs without hand-waving.
- Explain RLHF objective pieces (policy reward vs KL regularization).
- Defend chunk-size/overlap/retriever choices in RAG experiments.

### Weeks 12–13 (research + policy)
- Link technical design choices to reliability and societal outcomes.
- Build argument chains: model behavior → deployment decision → policy risk.

---

## 6) Where students usually struggle (and how to preempt it)

1. **Confusing tokenization levels** (wordpiece/subword/word):
   - Fix: create side-by-side tokenization examples for 3 different tokenizers.

2. **Treating metrics as decoration**:
   - Fix: for every model run, write “metric selected because ___ and fails when ___”.

3. **Jumping to LLMs without baseline grounding**:
   - Fix: keep one linear or tree baseline in every new task.

4. **RAG retrieval ignored while prompt tuning dominates effort**:
   - Fix: spend 70% of effort on corpus hygiene/chunking/index quality first.

5. **RLHF viewed as magic**:
   - Fix: explicitly track policy, reference policy, reward model, and KL coefficient effects.

---

## 7) Concrete study system you can start immediately

### A) Slide memory protocol (active recall)
For each slide deck, produce:
- 10 flashcards (concept ↔ definition)
- 5 “explain like I’m a teammate” prompts
- 3 equation/mechanism reconstructions from memory

### B) Notebook mastery protocol
For each notebook, maintain a short log:
- **Goal of notebook** (one sentence)
- **Core data transformations** (bullet list)
- **Model/objective used**
- **Most sensitive hyperparameter**
- **One ablation you ran and result**

### C) Homework-to-exam bridge
After each homework:
- rewrite one exercise as an exam-style prompt,
- solve it on paper,
- then verify against code.

This converts coding fluency into exam fluency.

---

## 8) 14-day intensive revision plan (if exam is close)

### Days 1–4: Foundations sprint
- D1: Week 1–2 (tokenization + preprocessing)
- D2: Week 3 (distance, tf-idf, clustering)
- D3: Week 4 (supervised ML pipeline)
- D4: Week 5 (embeddings)

### Days 5–9: Transformer sprint
- D5: attention mechanics
- D6: encoders
- D7: decoder LLMs/GPT internals
- D8: decoding + fine-tuning
- D9: RLHF

### Days 10–12: Systems sprint
- D10: RAG implementation details
- D11: reasoning/verifiers + research use
- D12: society/policy + tradeoff essays

### Days 13–14: Mock + patch
- Solve practice exam under timing.
- Patch weakest 3 topics with targeted notebook reruns.

---

## 9) “If you only remember 12 things” list

1. Tokenization determines everything downstream.
2. Sparse vectors are strong baselines.
3. Cosine similarity is geometry, not semantics.
4. Scaling and leakage handling matter as much as model choice.
5. Embeddings encode distributional meaning, imperfectly.
6. Attention is weighted information routing.
7. Transformer blocks = attention + MLP + residual/normalization.
8. Decoder masking creates autoregressive behavior.
9. Decoding strategy controls style/risk/diversity.
10. RLHF is optimization with constraints, not just reward chasing.
11. RAG quality is mostly retrieval quality.
12. Technical design has policy and social consequences.

---

## 10) Suggested next step (practical)

Start with a **single integrated pass** over Weeks 1–3 today:
- read each deck’s first 20–30 key slides,
- run corresponding notebook end-to-end,
- complete one mini-ablation,
- summarize in 1 page.

Repeat this pattern weekly and you’ll retain both conceptual and coding depth.
