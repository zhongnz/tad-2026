# Text as Data (2026) — Comprehensive Course Audit + Mastery Guide

This guide is designed for **deep mastery** of both the conceptual material (slides) and implementation work (notebooks/homeworks), with enough structure to support exam prep, project work, and long-term retention.

---

## 1) Repository-wide curriculum map

### 1.1 Lecture sequence (theory spine)
The slide arc moves from foundational NLP to modern LLM systems and societal implications:

1. `slides/01_course-intro.pdf` — framing, motivation, course foundations
2. `slides/02_tokens.pdf` — tokenization and linguistic preprocessing
3. `slides/03_distance.pdf` — vector spaces, dimensionality, similarity
4. `slides/04_ML-with-text.pdf` — supervised learning with text features
5. `slides/05_word_embeddings.pdf` — dense representations and semantics
6. `slides/06_sequences_attention.pdf` — sequence modeling and attention
7. `slides/07_transformers_encoders.pdf` — transformer encoders (BERT-like)
8. `slides/08_decoder_LLMs.pdf` — decoder LLMs (GPT-like)
9. `slides/09_IFT_RLHF.pdf` — instruction tuning, alignment, RLHF
10. `slides/10_reasoning_verifiers.pdf` — reasoning and verifier paradigms
11. `slides/11_RAG_agents.pdf` — retrieval + agents + orchestration
12. `slides/12_LLMs_research.pdf` — LLM workflows for research tasks
13. `slides/13_LLMs_society.pdf` — societal, legal, policy implications

Supplementary domain deck:
- `slides/12_StateLaws_Pres.pdf`

### 1.2 Notebook sequence (implementation spine)

- Week 1: `notebooks/01_intro-text.ipynb`, `notebooks/01_intro_text_lab.ipynb`
- Week 2: `notebooks/02_tokens.ipynb`
- Week 3: `notebooks/03_unsupervised.ipynb`, `notebooks/03_unsupervised_lab.ipynb`
- Week 4: `notebooks/04_ML_DL.ipynb`
- Week 5: `notebooks/05_Embeddings.ipynb`
- Week 6: `notebooks/06_Attention.ipynb`
- Week 7: `notebooks/07_Attention_Apps.ipynb`
- Week 8: `notebooks/08_GPT_Imp_Train.ipynb`
- Week 9: `notebooks/09_Decoding_Finetuning.ipynb`
- Week 10: `notebooks/10_RLHF.ipynb`
- Week 11: `notebooks/11_RAG.ipynb`
- Week 12: `notebooks/12_Agents_Verifiers.ipynb` (new)

### 1.3 Assignments and exam prep

- Student homeworks: `homeworks/homework_01.ipynb` … `homeworks/homework-10.ipynb`
- Solutions: `homeworks/solutions/*.ipynb`
- Practice exam + solution guide:
  - `exam_review/Text-as-Data-2026-Practice-Exam.pdf`
  - `exam_review/Text-as-Data-2026-Solution-Guide.html`

---

## 2) What skills this course actually trains

This is a deliberately hybrid program: **classic NLP/ML foundations + modern LLM engineering**.

### 2.1 Statistical/NLP foundations
- Corpus inspection, normalization, tokenization
- Bag-of-words/n-grams/tf-idf style representations
- Similarity geometry and dimensionality reduction
- Unsupervised modeling (topic/discovery/clustering)

### 2.2 Predictive modeling foundations
- End-to-end supervised pipelines
- Data splits, leakage prevention, validation discipline
- Baseline-first modeling habits

### 2.3 Neural and LLM engineering
- Embeddings and representation learning
- Attention mechanisms and transformer blocks
- Decoder-only LLM training and generation
- Fine-tuning, RLHF, and reward-based adaptation
- RAG design and agent/verifier patterns

### 2.4 Evaluation and policy judgment
- Experimental rigor and reproducibility
- Failure-mode analysis and robustness reasoning
- Deployment tradeoffs (safety, governance, social outcomes)

---

## 3) Concept dependency graph (study in this order)

### Layer A — Text representation basics
Preprocessing, tokenization, sparse features, vector geometry.

### Layer B — Prediction with text
Classical supervised ML pipelines and evaluation.

### Layer C — Dense semantics
Embeddings and neural representations.

### Layer D — Attention and transformers
Attention mechanics → encoder/decoder architectures.

### Layer E — LLM adaptation
Decoding, fine-tuning, instruction tuning, RLHF.

### Layer F — Systems and reliability
RAG, agents/verifiers, governance/policy impacts.

**Rule of thumb:** if Layer A/B is weak, Layer D/E/F will feel magical instead of understandable.

---

## 4) Detailed weekly deep-learning plan (slides + code + homework)

For each week, complete this 5-phase loop:

1. **Slide extraction (45–90 min):**
   - Write 5 key ideas, 3 mechanisms, 2 open questions.
2. **Notebook walkthrough (30–45 min):**
   - Identify data source, transformations, objective, outputs.
3. **Notebook replication + ablations (60–120 min):**
   - Re-run all cells, then do 2 ablations.
4. **Homework attempt (time-boxed):**
   - Solve cold before checking solutions.
5. **Post-mortem (20 min):**
   - Record errors, causes, and correction rules.

### Week-by-week focus targets

#### Week 1 — Intro/text data pipeline
- Understand ingestion, cleaning, and exploratory text diagnostics.
- Explain each preprocessing step and potential side effects.

#### Week 2 — Tokenization and linguistics
- Compare word-level, subword, and tokenizer-family differences.
- Know when linguistic annotations help or hurt downstream tasks.

#### Week 3 — Similarity and unsupervised analysis
- Derive cosine intuition from vector geometry.
- Explain tf-idf weighting qualitatively and operationally.

#### Week 4 — Supervised text ML
- Build baseline pipelines with explicit leakage controls.
- Interpret metrics in context (class imbalance, error costs).

#### Week 5 — Embeddings
- Contrast sparse vs dense representations on semantic behavior.
- Diagnose when embedding neighborhoods are misleading.

#### Week 6 — Attention mechanics
- Track tensor shapes through Q/K/V operations.
- Explain attention scores, weights, and context vectors.

#### Week 7 — Transformer applications
- Connect architecture choices to performance tradeoffs.
- Evaluate topic/discovery applications critically.

#### Week 8 — GPT implementation
- Understand each decoder block component and data flow.
- Reconstruct a forward pass from memory.

#### Week 9 — Decoding and fine-tuning
- Compare top-k/top-p/temperature with concrete failure cases.
- Explain architecture surgery for task-specific fine-tuning.

#### Week 10 — RLHF
- Explain reward model role and KL penalty rationale.
- Distinguish alignment improvement from over-optimization risks.

#### Week 11 — RAG
- Treat chunking + retrieval as first-order quality drivers.
- Evaluate retriever quality before prompt engineering.

#### Week 12 — Agents + verifiers (new notebook)
- Distinguish planning, tool-use, and verification loops.
- Compare generation-only vs verifier-augmented workflows.
- Build explicit failure taxonomies (retrieval, reasoning, tool I/O).

#### Week 13 — LLMs in society/policy
- Build argument chains from model behavior → policy impact.
- Surface tradeoffs among capability, safety, and governance.

---

## 5) Code mastery rubric (how to know you truly learned it)

For every notebook, score yourself 0–2 on each dimension:

1. **Pipeline clarity:** Can you explain input→output flow?
2. **Mechanism depth:** Can you explain why each block exists?
3. **Ablation skill:** Can you design and interpret 2 meaningful perturbations?
4. **Failure analysis:** Can you identify likely error modes?
5. **Reproducibility:** Can you rerun from clean runtime with consistent outcome?

**Mastery threshold:** 8/10+ consistently across weeks.

---

## 6) High-value ablations by module (do these for deep understanding)

### Tokens/similarity modules
- Change tokenizer granularity and compare feature sparsity + performance.
- Remove stopword filtering and inspect nearest-neighbor shifts.

### Supervised ML modules
- Swap linear baseline vs stronger model class; compare gains.
- Standardization on/off where relevant; inspect metric movement.

### Embeddings/attention modules
- Reduce embedding size; inspect semantic coherence degradation.
- Modify attention head count; track quality/compute tradeoff.

### Decoding/fine-tuning/RLHF modules
- Run temperature sweep (low/med/high) and log behavior types.
- Adjust KL coefficient in RLHF context and record policy drift.

### RAG/agents/verifiers modules
- Sweep chunk size/overlap and retrieval top-k.
- Compare no-verifier vs verifier-loop output reliability.

---

## 7) Common pitfalls and exact corrective actions

1. **“Tokenization is just preprocessing.”**
   - Correction: always inspect tokenization outputs before training.

2. **“Higher metric means better model.”**
   - Correction: attach metric to objective + deployment risk.

3. **“LLM prompt tweaks fix everything.”**
   - Correction: optimize data, retrieval, and evaluation first.

4. **“RLHF reward improvement = alignment solved.”**
   - Correction: inspect reward hacking symptoms and KL behavior.

5. **“RAG is a generation problem.”**
   - Correction: treat indexing/chunking/retrieval as primary levers.

6. **“Agents are just multi-step prompts.”**
   - Correction: separate planning, execution, memory, and verification layers.

---

## 8) Slide-learning system (active recall protocol)

For every slide deck:
- Create 10 flashcards (term ↔ mechanism).
- Write 5 “teach this to a peer” prompts.
- Reconstruct 3 diagrams/equations from memory.
- Record 2 real-world scenarios where concepts can fail.

Weekly spaced repetition cadence:
- Day 0 (after class), Day 2, Day 7, Day 14.

---

## 9) Notebook-learning system (engineering protocol)

For each notebook, maintain a compact lab log:

- Goal of notebook (1 sentence)
- Dataset and preprocessing decisions
- Model/objective summary
- Key hyperparameters and default values
- 2 ablations + observed changes
- Error analysis notes
- “If I had one more hour” improvement plan

This turns passive execution into reusable ML/LLM engineering judgment.

---

## 10) Homework-to-exam conversion pipeline

After completing each homework:

1. Convert one coding exercise into a conceptual exam question.
2. Solve from memory (paper-first).
3. Re-implement quickly in notebook.
4. Compare and write one “general rule” learned.

Repeat for 2–3 exercises per week to close theory/practice gaps.

---

## 11) 21-day intensive prep plan (thorough version)

### Phase I (Days 1–7): foundations + pipelines
- D1–2: intro + tokenization
- D3–4: similarity + unsupervised
- D5–7: supervised ML + embeddings

### Phase II (Days 8–14): transformer/LLM depth
- D8–9: attention + encoders
- D10–11: decoder LLM internals + decoding
- D12–14: fine-tuning + RLHF

### Phase III (Days 15–21): systems + synthesis
- D15–16: RAG end-to-end
- D17–18: agents/verifiers
- D19: society/policy synthesis
- D20: full practice exam simulation
- D21: targeted patching of weakest modules

---

## 12) “Remember these 15 forever” compression list

1. Data quality dominates clever modeling.
2. Tokenization choices propagate through the whole stack.
3. Sparse baselines are hard to beat without careful setup.
4. Cosine measures geometry, not truth.
5. Leakage can invalidate excellent-looking metrics.
6. Embeddings capture distributional patterns, not guaranteed facts.
7. Attention is differentiable routing under constraints.
8. Transformer stability depends on normalization/residual design.
9. Decoding policy controls behavior as much as weights do.
10. Fine-tuning shifts capabilities and risks together.
11. RLHF is constrained optimization, not pure reward maximization.
12. Retrieval quality is the main bottleneck in RAG.
13. Agents require orchestration discipline, not just bigger prompts.
14. Verifiers can raise reliability but introduce latency/complexity.
15. Technical design decisions create policy and social consequences.

---

## 13) Immediate next actions (today)

If you want to learn thoroughly starting now:

1. Do Weeks 1–3 in one integrated pass.
2. For each notebook, run 2 ablations and write a 6-line lab log.
3. Build 20 flashcards total from slides 1–3.
4. Solve one practice exam question timed (closed notes).
5. Review errors and create a one-page “mistakes ledger.”

Do this once and you’ll feel a major jump in both confidence and depth.
