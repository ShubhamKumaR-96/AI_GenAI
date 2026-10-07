# MY NOTES — LLM Fundamentals: From Neural Nets to GPT
*Source: 0031_Detailed notes.pdf (AlgoCamp AI Engineering) | Compiled: Sep 30, 2026*
*Companion to: 0030_Class Notes.pdf*

---

## 0. One-Line Essence (memorize first)

| # | Section | One line |
|---|---------|----------|
| 1 | What an LLM is | A neural network that predicts the next token — run in a loop |
| 2 | AI ⊃ ML ⊃ DL ⊃ LLM | Nested subsets; ML = learn rules from data, not hand-code them |
| 3 | NN = function approximator | Fits curves a straight line can't; training = tuning weights to minimize loss |
| 4 | Matrix mult → GPU | NNs are parallel matrix math; thousands of simple cores beat a few powerful ones |
| 5 | What LLMs solved | One general model → cost of new task dropped from "retrain" to "write a prompt" |
| 6 | Transformer (2017) | Every token weighs every token, in parallel — fixed RNN's sequential + forgetful limits |
| 7 | Tokenization | Text → sub-word IDs; BPE merges frequent pairs; byte-level BPE = nothing out of vocabulary |
| 8 | Generation | softmax → probability distribution → sample → repeat; temperature = randomness knob |
| 9 | 3-phase training | Pre-train (→ base) → SFT (→ instruct) → RLHF (→ aligned) |
| 10 | Base model | Fluent but not an assistant; knowledge frozen in weights |
| 11 | Scaling laws | Loss falls in smooth power-laws with params × data × compute; Chinchilla: balance them |
| 12 | Training economics | ~$100M per frontier model → AI engineering = building on top of others' models |
| 13 | What's next | OpenAI API hands-on; classic ML (KNN/SVM) still valid, just not center of gravity |

---

## 1. The 5 Big Ideas (deep explanations)

### Idea 1 — LLM = next-token predictor, capability is a SIDE EFFECT
```
input tokens → [ neural network ] → probability distribution over next token
"The cat sat on the" → mat 0.31, floor 0.12, roof 0.04 ...
```
- Pick a token → append → predict again → loop = essays, code, chat.
- **Mental hook:** Don't think "the model knows things." Think "it learned what token statistically tends to come next."
- **Aha moment:** Chat/reasoning were never explicitly programmed — they *emerge* from doing one tiny task extremely well at scale.

### Idea 2 — The learning ladder (and how humans do it too)
```
Traditional programming:  you write rules     → if temperature > 30 → "hot"
Machine learning:         you show examples   → model infers the rules itself
Deep learning:            many-layer NN       → abstract patterns: edges → shapes → objects
                                                          chars → words → meaning
```
- Human analogy: see enough cats → identify an unseen cat.
- Honest caveat from class: stock "earnings up → price up" is NOT clean — real signals are noisy and multi-causal. **Models approximate messy reality; they don't capture it perfectly.**

### Idea 3 — Why GPU won (before LLMs existed)
- NN forward pass = repeated `[input vector] × [weight matrix] → next layer`
- Output cells are computed **independently** → *embarrassingly parallel*
- **CPU vs GPU:** few maths professors (complex sequential logic) vs thousands of students each doing one small sum
- Historical twist: GPUs were built for graphics (also parallel matrix math) — the hardware was already sitting there when deep learning needed it.

### Idea 4 — Attention fixed the two sins of RNN/LSTM
| RNN/LSTM problem | Transformer fix |
|---|---|
| Reads 1 token at a time → **can't parallelize** → slow training | All tokens processed together → GPU heaven |
| Memory fades over distance → **forgetful** | Every token looks at every other token directly, regardless of distance |

- Example: *"The animal didn't cross the street because **it** was too tired."* → attention links "it" → "animal" (weights it heavily).
- Two wins: **context** (meaning from surroundings) + **parallelism** (massive-scale training became feasible).

### Idea 5 — Where does knowledge live? (the #1 beginner confusion)
| | Location | Purpose |
|---|---|---|
| **Model knowledge** | Compressed into **weights** (billions of numbers); training text is NOT stored | Pattern/statistics baked in at training |
| **RAG document lookup** | **Vector DB** stores embeddings of YOUR docs, fetched at query time | Application layer on top of an LLM |

- Keep these separate → avoids downstream confusion when we reach embeddings & RAG.

---

## 2. Key Mechanisms (formulas & knobs)

### Softmax (raw scores → probabilities summing to 1)
```
softmax(x_i) = e^(x_i) / Σ e^(x_j)
"The sky is" → { blue: 0.62, clear: 0.11, falling: 0.02, ... } → pick "blue" → loop
```

### Temperature (why same prompt ≠ same answer)
- Model gives the **same distribution** each time; generation **samples** from it.
- **Higher temperature** → flattens distribution → varied/creative
- **Temperature ≈ 0** → greedy decoding → near-deterministic (always top token)
- ⚠️ Correction to the common myth: variation comes from **stochastic sampling**, NOT from "different token windows."

### Tokenization + BPE
```
"tokenization" → ["token", "ization"] → [15496, 1634]
"AlgoCamp"     → ["Algo", "Camp"]     → [2348, 6488]
```
- **BPE:** start from bytes/chars, iteratively merge most frequent adjacent pairs.
- Vocabulary balances "few tokens per text" vs "manageable size."
- **Byte-level BPE:** worst case falls back to raw bytes → **nothing is out of vocabulary** (emoji, code, any language).
- Chain: `bits → bytes → tokens` (byte = 8 bits).

### Loss / training
- Linear: `y = wx + b` — fits only roughly linear data.
- NN: chains simple ops with **activation functions** → bends into any shape → **universal function approximator**.
- **Training = find the weights w that minimize loss** (measure error → nudge weights → repeat). LLMs = same idea × billions of weights × oceans of data.

---

## 3. The 3-Phase Training Pipeline (heart of the lecture)

```
Phase 1: PRE-TRAINING  → raw internet text, next-token objective → "base model"
Phase 2: SFT           → curated (user, good assistant) chats     → "instruct model"
Phase 3: RLHF          → humans rank answers → reward model → RL  → "aligned model"
```

**Phase 1 — Pre-training:** no labels, no "good answers" curation — just oceans of text. Learns grammar, facts, reasoning, code — all latent in statistics of human writing.

**Phase 2 — SFT example:**
```
User: How do I reverse a string in Python?
Assistant: You can use slicing: s[::-1]. For example...
```
- Scale AI, Innodata = companies built on armies of human labelers writing/rating responses.

**Phase 3 — RLHF loop:**
```
LLM generates 2+ answers → humans rank → train reward model
→ RL (classically PPO) optimizes LLM to maximize reward-model score
```
- **Analogy:** student takes practice tests (generates), teacher grades (reward model), student adjusts — a feedback loop, automated.
- **Precision point:** RLHF+PPO ≈ OpenAI/Anthropic lineage. **DeepSeek's GRPO (DeepSeek-R1)** is large-scale RL for *reasoning* — related but distinct. Don't merge the two threads.

---

## 4. Base Models — can / can't

| Can | Can't |
|---|---|
| Fluent, coherent continuations | Be an assistant (ask it a question → it may continue with MORE questions) |
| Probability-based next tokens reflecting training patterns | Complex multi-step math out of the box |
| | Memorize new info at inference (knowledge **frozen** in weights) |
| | Agentic behavior / reliably follow instructions (until post-trained) |

- **base ≈ foundation:** "foundation model" = broader term (any big pre-trained model you build on); "base" emphasizes the pre-SFT artifact. Casual use overlaps.
- **Wide vs deep:** more neurons/layer vs more layers — both raise parameter count; the balance affects what's learned & efficiency → feeds directly into scaling laws.

---

## 5. Scaling Laws & Economics

```
More PARAMETERS + more DATA + more COMPUTE → predictably lower loss
(smooth power-law curves → forecast a bigger model's quality BEFORE training it)
```

**Two assigned papers:**
1. **Kaplan et al. 2020 (OpenAI)** — "Scaling Laws for Neural Language Models" → established the power laws.
2. **Hoffmann et al. 2022 (DeepMind) = Chinchilla** — "Training Compute-Optimal LLMs" → earlier models were **undertrained**; for a compute budget, use MORE DATA relative to parameters. **A smaller model trained on more tokens beats a bigger model trained on less.**

**Upshot:** not just "make it bigger" — balance params & data against compute budget → "how many tokens should I train on?" became a first-class engineering question.

**Economics:** frontier GPT-class training ≈ **$100M** (tens of thousands of GPUs × months + data pipelines + human labeling).
→ **Why AI engineering exists:** you never train from scratch; you build value on top of nine-figure models via prompting, RAG, small adapters (fine-tuning), orchestration.

---

## 6. Comparison Tables (rapid revision)

| Topic | Old world | LLM world |
|---|---|---|
| Tasks | dataset + separately trained model per task (spam model, sentiment model...) | one general model; new task = prompt |
| Sequence models | RNN/LSTM: sequential + forgetful | Transformer: parallel + full context |
| Training labels | needed for most ML | pre-training needs none (raw text) |
| New-task cost | collect data + retrain | write a prompt |

| CPU | GPU |
|---|---|
| few cores (4–64), powerful | thousands of simpler cores |
| complex sequential logic | massive parallel arithmetic |
| maths professors | students each doing one sum |

---

## 7. Important Discussion Points

1. **"Capability is a side effect"** — provocative claim: reasoning/coding were never in the objective function; only next-token prediction was. Discuss: is that enough, or does it cap what LLMs can do?
2. **Temperature myth-busting** — learners hit nondeterminism immediately when calling APIs; teach the real cause (sampling) explicitly.
3. **Weights vs vector DB** — the confusion it causes later is predictable; separate "knowledge baked in" from "knowledge fetched."
4. **RLHF vs GRPO** — alignment (PPO, human prefs) vs reasoning elicitation (RL at scale) are two different threads often merged in casual talk.
5. **Chinchilla irony** — the "train smaller on more data" lesson, yet labs keep scaling up → discuss inference-time compute, data-scarcity, MoE as the modern resolution.
6. **Honest ML caveat** — noisy multi-causal domains (stocks) remind us models approximate reality; overclaiming = bad engineering.
7. **Classic ML question (KNN/SVM)** — still useful in broader ML, but not the center of gravity for AI-engineering jobs today.
8. **Economics shapes the field** — $100M training cost explains why prompting/RAG/fine-tuning (not training) is the profession.

---

## 8. Interesting Questions

### A. Quick-check (self-quiz)
1. What exactly does an LLM compute for its input — and what does it NOT do?
2. Why can a neural network fit a curved relationship but linear regression can't?
3. Why is matrix multiplication "embarrassingly parallel"?
4. Name the two limitations of RNN/LSTM that transformers fixed.
5. What is byte-level BPE's big practical guarantee?
6. Why does the same prompt give different answers? What single knob controls this?
7. Where does a model's training knowledge live? Where does YOUR document's knowledge live during RAG?
8. What does each training phase produce (name the 3 artifacts)?
9. Why might a base model respond to your question with more questions?
10. State Chinchilla's correction in one sentence.

### B. Interview-grade
11. Explain attention to a 10-year-old using one sentence of a story.
12. Base model vs instruct model vs aligned model — give a behavior example of each.
13. If temperature = 0, is output guaranteed 100% identical across runs? Discuss.
14. Why did deep learning take off when it did — what did GPUs have to do with it?
15. You must build a customer-support bot: walk me through which of the 3 phases you'd skip and why.
16. What's the difference between RLHF and DeepSeek's GRPO use of RL?
17. Why can't a model "memorize" today's news at inference time?

### C. Thought-provoking / research-flavored
18. If knowledge is compressed into weights, why do models hallucinate confidently?
19. Chinchilla says smaller+more data wins — why do frontier labs keep releasing ever-bigger models?
20. Next-token prediction produced chat and code "as a side effect." What might it still NOT produce — and what training change would you make?
21. $100M to train vs $0.001 to prompt: what happens to competition & innovation when the barrier is capital, not ideas?
22. If two models train on identical data, why do they behave differently?
23. Design challenge: you have $5,000 and must build a PDF Q&A product — where exactly do you spend it (model? RAG? labeling?) and why?

### Answers (cover while self-testing)
1. A probability distribution over the next token; it does not "know" or store text.
2. NN chains non-linear activations → can bend to any shape (universal function approximator).
3. Every output cell is independent — computable simultaneously.
4. Sequential (no parallelism) + forgetful (long-range fades).
5. Nothing is out of vocabulary — falls back to raw bytes (emoji/code/any language).
6. Stochastic sampling from the same distribution; temperature controls flattening.
7. Model → weights; your docs in RAG → vector DB.
8. base model → instruct model → aligned model.
9. Its pattern is "text continues with more questions" (pre-SFT behavior).
10. For a compute budget, train on more data relative to parameters — small+data beats big+undertrained.
11. e.g. "It" looks back at every earlier word and weights which one it refers to.
12. base: continues text / asks questions back; instruct: answers in chat format; aligned: prefers safe, helpful, ranked-better answers.
13. Practically near-identical, but not guaranteed (floating-point nondeterminism, batching, hardware).
14. NN training = parallel matrix math; GPUs (built for graphics) already had thousands of cores — repurposed before LLMs existed.
15. None — all 3 phases needed: pretraining gives knowledge, SFT gives assistant format, RLHF gives preference/safety polish.
16. RLHF: human prefs → reward model → PPO for alignment; GRPO: large-scale RL to elicit reasoning (DeepSeek-R1) — related, distinct goals.
17. Knowledge is frozen at training cutoff; inference only computes next tokens.
18. Weights store *statistics/patterns*, not verifiable facts; sampling + fluency can synthesize plausible-but-wrong text.
19. Data is now the bottleneck (high-quality tokens scarce), inference-time compute & MoE change the economics, and benchmark/capability competition rewards scale.
20. Speculative; candidates: persistent memory, tool use, grounding — fixes = retrieval, agentic loops, RL with tools (open research).
21. Consolidation risk, moats for incumbents, but open-weight models + cheap APIs partially democratize — good discussion material.
22. Different init, data order, hyperparameters, alignment data → different weights/behavior.
23. Spend mostly on extraction quality (chunking, embeddings, reranker) and eval; cheap model for answering — data/pipeline beats big model for narrow tasks.

---

## 9. Final Cheat Sheet (condensed)

- **LLM** = next-token predictor in a loop; capability = side effect of scale.
- **AI ⊃ ML ⊃ DL ⊃ LLMs**; ML learns rules from examples.
- **NN** = function approximator; training = minimize loss by tuning weights.
- **GPU** wins: NNs = parallel matrix multiplications.
- **Transformer (2017)**: attention = all tokens weigh all tokens, parallel; fixed RNN sequential + forgetful.
- **Tokenization**: sub-word IDs, BPE merges frequent pairs, byte-level = no OOV.
- **Generation**: softmax → sample → repeat; temperature = randomness (0 = greedy).
- **Knowledge** in weights; **vector DB** = RAG layer (separate!).
- **3 phases**: pre-train → base; SFT → instruct; RLHF (reward model + PPO) → aligned.
- **Scaling laws**: smooth power laws; Chinchilla = balance params with data, don't just grow.
- **$100M** training → AI engineering = build on top (prompt, RAG, adapters, orchestration).
