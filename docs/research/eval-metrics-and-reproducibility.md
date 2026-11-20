# Evaluation metrics and reproducibility for a GA-tuned RAG pipeline

Compiled 2026-11-20. Every claim is linked to the page or source file that owns it; all URLs accessed 2026-11-20.
Scope: Qwen3-4B-Instruct-2507 Q4_K_M behind Ollama (4096 ctx), sentence-transformers + FAISS, Python/Docker/PostgreSQL/Git
docs corpus, three laptops, LLM frozen, genetic algorithm (GA) choosing pipeline parameters.

Evidence labels used below:

| Tag | Meaning |
|---|---|
| **D** | Documented by the owning project (their doc page says it) |
| **S** | Read from the owning source code |
| **M** | Mechanism/derivation by me (arithmetic, floating-point behaviour); not an upstream guarantee |
| **O** | Open question — no primary source settles it |

---

## Part A — Retrieval metrics, defined precisely

### A.0 The running example

Five chunks, human relevance grades `0..3` (0 = useless, 3 = contains the answer verbatim):

| chunk | d1 | d2 | d3 | d4 | d5 |
|---|---|---|---|---|---|
| grade | 3 | 0 | 1 | 2 | 0 |

So the relevant set is `{d1, d3, d4}`, `|relevant| = 3`, and the best possible ordering is `d1, d4, d3`.
The system returns, in order:

```
R = [d3, d2, d1, d5, d4]      (rank 1 .. 5)
```

All numbers below are computed from this single list, so you can check every formula by hand.

### A.1 recall@k

```
recall@k = |relevant ∩ top_k(R)| / |relevant|
```

* **Question:** of all chunks that should have been found, what fraction is in the top k?
* **On the example:** `recall@1 = 1/3 = 0.333`, `recall@3 = 2/3 = 0.667`, `recall@5 = 3/3 = 1.0`.
* **Rewards:** coverage. Nothing else.
* **How you cheat it:** raise k. `recall@|corpus| = 1.0` for every system, including a random one. Recall is monotone
  non-decreasing in k by construction, so two systems evaluated at different k are not comparable.
* **When it is the wrong choice:** whenever rank matters. Recall@5 cannot tell `d1` at rank 1 from `d1` at rank 5, and
  it cannot tell a system that puts the answer first from one that buries it (the example scores 1.0 despite rank 3
  being the best hit). trec_eval's own help text warns that recall "is a fine single topic measure, but does not
  average well" across topics with different numbers of relevant documents
  ([`m_recall.c`](https://github.com/usnistgov/trec_eval/blob/main/m_recall.c)).
* **Note:** "does not average well" means the per-topic value has a different null distribution depending on
  `|relevant|`, so the mean over topics is not a mean of identically distributed quantities (M).

### A.2 precision@k

```
precision@k = |relevant ∩ top_k(R)| / k
```

* **Question:** of the k chunks I am about to paste into the prompt, how many are relevant?
* **On the example:** `P@1 = 1.0`, `P@3 = 2/3 = 0.667`, `P@5 = 3/5 = 0.6`.
* **Rewards:** a short, clean context window. This is the metric that actually tracks prompt pollution.
* **How you cheat it:** return fewer than k documents, or pick a tiny k. Also note the hard ceiling: with `|relevant| = 3`,
  `P@10 ≤ 0.3` no matter how good the system is. trec_eval documents this asymmetry: "P20 has very different expected
  characteristics if there are 300 total relevant docs for a topic as opposed to 10"
  ([`m_P.c`](https://github.com/usnistgov/trec_eval/blob/main/m_P.c)).
* **When it is the wrong choice:** when k exceeds `|relevant|` (the ceiling dominates), or when your relevance labels are
  incomplete — unjudged documents are scored as non-relevant, which is a measurement artefact, not a system failure.
  trec_eval treats unjudged docs as non-relevant by default and has a separate `-1` convention for pooled-but-unjudged
  documents ([README](https://github.com/usnistgov/trec_eval)).

### A.3 MRR (mean reciprocal rank)

```
RR(q) = 1 / rank(first relevant chunk)        (0 if none retrieved)
MRR   = (1/N) * Σ_q RR(q)
```

* **Question:** how far down do I have to scroll before the first useful chunk?
* **On the example:** first relevant is `d3` at rank 1, so `RR = 1/1 = 1.0`. If the order had been
  `[d2, d5, d3, ...]`, `RR = 1/3 = 0.333`.
* **Rewards:** one good hit, early. It is the natural metric for "navigate to the right page" tasks.
* **How you cheat it:** you cannot, much — but you also cannot improve it beyond the first hit. A system that finds one
  relevant chunk and nothing else scores the same as one that finds all of them.
* **When it is the wrong choice:** whenever a question needs more than one chunk (multi-hop, "compare X and Y", any
  answer that must be assembled). trec_eval: "Measure is most useful for tasks in which there is only one relevant
  doc" ([`m_recip_rank.c`](https://github.com/usnistgov/trec_eval/blob/main/m_recip_rank.c)). For this project, a
  Qwen3-4B answer over a 4096-token window usually needs 3–8 chunks, so MRR is a diagnostic, not a target.

### A.4 MAP (mean average precision)

```
AP(q)  = (1/|relevant|) * Σ_{k : R_k is relevant} precision@k
MAP    = (1/N) * Σ_q AP(q)
```

* **Question:** averaged over every relevant chunk, how precise was the ranking at the moment that chunk appeared?
* **On the example:** relevant hits at ranks 1, 3, 5 →
  `AP = (1/1 + 2/3 + 3/5) / 3 = (1 + 0.6667 + 0.6)/3 = 0.7556`.
* **Rewards:** putting *all* relevant chunks high, not just one. It is the only classic metric that penalises "one great
  hit, rest garbage".
* **How you cheat it:** with a single relevant chunk, MAP collapses to MRR (`AP = 1/rank`). With very few relevant
  chunks, one lucky hit dominates. AP also has no notion of graded relevance: a grade-3 chunk and a grade-1 chunk count
  identically.
* **When it is the wrong choice:** when relevance is graded and the difference between "mentions the topic" and "contains
  the answer" matters; or when `|relevant|` varies wildly across questions (AP's variance is then dominated by the
  questions with many relevant chunks). trec_eval calls MAP "the main single-valued number used to compare the entire
  rankings of two or more retrieval methods" ([`m_map.c`](https://github.com/usnistgov/trec_eval/blob/main/m_map.c)) —
  that reputation was earned on TREC ad-hoc collections with deep pools, not on a 150-question school set.

### A.5 nDCG@k

```
DCG@k  = Σ_{i=1..k} gain(r_i) / log2(i + 1)
IDCG@k = DCG@k of the ideal ordering (grades sorted descending)
nDCG@k = DCG@k / IDCG@k            ∈ [0, 1]
```

Two gain conventions, and you must state which one you used:

| Convention | gain(rel) | Where |
|---|---|---|
| linear (trec_eval default) | `gain = rel` | "Gain values are set to the appropriate relevance level by default", [`m_ndcg.c`](https://github.com/usnistgov/trec_eval/blob/main/m_ndcg.c) |
| exponential (common in web search) | `gain = 2^rel − 1` | the `2^rel−1` variant popularised by Järvelin & Kekäläinen follow-ups; **not** what trec_eval does by default |

trec_eval's exact discount, from the implementation (0-based `i`, so rank `r = i+1`):

```c
/* Note: i+2 since doc i has rank i+1 */
results_dcg += results_gain / log2((double) (i + 2));
```

([`m_ndcg.c`](https://github.com/usnistgov/trec_eval/blob/main/m_ndcg.c)). scikit-learn uses the same discount with a
configurable base and averages per sample ([`dcg_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.dcg_score.html),
[`ndcg_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)).

**Worked example, graded gain = rel:**

```
DCG@5  = 1/log2(2) + 0 + 3/log2(4) + 0 + 2/log2(6)
       = 1.0 + 0 + 1.5 + 0 + 0.7737 = 3.2737
IDCG@5 = 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.2619 + 0.5 = 4.7619
nDCG@5 = 3.2737 / 4.7619 = 0.6875
nDCG@3 = 2.5 / 4.7619 = 0.525      (DCG@3 = 1 + 0 + 1.5)
```

**Same list, binary gain (rel > 0 → 1):**

```
DCG@5  = 1 + 0 + 1/2 + 0 + 1/2.585 = 1.8869
IDCG@5 = 1 + 0.6309 + 0.5          = 2.1309
nDCG@5 = 0.8855
```

That gap (0.6875 graded vs 0.8855 binary) is the whole point: binary says "you found all three"; graded says "you found
them, but you put the only grade-3 chunk at rank 3, behind a grade-1 chunk". If your labels cannot be trusted at that
resolution, use binary and say so.

* **Question:** how good is the ordering, with early positions worth more and (optionally) better chunks worth more?
* **Rewards:** rank quality, gracefully. It is the only metric here that handles graded relevance natively.
* **How you cheat it:** choose a k that flatters you (nDCG@3 vs nDCG@10 can rank systems differently); inflate grades;
  or, with binary gains and exactly one relevant chunk, note that `nDCG@k = 1/log2(rank+1)` — you cannot improve it by
  retrieving *more* documents, only by moving the right one up. That last property is why nDCG is hard to game with k.
* **When it is the wrong choice:** when labels are binary and incomplete (nDCG then behaves like a soft MRR), when k is
  so large that the log discount is nearly flat, or when you need to compare against published TREC numbers computed
  with a different gain convention.

**Ties.** If two chunks get identical scores (common with FAISS quantised indexes), the ranking is arbitrary and nDCG
becomes implementation-dependent. scikit-learn averages over tied orderings by default and warns that `ignore_ties=True`
gives wrong results when ties exist ([`ndcg_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)).
FAISS says the same from the other side: ties make "the ordering of results … arbitrary"
([FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ)).

### A.6 Summary table

| Metric | Rewards | Cheat | Wrong when |
|---|---|---|---|
| recall@k | coverage | increase k | rank matters; k varies between systems |
| precision@k | clean context | small k; capped by `|rel|` | k > `|rel|`; labels incomplete |
| MRR | first hit | — (saturates) | multi-chunk answers |
| MAP | all relevant chunks, high | one lucky hit when `|rel|=1` | graded relevance; `|rel|` varies a lot |
| nDCG@k | ordering + grades | k choice; grade inflation | binary/incomplete labels; ties; cross-paper comparison |

---

### A.7 Retrieval-level vs answer-level evaluation

These measure different objects and must not be averaged into one number:

* **Retrieval-level** asks: did the right chunks reach the prompt? Input = question + chunk IDs; output ∈ [0,1].
  Deterministic given the index; cheap (milliseconds); can be computed for thousands of genomes.
* **Answer-level** asks: is the generated text correct/faithful/relevant? Input = question + chunks + generated answer
  (+ reference); output requires either string matching or another model.

**Extractive F1 / exact match are weak proxies.** SQuAD defines them precisely: exact match is "the percentage of
predictions that match any one of the ground truth answers exactly"; F1 is "the average overlap between the prediction
and ground truth answer … we treat the prediction and ground truth as bags of tokens"; both "ignore punctuations and
articles" ([SQuAD §6.1](https://arxiv.org/pdf/1606.05250)). Three consequences for a RAG report:

1. They score *surface form*, not meaning. "Use `git rebase -i`" and "run an interactive rebase" score 0.
2. They assume a short extractive span. A 4B chat model writes prose; EM is then near 0 for reasons unrelated to quality.
3. F1 over bag-of-tokens rewards quoting the question back. It is trivially inflated by verbosity.

Risch et al. measured this directly: "this comparison is mostly lexical-based and therefore misses out on answers that
have no lexical overlap but are still semantically similar, thus treating correct answers as false", and found
transformer-based semantic similarity "correlate[s] much better with human judgment than traditional lexical similarity
metrics" ([SAS, MRQA@EMNLP 2021](https://arxiv.org/abs/2108.06130)). Use EM/F1 only if your answer key is a literal span
and you say so.

**What LLM-as-judge measures, and why a small judge is unreliable.** An LLM judge scores a rubric ("is claim X supported
by context Y?") by prompting a model. It measures *the judge's opinion*, which correlates with human labels only to the
extent the judge was validated on your task. Primary evidence:

* Strong judges can match humans: GPT-4-class judges reach "over 80% agreement" with human preferences, "the same level
  of agreement between humans" — but the same paper documents position bias, verbosity bias, self-enhancement bias and
  limited reasoning ([MT-Bench / Chatbot Arena, NeurIPS 2023](https://arxiv.org/abs/2306.05685)).
* On questions with an objectively correct answer, judges are much worse: "many strong models (e.g., GPT-4o) perform
  just slightly better than random guessing" ([JudgeBench, ICLR 2025](https://arxiv.org/abs/2410.12784)).
* The RAG-specific systems that do work do **not** ship an off-the-shelf small judge. ARES "finetunes lightweight LM
  judges" on synthetic data and then corrects their errors with "prediction-powered inference (PPI)" using "a small set
  of human-annotated datapoints", needing "a few hundred human annotations during evaluation"
  ([ARES, NAACL 2024](https://arxiv.org/abs/2311.09476)).

For your setup the judge would be Qwen3-4B itself (self-enhancement bias, same tokenizer, same blind spots), running on
the same 4 GB GPU you are trying to keep free. **No study establishes that a 4B judge agrees with human raters on
Docker/Git faithfulness at the accuracy you need** (O). Treat a small-judge score as a noisy ordinal signal, never as
ground truth, and never as the GA's fitness.

---

### A.8 What RAG frameworks actually compute

RAGAS (reference-free RAG assessment) is the one you will meet first. Its current docs define:

| RAGAS metric | Formula (as documented) | Judge model needed? |
|---|---|---|
| **Context Precision** | `Σ_k (precision@k × v_k) / (number of relevant items in top K)`, `v_k ∈ {0,1}` — i.e. average precision over retrieved chunks | **Yes** for the default `ContextPrecision(llm=...)`, which labels each chunk against the reference answer. Variants `NonLLMContextPrecisionWithReference` and `IDBasedContextPrecision` need no model. ([docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)) |
| **Context Recall** | `claims in reference supported by retrieved context / total claims in reference` | **Yes**: the reference is decomposed into claims and each is attributed to the context by an LLM. `NonLLMContextRecall` / `IDBasedContextRecall` are string/ID variants. ([docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)) |
| **Faithfulness** | `claims in response supported by retrieved context / total claims in response` | **Yes** by default; the docs also offer `FaithfulnesswithHHEM`, a T5 classifier (HHEM-2.1-Open) for the support check, which removes the LLM. ([docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)) |
| **Answer / Response Relevancy** | mean cosine similarity between the embedding of the question and N questions generated *from the answer* | **Yes** to generate the N questions (default 3), **plus** an embedding model. ([docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/)) |

Three things to notice, because they decide whether RAGAS is usable here:

1. **"Context recall" is not IR recall.** It is claim coverage against a reference *answer*. If your gold label is a
   chunk ID, RAGAS context recall answers a different question and will not detect a chunking regression that moves the
   gold chunk from rank 1 to rank 4.
2. **"Context precision" is AP, not P@k.** It rewards early placement, which is good, but it is computed over whatever
   chunks you retrieved, so it is sensitive to how many chunks you retrieve.
3. **The RAGAS paper's selling point is "without having to rely on ground truth human annotations"**
   ([RAGAS, 2023](https://arxiv.org/abs/2309.15217)) — which is exactly the property that makes it unsuitable as a
   *fitness function* for a GA: a judge-free-ish metric that depends on a judge you have not calibrated adds variance
   you cannot separate from genome quality.

**ARES** computes context relevance, answer faithfulness and answer relevance, but with fine-tuned lightweight judges
plus PPI calibration against human labels (above). **TruLens** names the same three axes — context relevance,
groundedness, answer relevance — the "RAG triad", and implements them as feedback functions
([TruLens RAG Triad](https://www.trulens.org/getting_started/core_concepts/rag_triad/)); its docs route model-based
judgements through an LLM provider, so the same caveat applies.

---

### A.9 Statistical validity: N = 150, and what a GA does to it

#### The interval on one number

Recall@5 is a proportion: 150 questions, each hit or miss. With 105 hits (p̂ = 0.70), the 95% Wilson interval is

```
den    = 1 + z²/n                  = 1 + 3.8416/150      = 1.02561
centre = (p̂ + z²/(2n)) / den       = (0.70 + 0.012805)/1.02561 = 0.69501
half   = z·√(p̂(1−p̂)/n + z²/(4n²)) / den
       = 1.96·√(0.0021 + 0.0000427)/1.02561
       = 1.96·0.046302/1.02561      = 0.08848
CI     = 0.69501 ± 0.08848         = [0.622, 0.768]
```

Half-width ≈ **7.3 percentage points**. Wald gives [0.627, 0.773] here — nearly the same at p = 0.7, but Wald collapses
near 0 or 1 and its coverage is erratic; Brown, Cai & DasGupta recommend "the Wilson interval or the equal-tailed
Jeffreys prior interval for small n" and show the Wald interval's coverage is "chaotic"
([Statist. Sci. 16(2), 2001](https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full)).
Use Wilson.

At other n, same p̂ = 0.70 (M, same formula):

| n | 95% Wilson CI | half-width |
|---|---|---|
| 30 (one 5-fold split) | [0.521, 0.833] | 15.6 pp |
| 50 | [0.563, 0.809] | 12.3 pp |
| 100 | [0.604, 0.781] | 8.8 pp |
| 150 | [0.622, 0.768] | 7.3 pp |

**Read this before writing any sentence containing "improved".** With 150 questions you cannot distinguish 0.70 from
0.77 at 95% confidence on a single run.

#### How many questions to resolve 3 points

Two-proportion sample size (normal approximation, α = 0.05 two-sided, power 0.80), per arm:

```
n = [ z_{α/2}·√(2·p̄(1−p̄)) + z_β·√(p1(1−p1) + p2(1−p2)) ]² / (p2 − p1)²
p1 = 0.70, p2 = 0.73, p̄ = 0.715
n = [1.9600·√(2·0.203775) + 0.8416·√(0.21 + 0.1971)]² / 0.0009
  = [1.9600·0.63840 + 0.8416·0.63805]² / 0.0009
  = [1.2513 + 0.5370]² / 0.0009
  = 3.1975 / 0.0009  ≈ 3,553 questions per arm
```

**3,553 per arm** for an unpaired comparison. Pairing helps a lot *if* the two systems disagree rarely. For a paired
design (same questions, McNemar), with discordant rate `π_d`:

```
n ≈ (z_{α/2} + z_β)² · π_d / Δ²
π_d = 0.10 → n ≈ 7.84 · 0.10 / 0.0009 ≈   871
π_d = 0.30 → n ≈ 7.84 · 0.30 / 0.0009 ≈ 2,613
```

So even a paired A/B on the *same* 150 questions resolves only

```
MDE(80% power, π_d = 0.10) = 2.80 · √(0.10/150) = 7.2 pp
MDE(95% CI, unpaired)      = 1.96 · √(2·0.21/150) = 10.4 pp
```

i.e. **7–15 points, not 3**. A 3-point GA "win" on 150 questions is inside the noise band.

#### What that does to a GA

A GA does not run one comparison; it runs hundreds and keeps the maximum. That is a different statistical object:
selection bias, not per-comparison error.

Simulation (M, 8 lines of Python, seed 7): 150 questions, **every genome has the same true recall@5 = 0.70**, so no
genome is better. Score K genomes on the same dev set, keep the best, then score it on a fresh 150-question test set
drawn from the same distribution:

| K genomes | mean best dev recall@5 | mean fresh-test recall@5 | optimism |
|---|---|---|---|
| 20 | 0.767 | 0.698 | +6.9 pp |
| 50 | 0.782 | 0.697 | +8.6 pp |
| 100 | 0.791 | 0.700 | +9.1 pp |
| 200 | 0.799 | 0.701 | +9.8 pp |
| 500 | 0.809 | 0.701 | +10.8 pp |
| 1000 | 0.816 | 0.703 | +11.3 pp |

Reproduce with:

```python
import random, statistics
random.seed(7)
def sim(K, N=150, p=0.7, trials=400):
    dev, test = [], []
    for _ in range(trials):
        obs = [sum(random.random() < p for _ in range(N))/N for _ in range(K)]
        dev.append(max(obs))
        test.append(sum(random.random() < p for _ in range(N))/N)
    return statistics.mean(dev), statistics.mean(test)
```

**K = 500 genomes with zero real differences produces a best dev score of 0.809 and a test score of 0.701.** If your GA
report says "recall@5 improved from 0.70 to 0.81", that is what a null result looks like. (Caveat, in your favour:
genomes scored on the same questions share errors, so real optimism is smaller than this independent-draw simulation;
but it is still positive, and the order of magnitude is right. M)

This is a known, named phenomenon, not a quirk of your code:

* Model selection by maximising an estimate overfits that estimate, and "the effects of this form of over-fitting are
  often of comparable magnitude to differences in performance between learning algorithms, and thus cannot be ignored
  in empirical evaluation" — Cawley & Talbot, [JMLR 11:2079–2107, 2010](https://www.jmlr.org/papers/v11/cawley10a.html).
* Reusing a holdout adaptively "can easily lead to overfitting to the holdout set itself" — Dwork et al.,
  [Generalization in Adaptive Data Analysis and Holdout Reuse, 2015](https://arxiv.org/abs/1506.02629).

#### Held-out test set and k-fold: what each actually buys

| Protocol | What it fixes | What it does not fix |
|---|---|---|
| Fixed dev/test split (e.g. 100/50), test touched once | Removes the winner's-curse *estimate*; gives an honest number for the final genome | Test n = 50 → ±12 pp; a 10 pp optimism still fits inside the CI |
| k-fold on the dev set (k = 5) | Lower-variance estimate of a **fixed** procedure; every question is tested once | Does not fix selection bias if the GA sees all folds over generations — that is just a slower dev set. k = 5 on 150 gives 30-question folds, ±16 pp each |
| k-fold **inside** each fitness evaluation, averaged | Reduces fitness noise per genome (SE ≈ √(p(1−p)/150) ≈ 3.7 pp if folds were independent) | Costs k× compute; folds are correlated, so the real gain is smaller; the GA still selects on the same 150 questions |
| Repeated subsampling (e.g. 20× 80/70) reported as mean ± sd | Quantifies how much the *ranking* depends on which questions you drew | Still dev; still selection-biased if used to choose |

Ten-fold stratified CV is the classical recommendation for model selection (Kohavi,
[IJCAI 1995](https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf)), but Kohavi's setting is "choose one classifier from
a handful". A GA evaluating hundreds of genomes is the adaptive-reuse regime, where the dominant error is selection, not
fold variance.

**Practical protocol for this project (M):**

1. Freeze 150 labelled questions. Split 100 dev / 50 test **before** writing the GA. Hash both files.
2. GA only ever sees dev. Never print test scores during a run.
3. Fitness = mean over dev questions (paired by question, same order every generation).
4. Report the final genome on test **once**, with a Wilson CI, and report dev−test gap.
5. If you must iterate after seeing test, say so and treat the second test number as dev.

---

### A.10 Recommendation: one scalar, plus guardrails

**Fitness (single scalar, no LLM calls, deterministic, machine-independent):**

```
fitness(genome) = mean_q nDCG@10(q; binary gain)
                − λ · mean_q (retrieved_tokens(q) / 1000)
λ = 0.01, k capped at 10, same 100 dev questions every evaluation
```

Why this one:

* **Rank-sensitive.** With one gold chunk, `nDCG@10 = 1/log2(rank+1)`, so it pays for moving the gold chunk from rank 4
  to rank 1 — recall@10 cannot see that move at all.
* **k-inflation-proof.** With binary gains and a fixed `IDCG@10 = 1` for a single gold chunk, retrieving more chunks
  cannot raise the score. Contrast recall@k, which is monotone in k and therefore has a trivial cheat.
* **Graded-ready.** If you later label 0/1/2, the same code works; just freeze the gain convention (rel vs `2^rel−1`)
  and never change it mid-run.
* **Deterministic and cheap.** No judge, no sampling, milliseconds per question, so 500 genomes × 30 generations is
  feasible on a laptop.
* **Token penalty, not latency penalty.** Wall-clock latency differs 5–10× across your three laptops (M) and would make
  the fitness landscape machine-specific. Retrieved tokens are a hardware-independent proxy for the real cost you care
  about: prompt pollution inside a 4096-token window.

**Guardrails (all four, or the GA will find the loophole):**

1. **Cap k at 10** in the genome schema — do not let the GA evolve `k` upward. If you want to tune k, tune it in a
   separate 1-D sweep and report the curve.
2. **Hard constraint on recall@5:** reject any genome whose dev recall@5 is more than 1 pp below the baseline's. nDCG@10
   alone will happily trade a rank-5 hit for a rank-1 hit on a different question; this keeps top-5 coverage from
   silently collapsing.
3. **Freeze the token penalty λ before the run** and report it. If λ is tuned per generation, the GA is optimising the
   penalty, not the pipeline.
4. **Cap the genome's `top_k` effect on the prompt:** `retrieved_tokens ≤ 2000` (half of the 4096 context, leaving room
   for the system prompt and answer). Hard reject, not a soft penalty, so the constraint cannot be bought off.

**Secondary metrics to report (never optimise):** recall@5, recall@10, MRR@10, MAP@10, nDCG@10 with graded gains,
precision@5, mean retrieved tokens, p50/p95 retrieval latency per machine, and on a 30-question stratified sample:
faithfulness via HHEM-2.1-Open (a classifier, not a judge) plus manual spot-check agreement.

**What not to do:** do not put RAGAS faithfulness, answer relevancy, or any LLM-judge score into `fitness`. It costs a
generation per genome, adds judge variance larger than your effect, and cannot be reproduced on the other two laptops.

---

## Part B — Reproducibility across heterogeneous machines

### B.1 Does `temperature=0` + fixed seed give identical output?

**Short answer: no, and on a single machine the seed is not even the mechanism.**

**(a) What the seed does.** Ollama's API exposes `seed` as an integer described as "Random seed used for reproducible
outputs", alongside `temperature`, `top_k`, `top_p`, `min_p`, `num_ctx`, `num_predict`
([Ollama API, `/api/generate`](https://docs.ollama.com/api/generate)). That is the entire documented guarantee: a seed
makes *sampling* repeatable, not the arithmetic that produced the logits.

**(b) What temperature 0 does in llama.cpp.** In `llama_sampler_temp_impl`, `temp <= 0` sets every logit except the
maximum to `-INFINITY`:

```c
if (temp <= 0.0f) {
    // find the token with the highest logit and set the rest to -inf
    ...
    return;
}
```

([`src/llama-sampler.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp)). The distribution
sampler then draws from a one-hot vector, so the draw is deterministic **regardless of seed**. The `--seed` help string
confirms the seed only configures the RNG: "RNG seed (default: -1, use random seed for -1)"
([`common/arg.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp)), and
`llama_sampler_init_dist` documents `seed == LLAMA_DEFAULT_SEED` (0xFFFFFFFF) as "use a random seed"
([`include/llama.h`](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h)).

**Implication:** at temperature 0, any output difference is *numerical*, not stochastic. Setting a seed and calling it
reproducible is a category error. (Ollama does not document its sampler chain; if you need a documented greedy path, run
`llama-server` directly or verify empirically on your build — O.)

**(c) GPU vs CPU, and GPU vs GPU.** PyTorch states the general rule bluntly: "Completely reproducible results are not
guaranteed across PyTorch releases, individual commits, or different platforms. Furthermore, results may not be
reproducible between CPU and GPU executions, even when using identical seeds"
([PyTorch reproducibility notes](https://docs.pytorch.org/docs/2.15/notes/randomness.html)). The mechanism is spelled out
on the same page for attention: "Bitwise matching numerics across different SDPA backends are **not guaranteed**, even
for the same inputs and dtype. Each backend performs floating-point accumulation in a different order, and because
floating-point addition is not associative, the results will differ between backends." The same argument applies to
ggml's CUDA/Metal/CPU kernels, which are different implementations of the same matmuls (M). llama.cpp's own perplexity
tool says its numbers "are **not** directly comparable to those of other projects because the exact values depend
strongly on the implementation details"
([`tools/perplexity/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)).

**(d) Quantisation.** Q4_K_M is not a rounding of FP16 outputs; it is a different function. llama.cpp's own measurements
for LLaMA-2-7B compare each quant against FP16 on Wikitext-2:

| quant | KLD vs f16 | RMS Δp | "same top p" (argmax agreement) |
|---|---|---|---|
| q4_K_M | 0.031273 ± 0.000238 | 5.519% | 94.665% |

([`tools/perplexity/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)). Read the
last column literally: **~5.3% of tokens get a different argmax than the FP16 model**, i.e. greedy decoding of a 4-bit
model diverges from FP16 within a few dozen tokens. That is fine — you froze the quant — but it means (i) another team's
Q4_K_M file is not your model unless the digest matches, and (ii) tiny numerical differences between backends are enough
to flip a near-tie token, after which the whole continuation changes (M).

**(e) Batching and `num_parallel`.** `-b/--batch-size` is the "logical maximum batch size", `-ub/--ubatch-size` the
"physical maximum batch size", and for the server `-np/--parallel` is the "number of server slots"
([`common/arg.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp)). Ollama's FAQ: "`OLLAMA_NUM_PARALLEL`
- The maximum number of parallel requests each model will process at the same time, default 1" and "Parallel request
processing for a given model results in increasing the context size by the number of parallel requests"
([Ollama FAQ](https://docs.ollama.com/faq)). Different batch composition ⇒ different matmul shapes ⇒ different
accumulation order ⇒ different logits at the margins (M; consistent with the PyTorch SDPA statement above). **No
upstream document guarantees that `--parallel 4` and `--parallel 1` produce identical text**, and none guarantees they
differ; treat it as an empirical question you test.

**Protocol that actually settles it (M):**

1. On each laptop, same 150 dev prompts, `temperature=0`, `seed=42`, `num_parallel=1`, `num_ctx=4096`, same model digest.
2. Run the whole set **3×**; byte-compare. Any difference on one machine = nondeterminism within that machine (expect
   none for greedy on a fixed build; if you see some, the backend is using a nondeterministic reduction).
3. Across machines, report **agreement rate**, not "identical": fraction of the 150 answers that are byte-identical, and
   the mean token-level divergence point. Expect agreement to fall with answer length (M).
4. If cross-machine agreement is < 100%, do not pool answer-level metrics across laptops. Evaluate retrieval metrics
   (machine-independent) everywhere, and answer-level metrics on one designated machine.

### B.2 Embedding and index determinism

**sentence-transformers.** `SentenceTransformer.encode` takes `batch_size=32`, `precision='float32'`, `device`,
`normalize_embeddings`; the constructor takes `revision` and `local_files_only`
([API reference](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html)). Consequences:

* **Pin the revision.** Pass `revision="<40-hex-commit>"`. Hugging Face requires the full hash, not a 7-char prefix:
  "When using the commit hash, it must be the full-length hash instead of a 7-character commit hash"
  ([HF Hub download guide](https://huggingface.co/docs/huggingface_hub/guides/download)). `main` moves; a re-download
  silently changes your embeddings.
* **CPU vs GPU.** Embedding is a forward pass through a transformer; PyTorch's statement above applies unchanged: no
  bitwise CPU/GPU guarantee even with identical seeds. Expect differences at ~1e-7 relative (M) — harmless for ranking
  in most cases, but enough to flip near-ties and enough to make a hash of the embedding matrix differ. **Never compare
  embedding hashes across devices; compare retrieved IDs.**
* **Batch size.** `encode` pads each batch to its longest sequence and runs batched matmuls; changing `batch_size`
  changes the reduction shapes. Not documented as affecting output, but it is the same non-associativity mechanism (M).
  Fix `batch_size` in the run record and keep it fixed.
* **`max_seq_length`.** Truncation changes the vector. Record it.

**FAISS.** The FAQ is explicit: "In Faiss, all random seeds are set to constant values, which normally means that two
runs of training produce bit-exact same results. This is not always possible for performance reasons"
([FAQ §Reproducibility](https://github.com/facebookresearch/faiss/wiki/FAQ)). The details that matter:

* **k-means seed.** `ClusteringParameters::seed = 1234` by default; "negative values lead to seeding an internal rng with
  `std::high_resolution_clock`" ([`faiss/Clustering.h`](https://github.com/facebookresearch/faiss/blob/main/faiss/Clustering.h)).
  Never use a negative seed.
* **Thread count.** "When computations are multi-threaded, this is true as well, since most computations can be balanced
  statically … several runs of the same operation with the same number of threads give the same result bitwise." Fix
  `OMP_NUM_THREADS` and/or call `faiss.omp_set_num_threads(n)`. Known exceptions: LAPACK eigenvalue extraction
  (`PCAMatrix`, `OPQMatrix`) is not exactly reproducible with MKL, and `HNSW.add` runs in unspecified order (search is
  deterministic) ([Threads and asynchronous calls](https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls)).
* **BLAS, not FAISS.** "Even when the number of threads is fixed and set to 1, MKL does not by default guarantee that
  other operations, include matrix multiplication, are bit-exact reproducible … the matrix multiplication is implemented
  differently depending on the processor it runs on." The escape hatch is `MKL_CBWR` (same page). On ARM/Metal you are
  not using MKL at all, so cross-architecture bitwise equality is off the table (M).
* **Ties.** If distances tie, "the ordering of results is arbitrary" ([FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ)).
  With `IndexFlatIP` over normalised vectors, ties are rare but not impossible for duplicate chunks.

**Recommendation for this corpus (M):** use `IndexFlatIP` (exact, no training, no k-means, no coarse quantiser). A docs
corpus of 10⁴–10⁵ chunks is small enough that exact search is milliseconds on CPU. That deletes the entire class of
training nondeterminism, and it makes your retrieval metric a function of (vectors, query) only. If you later need IVF,
fix `nlist`, `nprobe`, `seed=1234`, `OMP_NUM_THREADS`, and record all four.

### B.3 What a minimal, honest run record looks like

One JSON object per evaluated genome per run, one JSONL file per experiment, plus a `manifest.json` for the run.

**Manifest (once per run):**

| Field | Example | Why |
|---|---|---|
| `run_id` | `2026-11-20T14-03Z-ga-r3` | joins logs to report |
| `git_sha` | `a1b2c3d` (dirty: false) | code that produced the numbers |
| `python` / `packages` | `3.12.7`, `torch 2.14.0+cu121`, `faiss-cpu 1.9.0`, `sentence-transformers 5.x`, `ollama 0.40.16` | engine differences change outputs |
| `llm.name` / `llm.digest` | `qwen3:4b-instruct-2507-q4_K_M`, `sha256:…` | Ollama exposes the digest via `/api/tags` |
| `llm.params` | `temperature 0, seed 42, top_k 20, top_p 1.0, min_p 0, repeat_penalty 1.0, num_ctx 4096, num_predict 512` | the full sampler chain, not just temp/seed |
| `llm.server` | `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_KV_CACHE_TYPE=f16`, flash-attn on/off | batching + KV quant change logits |
| `embed.model` / `revision` | `BAAI/bge-small-en-v1.5`, `5c38ec7c…` (40 hex) | `main` moves |
| `embed.params` | `normalize=true, max_seq_length 512, batch_size 32, device cpu` | truncation + batching |
| `index` | `FlatIP, d=384, ntotal=41233, metric=IP` | plus `nlist/nprobe/seed/OMP_NUM_THREADS` if IVF |
| `corpus` | `sha256` of the concatenated, sorted chunk texts; chunker params | the actual retrieval target |
| `questions` | `sha256` of the 100 dev / 50 test JSONL | prevents silent edits |
| `hardware` | GPU model, driver, CUDA/Metal version, CPU, RAM | explanation for latency + divergence |
| `metrics` | `nDCG@10`, `recall@5`, `recall@10`, `MRR@10`, `MAP@10`, `tokens`, `latency_p50/p95` | |
| `stats` | `N=100`, `ci=wilson95`, `bootstrap=1000`, `repeats=3` | so the CI is auditable |

**Per-genome record (one line):**

```json
{"genome_id":"g0417","gen":12,"params":{"chunk_size":512,"overlap":64,"k":6,"embed":"bge-small","normalize":true,"rerank":false},
 "dev":{"n":100,"ndcg10":0.7142,"recall5":0.81,"recall10":0.93,"mrr10":0.772,"map10":0.701,"tokens_mean":1480},
 "ci":{"ndcg10":[0.652,0.771],"method":"wilson"},
 "latency_ms":{"p50":41,"p95":88},"device":"rtx3050ti","repeats":3,"outputs_sha256":"…"}
```

**Reporting rules (M):**

* Report **mean + 95% CI over questions** (Wilson for a single proportion; 1000-resample paired bootstrap for A/B
  differences, resampling *questions*, not generations).
* State N, the split, and the number of genomes evaluated. A score without K is uninterpretable once selection is
  involved.
* **3 repeats** are for detecting nondeterminism, not for estimating uncertainty. Repeating a deterministic pipeline 3×
  gives zero variance and false confidence; the uncertainty lives in *which questions you drew*.
* Report the dev−test gap for the winning genome. If it is ≥ 5 pp, say the GA overfit.
* Round to 0.01 and never report 4 decimals on N = 100.

### B.4 What to standardise vs what is genuinely machine-specific

| Standardise (must be identical or numbers are not comparable) | Machine-specific (record, do not compare) |
|---|---|
| Question sets + labels (hash both) | Wall-clock latency, tokens/sec, load time |
| Chunker implementation + params | GPU backend (CUDA vs Metal vs CPU), driver version |
| Embedding model + **revision** + `max_seq_length` + normalisation | `batch_size`, `OMP_NUM_THREADS`, `num_parallel` actually used |
| Index type + params (`FlatIP`; or `nlist/nprobe/seed`) | VRAM/RAM, whether the model spills to CPU |
| Prompt template (hash it), system prompt | Quant file *if* you rebuild it locally (don't — see below) |
| Decoding params: `temperature=0`, `top_k`, `top_p`, `min_p`, `repeat_penalty`, `num_ctx` | Ollama/llama.cpp version (record, but keep in a narrow band) |
| Metric implementation (one module, one test suite, one rounding rule) | Absolute answer text at the token level |
| Seed policy (all seeds fixed; document defaults) | Which laptop produced the answer-level sample |
| Output JSONL schema | |

**The one thing people get wrong:** they try to make *everything* identical, fail on latency and logits, then declare
the whole comparison unreproducible. Split the claim in two: **retrieval metrics must be identical across machines**
(they are, given the same vectors and index — and if they are not, you have a bug, not "hardware variance"), while
**answer-level text is allowed to differ** and must be reported as an agreement rate, not as equality.

**Ship the model file, not a rebuild.** Ollama's `/api/tags` returns a SHA256 digest for each model; pin
`model[:tag]@sha256:…`. Two people running `ollama pull qwen3:4b-instruct-2507-q4_K_M` on different days can get
different blobs (M — tags are mutable); the digest is the identity.

---

## Part C — Reading list, in order

1. **Manning, Raghavan & Schütze, *Introduction to Information Retrieval*, ch. 8 "Evaluation in information retrieval"** —
   <https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html> (PDF:
   <https://nlp.stanford.edu/IR-book/pdf/08eval.pdf>). Free, and the only source here that derives recall/precision/MAP
   from worked examples. Read first; everything else assumes it. *Note: the Stanford host was unreachable from this
   network on 2026-11-20; the chapter is also in the print book (CUP 2008), §8.4–8.5.*
2. **scikit-learn `ndcg_score` / `dcg_score` API pages** —
   <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html>. Read second because you will
   implement it: exact discount, `k`, tie handling, and a runnable example you can compare against your own function.
3. **NIST `trec_eval` (README + `m_map.c`, `m_recall.c`, `m_ndcg.c`)** — <https://github.com/usnistgov/trec_eval>.
   Read third to see what the field actually runs, including the blunt warnings about precision/recall not averaging
   well and MAP being the "main single-valued number".
4. **BEIR paper + repo** — <https://arxiv.org/abs/2104.08663>, <https://github.com/UKPLab/beir>. Read fourth for the
   evaluation *protocol* (nDCG@10 as the headline, Recall@100 as a ceiling check, BM25 as the mandatory baseline) and
   for why a single-corpus number does not generalise.
5. **Brown, Cai & DasGupta, "Interval Estimation for a Binomial Proportion", *Statistical Science* 16(2), 2001** —
   <https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full>.
   Read fifth, before you write a single "improved by 3%" sentence. Explains why Wald fails and Wilson is the default.
6. **Cawley & Talbot, "On Over-fitting in Model Selection…", JMLR 11, 2010** —
   <https://www.jmlr.org/papers/v11/cawley10a.html>. Read sixth: the quantitative argument for why selecting the max over
   many noisy estimates produces gains "of comparable magnitude to differences between algorithms". This is your GA's
   failure mode, stated 15 years early.
7. **Dwork et al., "Generalization in Adaptive Data Analysis and Holdout Reuse", 2015** —
   <https://arxiv.org/abs/1506.02629>. Read seventh for the formal version of "if you look at the test set twice, it is
   now a dev set", and for what a reusable holdout would cost.
8. **RAGAS docs (metrics list + context precision/recall/faithfulness/answer relevancy pages)** —
   <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/>. Read last, and critically: it is the map of
   what the RAG-eval ecosystem computes and, page by page, which metrics silently require a judge model.

---

## Recommendation

**Metric suite**

| Layer | Metric | Computed by | Role |
|---|---|---|---|
| Primary (fitness) | `nDCG@10`, binary gain, k = 10, minus `0.01 ×` retrieved tokens/1000 | your code, ~40 lines | the only thing the GA optimises |
| Guard | `recall@5` ≥ baseline − 1 pp, `retrieved_tokens ≤ 2000` | your code | hard reject, prevents rank/coverage trade |
| Reported | `recall@5`, `recall@10`, `MRR@10`, `MAP@10`, `nDCG@10` graded, `precision@5` | your code | diagnostic table in the report |
| Answer-level | `faithfulness` via HHEM-2.1-Open on 30 stratified questions + manual spot check | classifier + you | sanity check only, never fitness |
| Cost | p50/p95 retrieval latency, mean context tokens, index size | your code | per-machine, never compared across machines |

**Fitness scalar**

```
fitness(g) = mean_{q ∈ dev} nDCG@10(q, binary gains, k=10)
             − 0.01 · mean_{q ∈ dev} retrieved_tokens(q)/1000
subject to: recall@5(g) ≥ recall@5(baseline) − 0.01
            retrieved_tokens(q) ≤ 2000 for all q
```

**Protocol:** 100 dev / 50 test, split and hashed before coding; GA sees dev only; test evaluated once on the winner;
report mean + Wilson 95% CI, N, K genomes, and dev−test gap; if the gap ≥ 5 pp, report the GA as overfit rather than
reporting the dev number.

---

## Confidence / gaps

**High confidence (directly verified in owning docs/source):**

* Formulas and caveats for recall/precision/MRR/MAP/nDCG: trec_eval source + scikit-learn API docs, with exact discount
  read from `m_ndcg.c`.
* RAGAS metric definitions and which variants require an LLM: current RAGAS docs, per metric page.
* `temp ≤ 0` ⇒ argmax and seed-only-affects-sampling: llama.cpp `llama-sampler.cpp`, `arg.cpp`, `llama.h`.
* `OLLAMA_NUM_PARALLEL` default 1 and context scaling: Ollama FAQ.
* FAISS seed default 1234, negative-seed behaviour, thread-count reproducibility and the MKL exception: `Clustering.h` +
  FAISS wiki (FAQ, Threads).
* HF full-hash revision requirement: HF Hub download guide.
* No CPU/GPU bitwise reproducibility, and the non-associativity reason: PyTorch reproducibility notes.
* Wilson-over-Wald: Brown, Cai & DasGupta abstract.
* Selection-overfitting magnitude: Cawley & Talbot abstract; holdout reuse: Dwork et al. abstract.
* Judge limits: MT-Bench (>80% agreement, listed biases), JudgeBench (strong models near random on hard pairs), ARES
  (fine-tuned judges + PPI + hundreds of human labels).

**Medium confidence (arithmetic/mechanism, not an upstream guarantee):**

* All CI and sample-size numbers are my arithmetic from the stated formulas; they are reproducible but not copied from a
  citable table. The GA optimism simulation is mine (8 lines, seed 7) and assumes independent draws across genomes; real
  genomes share questions, so true optimism is somewhat smaller.
* The claim that batch/`num_parallel` changes logits is a mechanism argument (different matmul shapes ⇒ different
  accumulation order), consistent with PyTorch's SDPA note but not stated by llama.cpp or Ollama for decoding.
* "Ollama tags are mutable" — true of registry tags generally, but I did not find an Ollama doc saying so explicitly.
  Pinning by digest is the safe move regardless.

**Gaps / open questions:**

1. **No primary source on small-judge reliability for RAG faithfulness.** ARES fine-tunes judges rather than using a 4B
   off-the-shelf model, which is suggestive but not a measurement of Qwen3-4B-as-judge. If you use a judge, calibrate it
   yourself on ≥ 50 human-labelled items and report Cohen's κ; there is no citation that will do this for you.
2. **No verified cross-backend (CUDA vs Metal) measurement for llama.cpp.** The PyTorch and FAISS statements are about
   their own kernels; the llama.cpp equivalent is inference, not documentation. Your 3× repeat + cross-machine
   agreement test is the only evidence you will have.
3. **Qwen3-4B-Instruct-2507 Q4_K_M quant-quality numbers are not published by the Ollama model card** as far as I could
   verify; the table above is LLaMA-2-7B from llama.cpp's perplexity page, used only to show the *magnitude* of quant
   drift, not as a claim about your model.
4. **The IR book chapter could not be fetched** (host unreachable). Definitions in Part A were cross-checked against
   trec_eval and scikit-learn instead; if the reading list link matters for the report, verify it from another network.
5. **Optimal λ for the token penalty is a judgement call**, not a derived value. 0.01 means "1000 extra context tokens
   costs 1 pp of nDCG". Sanity-check it once against your baseline's token count, then freeze it.
