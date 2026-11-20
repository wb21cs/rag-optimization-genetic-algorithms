# Free hosted LLM APIs for one-off question generation

Research for ticket #4. Compiled 2026-11-20. The API is used **once, offline**, to turn ~1000
documentation passages into questions; the running RAG system stays local.

## Summary

| Provider | Free allowance (documented) | Per-request cost | 1000 questions | Verdict |
|---|---|---|---|---|
| **Groq** | `openai/gpt-oss-120b` and `qwen/qwen3.8-27b`: 30 RPM, **1,000 RPD**, 8K TPM, 200K TPD | ~1k input + ~80 output tokens | one day, ~1.5–3 h at 8K TPM | **First choice**: the RPD ceiling exactly matches the job |
| **Gemini API** | Free tier exists; per-model RPM/TPM/RPD are shown only in AI Studio and "not guaranteed" | ~1k input + ~80 output | same day if quota allows | Good backup; record the numbers AI Studio shows you |
| **OpenRouter** | `:free` models: 20 RPM, **50 RPD** with <10 credits, **1,000 RPD** after ≥10 credits purchased | varies | 20 days at 50 RPD | Last resort unless a credit purchase is acceptable |
| **Local Qwen3-4B** | no quota | prefill-bound | ~2–4 h on the 4 GB laptop (estimate) | Fallback with zero dependencies |

## Groq — free plan, measured from the published table

Source: <https://console.groq.com/docs/rate-limits> (accessed 2026-11-20). The page labels the table
"Free Plan Limits" and lists, per model: RPM, RPD, TPM, TPD.

| Model | RPM | RPD | TPM | TPD |
|---|---|---|---|---|
| `openai/gpt-oss-120b` | 30 | 1,000 | 8K | 200K |
| `qwen/qwen3.8-27b` | 30 | 1,000 | 8K | 200K |
| `openai/gpt-oss-20b` | 30 | 1,000 | 8K | 200K |

Rate limits are **per organisation**, not per key; extra keys do not add quota (same page). 429s
carry `retry-after` and `x-ratelimit-*` headers, so back-off is mechanical.

Practical arithmetic: one passage (~800–1,200 tokens with instructions) plus an ~80-token question is
~1.1k tokens. 8,000 TPM ≈ **7 requests/minute** → 1,000 questions in **≈ 2.4 hours**, comfortably
inside the 1,000 RPD and 200K TPD ceilings. If you batch 5 passages per request (~5k tokens) you hit
TPM in ~1.6 requests/min but finish in fewer calls; single-passage requests are simpler to filter.

**Unverified:** whether Groq's terms permit generating a dataset from third-party documentation. Their
"Your Data" page (<https://console.groq.com/docs/your-data>) is the place to check before committing;
this document does not assert permission.

## Gemini API — free tier exists, numbers are account-specific

Source: <https://ai.google.dev/gemini-api/docs/rate-limits> (accessed 2026-11-20). The page states
that limits are applied per project, measured in RPM / TPM / RPD, and that "specified rate limits are
not guaranteed and actual capacity may vary"; the concrete table is behind AI Studio
(<https://aistudio.google.com/rate-limit>). The only free-tier row the public page still prints is for
an experimental model: 200 RPM, 50,000 TPM, 10,000 RPD.

Implication: do not plan against a remembered Gemini number. Open AI Studio, screenshot the free-tier
row for the model you intend to use, and record it in the run manifest. Free-tier data-use terms also
differ from paid — check <https://ai.google.dev/gemini-api/terms>.

## OpenRouter — the 50/day wall

Source: <https://openrouter.ai/docs/api-reference/limits> (accessed 2026-11-20). Free model variants
(ID ending `:free`) get **20 requests/minute** and **50 requests/day** with fewer than 10 credits
purchased all time, **1,000/day** at ≥10 credits. Accounts and rate limits are global; "making
additional accounts or API keys will not affect your rate limits". At 50/day, 1,000 questions takes
20 days — only viable if someone buys the ~10 credits, which stops being a free option.

## Local alternative — Qwen3-4B on the 4 GB laptop

Estimate, not a measurement (no benchmark was run for this document): Qwen3-4B Q4_K_M at 4096 context
decodes at roughly 10–20 tok/s on a 4 GB Ampere laptop and prefills a 1,000-token passage at roughly
150–400 tok/s. Per question: ~3–7 s prefill + ~40 tokens at 15 tok/s ≈ 3 s decode → **6–10 s per
question**, so 1,000 questions ≈ **1.7–2.8 hours**. Mark these as estimates until ticket #2 measures
the real throughput.

Quality: a 4B model writing questions from a passage tends to copy the passage's phrasing — exactly
the lexical-shortcut failure mode described below. A hosted 20B+ model produces more paraphrased,
user-like questions. If quality matters more than independence, hosted wins; if the course forbids
third-party APIs for dataset construction, local is acceptable with heavier filtering.

## Generation plan

Prompt shape (one passage per call):

```
You are writing evaluation questions for a documentation search engine.
Read the passage below and write ONE question that a real user might ask,
which is answered by this passage.

Rules:
- Do not reuse the passage's title or headings.
- Paraphrase; do not copy more than 4 consecutive words from the passage.
- The question must be answerable from this passage alone.
- Output JSON only: {"question": "..."}

PASSAGE
<text>
```

Batch strategy: fixed passage list → queue file with one item per line → worker with concurrency 1–4,
retry with exponential back-off on 429/5xx, append completed IDs to a ledger so a re-run resumes.
Store `{id, question, gold_passage_id, model, prompt_hash, created_at}`. Never regenerate on retry:
dedupe by `gold_passage_id`.

## Quality control — concrete filters

Synthetic RAG questions fail in four repeatable ways. Apply these automatically:

1. **Answer leakage / lexical shortcut.** Reject if the question shares an 8-gram with the passage, or
   contains the passage's heading verbatim. (Fixes the "generated from the answer" cheat.)
2. **Unanswerable.** Ask the generator to also emit a supporting quote, then check the quote is a
   substring of the passage (fuzzy, ≥95% token match). Reject otherwise.
3. **Duplicate paraphrase.** Embed all questions; reject cosine > 0.95 against an existing one.
4. **Degenerate form.** Reject questions under 6 words or over 40, and anything containing "the
   passage", "the text", "according to the above".

Manual audit: **random 5% (≥50 questions)**, two reviewers, label each as answerable / leaky /
ambiguous. Target ≥90% answerable, <5% leaky. Report the audit numbers — they are what makes the
synthetic set defensible in the report.

Before any of this, produce the 1000 questions per domain in proportion to corpus size (Python ≈ 35%,
PostgreSQL ≈ 32%, Docker ≈ 19%, Git ≈ 14%) so no domain is starved.

## Recommendation

1. **Use Groq free (`openai/gpt-oss-120b`, 1,000 RPD)** for the single batch run. It is the only free
   tier whose documented daily limit matches the job, and 8K TPM only makes it a 2–3 hour job.
2. Record the exact model ID, the prompt hash, and the per-request count in the corpus manifest.
3. **Filter aggressively, audit 5%, and report the audit** — that is what separates a usable synthetic
   dev set from a pile of answer-leaked questions.
4. Keep local Qwen3-4B as the fallback if the terms check fails or the API is unavailable; budget an
   afternoon and expect more copy-paste phrasing.

## Confidence / gaps

High: Groq free-plan RPM/RPD/TPM/TPD and org-level scope; OpenRouter 20 RPM / 50 vs 1,000 RPD /
≥10-credit threshold; Gemini's "limits are per project, not guaranteed, see AI Studio" policy.
Medium: throughput arithmetic (depends on actual passage and question lengths).
Low / unverified: whether either provider's terms permit generating and redistributing a question set
derived from third-party documentation — check `console.groq.com/docs/your-data` and
`ai.google.dev/gemini-api/terms` before the batch. Local throughput is an estimate from architecture,
not a benchmark.
