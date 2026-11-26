# Local runtimes, model sizing, free compute, Python versions

Research compiled 2026-11-17. Every claim is linked to the page that owns it; all URLs accessed 2026-11-17.
Scope: 3-student team, each on their own laptop, needs (a) a local LLM behind an HTTP API that an evaluation
harness can drive unattended, (b) a working demo, (c) reproducible numbers for a report.

Fleet assumed:

| Machine | GPU | VRAM | Compute capability | Note |
|---|---|---|---|---|
| Laptop A | RTX 3050 Ti Laptop (Ampere) | 4 GB | 8.6 | listed as supported by Ollama — <https://docs.ollama.com/gpu> |
| Laptop B | GTX 1060 (Pascal) | 6 GB | 6.1 | same table, 6.1 row — <https://docs.ollama.com/gpu> |
| Laptop C | Apple Silicon **or** Intel Mac | unified / none | n/a | the two cases behave very differently, see 1.5 |

Two upstream facts constrain everything below:

* **CUDA 13 dropped Pascal.** PyTorch 2.15.0 states: "CUDA 13 wheels need NVIDIA driver 580 or newer and a GPU
  of Turing (`sm_75`) or newer, so Maxwell, Pascal and Volta GPUs and hosts with a CUDA 12.x driver should stay
  on 2.14" — <https://github.com/pytorch/pytorch/releases/tag/v2.15.0> (2026-10-28). Any Python-side GPU work on
  the GTX 1060 is therefore pinned to torch 2.14 + a CUDA 12.x wheel.
* **Assumption used for sizing (estimate, not a vendor figure): ~3.6 GB of the 4 GB card is usable** under Windows
  once the desktop compositor and driver reserve are subtracted. Treat every "fits 4 GB" verdict in §2 as
  "fits ~3.6 GB".

---

## 1. Cross-platform local LLM runtime

| | Ollama | llama.cpp (`llama-server`) | LM Studio | llama-cpp-python |
|---|---|---|---|---|
| Current version | v0.40.16 (2026-11-09) <sup>[1]</sup> | v1.0.0 (2026-11-13); nightlies b12474 (2026-11-17) <sup>[2]</sup> | 0.4.x app + `llmster` headless daemon <sup>[3]</sup> | 0.3.37 (2026-11-15) <sup>[4]</sup> |
| Linux | ✅ install script / tarball <sup>[5]</sup> | ✅ build from source or official binaries <sup>[6]</sup> | ✅ AppImage, Ubuntu 20.04+ <sup>[7]</sup> | ✅ pip (+ source build for CUDA) <sup>[8]</sup> |
| Windows | ✅ Win10 22H2+, native app <sup>[9]</sup> | ✅ official CUDA 12.4 / CPU zips <sup>[2]</sup> | ✅ x64 + ARM64, needs AVX2 <sup>[7]</sup> | ✅ wheels for cu118–cu132 <sup>[8]</sup> |
| macOS Apple Silicon | ✅ GPU (Metal) <sup>[10]</sup> | ✅ Metal on by default <sup>[6]</sup> | ✅ Apple Silicon + MLX <sup>[7]</sup> | ✅ Metal wheels <sup>[8]</sup> |
| macOS Intel | ⚠️ CPU only — "Apple M series (CPU and GPU support) or x86 (CPU only)" <sup>[10]</sup> | ⚠️ builds, no verified GPU path | ❌ "Intel-based Macs are currently not supported" <sup>[7]</sup> | ⚠️ CPU |
| RTX 3050 Ti (sm_86, 4 GB) | ✅ | ✅ | ✅ (no CC documented) | ✅ |
| GTX 1060 (sm_61, 6 GB) | ✅ **but driver ≥ 570** <sup>[11]</sup> | ✅ with the CUDA-12.x build <sup>[12]</sup> | ❓ undocumented (see gaps) | ✅ cu12x wheels: "compute capability is … 6.0 or newer for CUDA 12 wheels" <sup>[8]</sup> |
| HTTP API | REST `:11434`, OpenAI-compatible <sup>[13]</sup> | `/completion`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/rerank`, `/metrics`, `--api-key` <sup>[14]</sup> | OpenAI-compatible `:1234` <sup>[3]</sup> | `python -m llama_cpp.server` (FastAPI, OpenAI-ish) <sup>[8]</sup> |
| Pin exact model blob | ✅ `model[:tag]@sha256:…`, `/api/tags` returns "SHA256 digest identifier of the model contents" <sup>[15]</sup> | ✅ file on disk; record your own sha256 (`--hf-file` downloads by name, no verification) <sup>[14]</sup> | ❌ no documented content digest | ✅ file on disk + your own sha256 |
| Pin runtime version | ✅ GitHub release tag <sup>[1]</sup> | ✅ git tag / nightly tag | ⚠️ runtime packs are versioned (e.g. "llama.cpp 2.76.0 extension packs"), but only via app changelog <sup>[3]</sup> | ✅ pip version |
| Licence | MIT (open source) | MIT | proprietary freeware | MIT |

<sup>[1]</sup> <https://github.com/ollama/ollama/releases> · <sup>[2]</sup> <https://github.com/ggml-org/llama.cpp/releases> ·
<sup>[3]</sup> <https://lmstudio.ai/docs/developer/core/headless_llmster> and <https://lmstudio.ai/changelog> ·
<sup>[4]</sup> <https://pypi.org/pypi/llama-cpp-python/json> · <sup>[5]</sup> <https://docs.ollama.com/linux> ·
<sup>[6]</sup> <https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md> · <sup>[7]</sup> <https://lmstudio.ai/docs/app/system-requirements> ·
<sup>[8]</sup> <https://github.com/abetlen/llama-cpp-python> · <sup>[9]</sup> <https://docs.ollama.com/windows> ·
<sup>[10]</sup> <https://docs.ollama.com/macos> · <sup>[11]</sup> <https://docs.ollama.com/gpu> · <sup>[12]</sup> <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/CMakeLists.txt> ·
<sup>[13]</sup> <https://docs.ollama.com/api/introduction> · <sup>[14]</sup> <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md> · <sup>[15]</sup> <https://github.com/ollama/ollama/blob/main/types/model/name.go>

### 1.1 Ollama

> **Update 2026-11-26:** the team pin is now **v0.40.18** (2026-11-24); see
> `docs/setup/hardware.md` for install commands and measured rows. The version
> column above is left as written on 2026-11-17.

* GPU support statement: "Ollama supports Nvidia GPUs with compute capability 5.0+ and driver version 550 and newer.
  Nvidia GPUs with compute capability 5.0 through 6.2 require driver version 570 or newer."
  <https://docs.ollama.com/gpu> — i.e. the GTX 1060 still works, but **only** with a ≥570 driver on both Linux and
  Windows (the Windows page's "NVIDIA 551.61 or newer Drivers" floor is superseded by this per-architecture rule:
  <https://docs.ollama.com/windows>).
* Default context on <24 GiB VRAM is **4k**, which is exactly the range we need and keeps KV cache small
  <https://docs.ollama.com/context-length>.
* Flash Attention is used automatically when the backend supports it; KV cache can be quantized with
  `OLLAMA_KV_CACHE_TYPE=q8_0` ("approximately 1/2 the memory of `f16`") or `q4_0` (1/4)
  <https://docs.ollama.com/faq>. On Pascal the kernel falls back to non-tensor-core paths
  (`ggml_cuda_get_best_fattn_kernel` gates the MMA kernels on `turing_mma_available(cc)`:
  <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn.cu>), so expect slower prefill, not
  a hard failure.
* Reproducibility: model refs accept a digest suffix (`{model}[:{tag}][@{digest}]`,
  <https://github.com/ollama/ollama/blob/main/types/model/name.go>), and `/api/tags` reports the digest as
  "SHA256 digest identifier of the model contents" (<https://docs.ollama.com/api/tags>). Pin both the pulled
  digest and the Ollama release tag; note that upstream ships "Updated llama.cpp" in point releases
  (e.g. v0.40.16), so the inference engine version is only reproducible at the Ollama-tag level.
* One caveat for scoring: the Ollama version determines the bundled engine, so a "model + params" record is not
  enough — log `ollama -v` and the digest in every eval row.

### 1.2 llama.cpp directly

* Build: `cmake -B build -DGGML_CUDA=ON` (<https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md>).
  The default arch list is `50-virtual 61-virtual 70-virtual` + `75-virtual 80-virtual 86-real`
  (<https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/CMakeLists.txt>), and the release
  workflow deliberately builds the CUDA 12.8 artifacts with that broad default set "so the release binary covers
  many GPUs" (<https://github.com/ggml-org/llama.cpp/blob/master/.github/workflows/release.yml>). So the
  `llama-*-bin-ubuntu-cuda-12.8` and `llama-bin-win-cuda-12.4` zips run on Pascal; the CUDA 13.5 artifacts do not.
* Server defaults changed: default port is **9931**, and `-c/--ctx-size` defaults to `0` = take the model's
  training context (<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>) — set `-c 4096`
  explicitly or you will blow the 4 GB card's budget on a 32k-context model.
* Best reproducibility of the four: a pinned GGUF file + pinned tag + explicit `-c`, `-ngl`, `--seed` is a
  complete description of the run. `--metrics` gives Prometheus counters; `--api-key` gates the port.
* Cost: no GUI, no model manager, no auto-update — the team owns versions and CUDA toolkit.

### 1.3 LM Studio

* macOS is Apple Silicon only; Intel Macs are explicitly unsupported (<https://lmstudio.ai/docs/app/system-requirements>).
  Windows x64/ARM needs AVX2; "at least 4GB of dedicated VRAM is recommended". Linux is an AppImage for
  Ubuntu 20.04+ (newer than 22 "not well tested").
* Headless works and is documented: `lms daemon up`, `lms load …`, `lms server start` (port 1234), with a systemd
  unit example (<https://lmstudio.ai/docs/developer/core/headless_llmster>) — so it is scriptable, unlike the
  desktop-only impression the docs give elsewhere.
* No published minimum CUDA compute capability anywhere in the docs I could reach; the CUDA runtime is a
  closed binary "extension pack" (e.g. "llama.cpp 2.76.0 extension packs" in the Nov 2026 changelog:
  <https://lmstudio.ai/changelog>). Whether the 1060 works is an empirical question — see gaps.
* Pinning: you can pin the runtime pack version, but there is no content-hash model identity; models are keyed by
  publisher/name/path. For a report that must reproduce numbers, this is the weakest of the four.
* Proprietary licence, free to use — fine for a demo machine, awkward if the report claims "fully open stack".

### 1.4 llama-cpp-python

* Prebuilt CUDA wheels exist for cu118, cu121–cu125, cu130, cu132, with an explicit capability contract:
  "NVIDIA GPU compute capability is 6.0 through 8.9 for CUDA 11.8 wheels, 6.0 or newer for CUDA 12 wheels, or 7.5
  or newer for CUDA 13 wheels" (<https://github.com/abetlen/llama-cpp-python>). The 1060 is
  fine on cu12x, excluded on cu13x.
* Builds from source need `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python` and a CUDA toolkit — a
  recurring source of "works on my laptop" on a 3-person team.
* Value: in-process control (no server), direct `Llama(...)` object, `seed`, `n_gpu_layers`, logits access — handy
  if the harness wants token-level scoring (logprobs for answer-choice metrics) rather than text.
* Version 0.3.37 (2026-11-15) still declares `Requires-Python >=3.8` and publishes **no** cp-tagged wheels on PyPI
  (<https://pypi.org/pypi/llama-cpp-python/json>), so every install compiles.

### 1.5 The macOS split

* Apple Silicon: all four accelerate via Metal. llama.cpp: "On MacOS, Metal is enabled by default"
  (<https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md>). Ollama: M-series = "CPU and GPU support".
  LM Studio adds MLX (<https://lmstudio.ai/docs/app/system-requirements>). Expect the Mac to be the fastest
  machine of the three for 3–4B models at 4k.
* Intel Mac: Ollama is CPU-only by documentation (<https://docs.ollama.com/macos>), LM Studio refuses to run, and
  PyTorch MPS is not a fallback (MPS is Apple-Silicon-only; torch 2.15 additionally requires macOS 15+ for MPS:
  <https://github.com/pytorch/pytorch/releases/tag/v2.15.0>). Treat an Intel Mac as a CPU-only eval box; a 3B
  Q4_K_M model will be ~10–30× slower than on the 3050 Ti and is only usable for smoke tests.

### Recommendation (Q1)

1. **Standardise on Ollama** for the demo and the default eval path: one install story on all three OSes, a stable
   OpenAI-compatible endpoint, 4k context by default, digest pinning (`name@sha256:…`), and documented Pascal
   support. Two operational rules: everyone must be on the **same Ollama release tag**, and every eval row records
   `ollama -v` + model digest.
2. **Keep `llama-server` as the reference backend** for any number that goes into the report: it is the only stack
   where the model file, context size, seed, and engine commit are all explicitly pinned, and its CUDA-12.8 build
   is guaranteed to carry sm_61 kernels.
3. **Skip LM Studio** as a team dependency (no Intel Mac, no digest pinning, undocumented Pascal support) — it is
   fine as a one-off GUI on the 3050 Ti laptop.
4. **Use llama-cpp-python only if** the harness needs in-process logits; otherwise the extra source-build burden
   buys nothing over llama-server.
5. Have the 1060 owner install a ≥570 NVIDIA driver first; that single step decides whether Pascal is usable at all.

---

## 2. Model sizing for 4 GB and 6 GB at 4k–8k context

Method: Q4_K_M byte sizes are the actual file sizes listed in each GGUF repo on the Hub (checked 2026-11-17);
KV cache per token = `2 (K+V) × n_layers × n_kv_heads × head_dim × 2 bytes (f16)`, computed from each model's
`config.json`. Total = weights + KV + **0.4 GB** for CUDA context/compute buffers. KV numbers are per sequence
(no batching); with `OLLAMA_NUM_PARALLEL>1` KV scales linearly
(<https://docs.ollama.com/faq>).

| Model | Params | Licence | Q4_K_M file | KV f16 / token | 4k total | 8k total |
|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 1.24B | Llama 3.2 Community | 0.81 GB | 32 KiB | 1.3 GB | 1.5 GB |
| Qwen3-1.7B | 1.72B | Apache-2.0 | 1.11 GB | 112 KiB | 1.9 GB | 2.4 GB |
| Llama-3.2-3B-Instruct | 3.21B | Llama 3.2 Community | 2.02 GB | 112 KiB | 2.9 GB | 3.3 GB |
| Phi-4-mini-instruct | 3.84B | **MIT** | 2.49 GB | 128 KiB | 3.4 GB | 3.9 GB |
| Gemma-3-4B-it | 4.30B | Gemma Terms | 2.49 GB | 136 KiB | 3.4 GB | 4.0 GB |
| Qwen3-4B-Instruct-2507 | 4.02B | Apache-2.0 | 2.50 GB | 144 KiB | 3.5 GB | 4.0 GB |
| Mistral-7B-Instruct-v0.3 | 7.25B | Apache-2.0 | 4.37 GB | 128 KiB | 5.3 GB | 5.8 GB |
| Qwen3-8B | 8.19B | Apache-2.0 | 5.03 GB | 144 KiB | 6.0 GB | 6.6 GB |

Sources: file sizes and parameter counts from the model repos' file listings, e.g.
<https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/tree/main>,
<https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/tree/main>,
<https://huggingface.co/Qwen/Qwen3-8B-GGUF/tree/main>,
<https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/tree/main>,
<https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/tree/main>,
<https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/tree/main>;
architectures from each repo's `config.json`; licence flags from the repo metadata
(`cardData.license` = `apache-2.0`, `mit`, `llama3.2`, `gemma`).

Fit verdicts (usable VRAM ≈ 3.6 GB of the 4 GB card, ≈ 5.5 GB of the 6 GB card — estimates, see gaps):

* **4 GB (3050 Ti)** — comfortable: 1B/1.7B at 8k; 3B at 8k (3.3 GB, f16 KV) works on Linux and is tight on
  Windows. 4B models run at 4k (3.4–3.5 GB); at 8k they need `OLLAMA_KV_CACHE_TYPE=q8_0`, which halves KV
  (1.2 GB → ~0.6 GB, total ≈ 3.5 GB) — still tight, so prefer 3B if you need 8k
  (<https://docs.ollama.com/faq>). 7B/8B: no.
* **6 GB (1060)** — 4B at 8k is easy; Mistral-7B at 4k (5.3 GB) fits but leaves no headroom for the desktop;
  Qwen3-8B at 4k (6.0 GB) is over budget and will spill to CPU. Prefer 4B on this card, or 7B with q8_0 KV.
* **Pascal is slow, not broken**: no tensor cores → prefill is the bottleneck on RAG prompts (long contexts).
  Budget ~2–4× the 3050 Ti's prefill time for the same model; generation speed is bandwidth-bound and roughly
  proportional to `bandwidth / model_bytes`.

Other sizing facts worth putting in the report:

* Q4_K_M is the default quantization used by llama.cpp's own `-hf` downloader ("quant is optional … default to
  Q4_K_M", <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>) — a defensible default.
* KV quantization only works with Flash Attention enabled, which Ollama does automatically when supported
  (<https://docs.ollama.com/faq>).
* Licence split matters for a school report: **Apache-2.0** = Qwen3 family and Mistral-7B-v0.3; **MIT** =
  Phi-4-mini; **Llama 3.2 Community Licence** = Llama-3.2 (custom, not OSI); **Gemma Terms of Use** = Gemma-3.
  If the report/demo must be unencumbered, use Qwen3 or Phi-4-mini and avoid Gemma/Llama weights.

### Recommendation (Q2)

* **Default model: Qwen3-4B-Instruct-2507 (Q4_K_M, 2.5 GB, Apache-2.0)** — fits both cards at 4k; at 8k on the
  4 GB card only with q8_0 KV (~3.5 GB, tight). Permissively licensed.
* **Small/fast tier for GA inner loops: Qwen3-1.7B, or Llama-3.2-3B if you can live with the Llama licence.**
* **6 GB card may use Mistral-7B-Instruct-v0.3 (Apache-2.0) at 4k** when you want a bigger model for quality
  comparisons; keep it out of the 4 GB machine's config.
* Fix context at **4096** for all reported runs (matches Ollama's default for <24 GiB VRAM) and state KV dtype
  (f16 vs q8_0) next to every latency/memory number.

---

## 3. Free cloud options for repeated inference/eval runs (Nov 2026)

| Option | Hardware | Free allowance | Unattended / batch? | Catch |
|---|---|---|---|---|
| **Google Colab (free)** | GPU/TPU, model not documented | No stated hours; "access to expensive resources like GPUs is heavily restricted", resources "not guaranteed and not unlimited" | ❌ Not really: 12 h max runtime, idle timeouts, priority to interactive users, "background execution" is a paid-plan feature | ToS prohibits "remote control such as SSH shells, remote desktops bypassing the notebook UI", "running distributed computing workers", torrents, crypto <sup>[c1]</sup> |
| **Kaggle Notebooks** | 1× P100 16 GB or 2× T4 (29 GB RAM); TPU v3-8 | **30 GPU h/week** (resets weekly, "sometimes higher"), TPU **20 h/week**; 12 h/session (9 h TPU) | ✅ Yes: scheduled notebooks (daily/weekly/monthly), and `kaggle kernels push` runs a version headlessly | Interactive sessions die when idle (20 min per one page, 60 min per another); quota is unguaranteed; free GPU queues <sup>[c2]</sup> |
| **HF Spaces CPU Basic** | 2 vCPU, 16 GB RAM, 50 GB disk | $0/h | ⚠️ Sleeps when unused | **New Gradio/Docker Spaces require a paid plan (PRO)**; free accounts only get 2 ZeroGPU Gradio Spaces; static Spaces free <sup>[c3]</sup> |
| **HF ZeroGPU** | ½ or full RTX PRO 6000 Blackwell (48/96 GB) | **5 min GPU/day** free account, 40 min PRO, 2 min unauthenticated; 60 s default per call | ❌ Interactive/Gradio only, Gradio SDK only | Quota resets 24 h after first use; queue priority tied to remaining quota <sup>[c4]</sup> |
| **Modal** | T4/A10/L4/L40S/B200 etc. | **$30/month free compute** on Starter (T4 ≈ $0.000164/s ≈ **$0.59/h → ~50 T4-h/month**) | ✅ Yes: scheduled functions (limited on Starter), batch jobs, per-second billing | Model APIs are **not** covered by the free credit (since 2026-09-01) <sup>[c5]</sup> |
| **GitHub Codespaces** | 2/4-core CPU, **no GPU** | 120 h/month of compute, billed per core (so 60 h on a 2-core box) + 15 GB-month storage (Free plan) | ✅ (a VM you drive) | CPU only → fine for embeddings/retrieval evals, useless for GPU LLM throughput numbers <sup>[c6]</sup> |
| **Oracle Cloud Always Free** | Ampere A1 (Arm), 2× E2.1.Micro | 1,500 OCPU-h + 9,000 GB-h/month ≈ **2 OCPU / 12 GB continuously**; 200 GB block | ✅ 24/7 server | No GPU, Arm-only for A1 → a CPU inference / embedding / index host, not an LLM-throughput box <sup>[c7]</sup> |
| **Google Cloud free tier** | e2-micro (us-west1/us-central1/us-east1) | 1 e2-micro + 30 GB disk | ✅ | "GPUs and TPUs are not included in the Free Tier offer"; the $300 trial forbids adding GPUs <sup>[c8]</sup> |
| **AWS Free plan** | select services | $100 + up to $100 credits, **6 months**, account auto-closes | ✅ | No GPU allowance; the page now describes only the credit-based plan (no 750 h/12-month EC2 offer) <sup>[c9]</sup> |
| **Lightning AI** | T4/L4/A100/… | "Up to 50 free GPU hours" + "up to 30 credits to start"; 1 free Studio with 4-hour restarts | ⚠️ Studios restart every 4 h on free tier | Headline "50 GPU hours" is marketing with an asterisk; free plan table says "up to 30 credits" — treat as one-off, not recurring <sup>[c10]</sup> |

<sup>[c1]</sup> <https://research.google.com/colaboratory/faq> · <sup>[c2]</sup> <https://www.kaggle.com/docs/notebooks> (§"Adding a free GPU", 12 h/9 h session limits), <https://www.kaggle.com/docs/efficient-gpu-usage> ("quota … is 30 hours or sometimes higher"), <https://www.kaggle.com/docs/tpu> ("20 hours per week of TPUs and up to 9h at a time"), <https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md> (`kaggle kernels push` "runs the kernel") ·
<sup>[c3]</sup> <https://huggingface.co/docs/hub/spaces-overview> · <sup>[c4]</sup> <https://huggingface.co/docs/hub/spaces-zerogpu> ·
<sup>[c5]</sup> <https://modal.com/pricing> · <sup>[c6]</sup> <https://docs.github.com/en/billing/concepts/product-billing/github-codespaces> ·
<sup>[c7]</sup> <https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm> ·
<sup>[c8]</sup> <https://cloud.google.com/free/docs/free-cloud-features> · <sup>[c9]</sup> <https://aws.amazon.com/free/> ·
<sup>[c10]</sup> <https://lightning.ai/pricing>

Dead ends that blog posts still recommend (verified dead):

* **GitHub Models** — "As of July 30, 2026, GitHub Models has been fully retired. The playground, model catalog,
  inference API … are no longer available to any customer." <https://docs.github.com/en/github-models>
* **SageMaker Studio Lab** — "no longer open to new customers" <https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab.html>
* **Colab as a headless worker** — explicitly in the prohibited list (SSH shells / remote desktops "bypassing the
  notebook UI", distributed workers) <https://research.google.com/colaboratory/faq>; background execution is a
  paid feature.
* **HF Spaces free CPU as a long-running endpoint** — new Gradio/Docker Spaces now need PRO; free hardware sleeps
  when idle <https://huggingface.co/docs/hub/spaces-overview>.

Hosted-API free tiers are a different tool (they cannot run *your* pinned GGUF, so they cannot produce the
throughput/memory numbers the report needs, but they are fine for a second opinion on answer quality):
Gemini API free tier with per-model RPM/RPD limits (<https://ai.google.dev/gemini-api/docs/rate-limits>),
OpenRouter `:free` models at 20 req/min and 50 req/day (1,000/day after ≥10 credits purchased)
(<https://openrouter.ai/docs/api_reference/limits>), Groq per-model RPD/TPM tables
(<https://console.groq.com/docs/rate-limits>).

### Recommendation (Q3)

1. **Free unattended batch → Kaggle Notebooks.** 30 GPU-h/week per account, P100 or 2×T4, scheduled daily runs,
   and `kaggle kernels push` to execute a 12-hour version headlessly. Three students = three accounts = up to
   90 GPU-h/week, which is plenty for a GA over prompt/retrieval configs.
2. **When Kaggle's queue or quota bites → Modal**, $30/month of T4/A10 compute ≈ 50 T4-hours, real cron scheduling,
   per-second billing, no ToS games. This is the only free-tier option that behaves like a normal cloud.
3. **Colab only for interactive exploration**, and never script it: the ToS language about SSH/remote control and
   distributed workers is exactly what an eval harness looks like from the outside.
4. **Do not plan on HF Spaces free CPU or ZeroGPU** for evals (5 min GPU/day, sleeps when idle, PRO required for new
   Gradio Spaces). Use a Space only as the public demo link if you want one.
5. **Codespaces / Oracle A1 are the right home for the CPU half** of the pipeline (embeddings, BM25, reranking,
   metric computation) — 120 h/month of compute billed per core (60 h on a 2-core box) or a 24/7 2-OCPU Arm box,
   both free.

---

## 4. Python version support (Nov 2026)

**Direct answer: yes — the current stable PyTorch ships CPython 3.14 wheels for Linux.**

* PyTorch **2.15.0** (released 2026-10-28) supports Python 3.11–3.15 and publishes
  `torch-2.15.0-cp314-cp314-manylinux_2_28_x86_64.whl` (also aarch64 and macOS arm64, plus cp314t free-threaded
  builds): <https://github.com/pytorch/pytorch/releases/tag/v2.15.0> and <https://pypi.org/pypi/torch/json>.
* Same release raises the binary floor to 3.11 (no cp310 wheels) and makes the PyPI wheel CUDA **13.2**, which
  requires driver ≥580 and `sm_75+` — so on the **GTX 1060 you must install torch 2.14** (`cu126` index); 2.14.0
  also publishes cp314 Linux wheels (<https://pypi.org/pypi/torch/json>), so Pascal + Python 3.14 is possible.
* torch 2.15 also raises the MPS floor to **macOS 15** for Apple Silicon (macOS 14 → `mps.is_available()` false).

Rest of the stack (PyPI metadata checked 2026-11-17):

| Package | Version | `Requires-Python` | cp314 Linux wheel? |
|---|---|---|---|
| torch | 2.15.0 | `>=3.11` | ✅ `cp314-manylinux_2_28_x86_64` |
| transformers | 5.23.0 | `>=3.11.0` | ✅ pure-python (`py3-none-any`) |
| tokenizers (transformers dep) | 0.23.3 | `>=3.10` | ✅ abi3 wheel tagged cp310/cp314 |
| sentence-transformers | 6.3.0 | `>=3.10` | ✅ pure-python |
| faiss-cpu | 1.16.0 | `>=3.10` | ✅ cp310…cp314, uploaded 2026-11-10 |
| chromadb | 1.5.9 | `>=3.9` | ✅ `cp39-abi3` (stable ABI, works on 3.9+) |
| onnxruntime (chromadb default embedder) | 1.32.0 | `>=3.11` | ✅ cp311…cp314 |
| numpy | 2.5.5 | `>=3.12` | ✅ cp312…cp315 |
| scipy | 1.18.1 | `>=3.12` | ✅ cp312…cp315 |
| llama-cpp-python | 0.3.37 | `>=3.8` | ❌ no cp-tagged wheels at all (sdist → compiles) |

Sources: <https://pypi.org/pypi/transformers/json>, <https://pypi.org/pypi/sentence-transformers/json>,
<https://pypi.org/pypi/faiss-cpu/json>, <https://pypi.org/pypi/chromadb/json>,
<https://pypi.org/pypi/onnxruntime/json>, <https://pypi.org/pypi/tokenizers/json>,
<https://pypi.org/pypi/numpy/json>, <https://pypi.org/pypi/scipy/json>,
<https://pypi.org/pypi/llama-cpp-python/json>.

Consequences:

* 3.14 is **not** blocked by any of the core RAG dependencies. The only package that will not give you a wheel is
  `llama-cpp-python` (true on every Python version, not a 3.14 problem).
* 3.11 is the version to avoid going forward: numpy 2.5.5 and scipy 1.18.1 already require `>=3.12`, so a 3.11 venv
  silently pins older scientific stacks.
* PyTorch 2.15's CUDA 13 wheels mean **the Python version and the GPU generation are coupled** on the Pascal box:
  install `torch==2.14.*` from the `cu126` index there, and do not let a `pip install -U torch` sneak in 2.15.

### Recommendation (Q4)

* **Pin Python 3.13.x for the team** (all wheels above exist; one minor behind the newest; avoids being the first
  to hit 3.14-only packaging bugs).
* **3.14 is acceptable** if someone prefers it — evidence shows torch 2.14/2.15 + transformers + faiss + chromadb +
  sentence-transformers all install from wheels — but then `llama-cpp-python` still compiles from source.
* **Never 3.11** for new envs; 3.15 is too new for a semester project.
* Pin `torch==2.14.*` + `--index-url https://download.pytorch.org/whl/cu126` on the GTX 1060 machine, and
  `torch==2.15.*` (cu13x) elsewhere; record both in the report's environment table.

---

## Confidence / gaps

High confidence (read directly from the owning page, quoted above): Ollama's compute-capability/driver rules and
macOS Intel CPU-only status; llama.cpp's default CUDA arch list and release-build policy; LM Studio's Apple-Silicon-only
macOS support, headless CLI and 4 GB VRAM guidance; llama-cpp-python's per-CUDA capability matrix; PyTorch 2.15's
Python/CUDA/macOS floors and cp314 wheel availability; all Python package `Requires-Python` and cp314 wheel
presence; Kaggle's 30 GPU-h/week and 12 h/9 h session limits; HF ZeroGPU's 5/40 min daily quotas; Modal's
$30/month and T4 per-second price; Codespaces' 120 h/month compute quota; Oracle's 1,500 OCPU-h A1 allowance; GCP's "GPUs
are not included"; AWS's $200/6-month free plan; GitHub Models' retirement and Studio Lab's closure.

Unverified / gaps:

1. **LM Studio on Pascal (GTX 1060).** No published minimum compute capability; the CUDA runtime is a closed
   binary. Needs a 10-minute empirical test on the machine. If it fails, that laptop uses Ollama or llama.cpp.
2. **Colab free-tier GPU model.** The FAQ no longer names the accelerator ("GPU and TPU", heavily restricted,
   <https://research.google.com/colaboratory/faq>); the plans page renders client-side and I could not read a
   hardware table. Assume T4-class at best, but do not cite a model.
3. **HF Spaces free-CPU sleep duration.** Docs say free hardware sleeps "after a period of time if unused" without
   a number (<https://huggingface.co/docs/hub/spaces-overview>); the old 48 h figure is not currently documented.
4. **Kaggle scheduling limits.** Kaggle says "there are limits to the number of notebooks you can have scheduled"
   without publishing the number (<https://www.kaggle.com/docs/notebooks>), and its two docs pages disagree on the
   interactive idle timeout (20 min vs 60 min) — measure it rather than quoting one.
5. **Lightning AI free hours.** The pricing page states both "up to 50 free GPU hours" (headline, asterisked) and
   "up to 30 credits to start" (plan table); the real recurring allowance is unclear.
6. **Throughput numbers are estimates.** No tokens/s benchmark was run for this document; the prefill/generation
   guidance in §2 is derived from architecture (no tensor cores on Pascal; bandwidth-bound decode) and should be
   replaced by measured numbers on the three laptops before it goes in the report.
7. **Kaggle docs were read through a rendering proxy** (`r.jina.ai`) because the docs site is client-rendered; the
   text is Kaggle's own, but quote it from kaggle.com directly before submitting the report.
8. **Ollama digest pulls are documented only in source, not in the user docs.** The `model[:tag]@sha256:…` grammar
   comes from `types/model/name.go`; the user-facing docs describe the digest only as reported by `/api/tags`.
   Verify `ollama pull <model>@sha256:…` works on the pinned release before relying on it in the harness.
9. **Licence texts were not read in full** — only the Hub `license` metadata. The Llama 3.2 Community Licence and
   Gemma Terms have use/redistribution clauses that a published demo may need to respect; read the licence files
   before shipping weights in a public repo.
