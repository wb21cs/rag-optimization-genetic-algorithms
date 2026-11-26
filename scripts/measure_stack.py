#!/usr/bin/env python3
"""Measure the pinned local inference stack on one machine.

Stdlib only, so it runs on Linux, Windows and macOS with a bare Python 3.8+.
It drives Ollama's HTTP API, so the numbers are derived from the server's own
counters (prompt_eval_count/eval_count and their durations), not from a
wall-clock stopwatch.

Protocol (see docs/setup/hardware.md):
  * two fixed prompts, sized ~512 and ~2048 tokens, identical bytes everywhere
  * num_ctx=4096, temperature=0, seed=42, num_predict=256
  * per model: unload, one cold run at 2048 (real load_duration), three warm
    runs at 2048, three warm runs at 512
  * VRAM sampled from nvidia-smi every 100 ms while a run is in flight
  * residency from /api/ps: size_vram / size

Usage:
    python3 scripts/measure_stack.py --label "Laptop A - Linux RTX 3050 Ti"
    python3 scripts/measure_stack.py --label "Laptop B" --out results.json
"""

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------- thresholds

MIN_DECODE_TPS = 15.0       # token generation, warm
MIN_PREFILL_TPS = 50.0      # prompt processing at 2048 tokens, warm
MAX_LOAD_S = 60.0           # cold model load
RESIDENT_FRACTION = 0.99    # size_vram / size below this = spilled to CPU

DEFAULT_MODELS = [
    "qwen3:4b-instruct-2507-q4_K_M",
    "qwen3:1.7b",
]

NUM_CTX = 4096
NUM_PREDICT = 256
SEED = 42
WARM_RUNS = 3
SMALL_TOKENS = 512
LARGE_TOKENS = 2048

# ---------------------------------------------------------------- fixed prompt

PARAGRAPH = (
    "A retrieval pipeline begins by splitting a document into passages, then "
    "encoding each passage into a dense vector that is written to an index. "
    "At query time the same encoder maps the question into that vector space, "
    "the index returns the nearest passages, and the language model answers "
    "using only the passages supplied in its context window. Every stage "
    "carries a cost: the chunker decides how much surrounding text survives "
    "a split, the encoder fixes the geometry of the index, and the context "
    "budget fixes how many passages can be shown at once. Prefill processes "
    "the prompt tokens in parallel and grows with the square of the context "
    "length, while decode generates one token at a time and is bound by the "
    "bandwidth of whichever device holds the weights. "
)

QUESTION = "\n\nQuestion: Summarise the pipeline above in one short paragraph.\nAnswer:"


def build_prompt(target_tokens):
    """Deterministic, byte-identical on every machine. Token count is read
    back from prompt_eval_count; the estimate is only for sizing."""
    target_chars = target_tokens * 4
    body = PARAGRAPH * max(1, target_chars // len(PARAGRAPH))
    return body + QUESTION


# ---------------------------------------------------------------- HTTP helpers

HOST = None


def _url(path):
    return HOST.rstrip("/") + path


def _request(path, payload=None, timeout=600):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(_url(path), data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def get_json(path, timeout=30):
    return _request(path, None, timeout)


def post_json(path, payload, timeout=600):
    return _request(path, payload, timeout)


def wait_for_server():
    for _ in range(30):
        try:
            return get_json("/api/version", timeout=5)
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    return None


# ---------------------------------------------------------------- GPU sampling

def gpu_mem_used_mb():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output(
            [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return float(out.strip().splitlines()[0])
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def gpu_total_mb():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output(
            [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return float(out.strip().splitlines()[0])
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def gpu_name_and_driver():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None
    try:
        out = subprocess.check_output(
            [exe, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True, timeout=5,
        ).strip().splitlines()[0]
        name, driver = [p.strip() for p in out.split(",")[:2]]
        return name, driver
    except (subprocess.SubprocessError, OSError, IndexError):
        return None, None


class VramSampler(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self.peak = 0.0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            v = gpu_mem_used_mb()
            if v is not None:
                self.samples.append(v)
                self.peak = max(self.peak, v)
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        self.join(timeout=2)


# ---------------------------------------------------------------- machine facts

def physical_cores():
    try:
        if platform.system() == "Linux":
            out = subprocess.check_output(["lscpu"], text=True, timeout=5)
            cores = int(re.search(r"Core\(s\) per socket:\s+(\d+)", out).group(1))
            sockets = int(re.search(r"Socket\(s\):\s+(\d+)", out).group(1))
            return cores * sockets
    except Exception:
        pass
    try:
        if platform.system() == "Darwin":
            return int(subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"], text=True, timeout=5).strip())
    except Exception:
        pass
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).NumberOfCores"],
                text=True, timeout=15,
            )
            return int(out.strip().splitlines()[0])
    except Exception:
        pass
    return None


def total_ram_gb():
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo", encoding="utf-8") as fh:
                kb = int(re.search(r"MemTotal:\s+(\d+)\s+kB", fh.read()).group(1))
            return round(kb / 1048576, 1)
        if platform.system() == "Darwin":
            return round(int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, timeout=5).strip()) / 1e9, 1)
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                text=True, timeout=15,
            )
            return round(int(out.strip().splitlines()[0]) / 1e9, 1)
    except Exception:
        pass
    return None


def machine_info():
    gpu, driver = gpu_name_and_driver()
    return {
        "hostname": platform.node(),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "physical_cores": physical_cores(),
        "logical_cores": os.cpu_count(),
        "ram_gb": total_ram_gb(),
        "gpu": gpu,
        "driver": driver,
        "vram_total_mb": gpu_total_mb(),
    }


# ---------------------------------------------------------------- Ollama calls

def model_digests():
    try:
        tags = get_json("/api/tags")
    except (urllib.error.URLError, OSError):
        return {}
    return {m["name"]: m.get("digest") for m in tags.get("models", [])}


def loaded_models():
    try:
        return get_json("/api/ps").get("models", [])
    except (urllib.error.URLError, OSError):
        return []


def unload(model):
    try:
        post_json("/api/generate",
                  {"model": model, "prompt": "", "keep_alive": 0}, timeout=60)
    except (urllib.error.URLError, OSError):
        pass
    for _ in range(60):
        names = [m.get("name") or m.get("model") for m in loaded_models()]
        if model not in names:
            return
        time.sleep(0.5)


def run_once(model, prompt, sampler):
    started = time.time()
    resp = post_json("/api/generate", {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "temperature": 0,
            "seed": SEED,
        },
    })
    resp["_wall_s"] = time.time() - started
    if sampler is not None:
        time.sleep(0.2)  # let VRAM sampling catch the tail of usage
    return resp


def ns_to_s(value):
    return (value or 0) / 1e9


def metrics(resp):
    prompt_n = resp.get("prompt_eval_count") or 0
    eval_n = resp.get("eval_count") or 0
    prefill_s = ns_to_s(resp.get("prompt_eval_duration"))
    decode_s = ns_to_s(resp.get("eval_duration"))
    return {
        "prompt_tokens": prompt_n,
        "output_tokens": eval_n,
        "prefill_tps": (prompt_n / prefill_s) if prefill_s > 0 else 0.0,
        "decode_tps": (eval_n / decode_s) if decode_s > 0 else 0.0,
        "load_s": ns_to_s(resp.get("load_duration")),
        "total_s": ns_to_s(resp.get("total_duration")),
        "wall_s": resp.get("_wall_s", 0.0),
    }


def residency(model):
    for entry in loaded_models():
        name = entry.get("name") or entry.get("model")
        if name == model:
            size = entry.get("size") or 0
            vram = entry.get("size_vram") or 0
            fraction = (vram / size) if size else 0.0
            return {
                "size_gb": round(size / 1e9, 2),
                "size_vram_gb": round(vram / 1e9, 2),
                "resident": fraction,
                "percent_gpu": round(fraction * 100),
            }
    return None


def measure_model(model, label, kv_type, warm_runs=WARM_RUNS, verbose=True):
    if verbose:
        print(f"\n=== {label} :: {model} (KV {kv_type}) ===", file=sys.stderr)

    small = build_prompt(SMALL_TOKENS)
    large = build_prompt(LARGE_TOKENS)

    unload(model)
    baseline = [v for v in (gpu_mem_used_mb() for _ in range(5)) if v is not None]
    baseline_mb = statistics.median(baseline) if baseline else None

    sampler = VramSampler()
    sampler.start()

    cold = run_once(model, large, sampler)
    if verbose:
        print(f"  cold  large: {metrics(cold)['prefill_tps']:.1f} tok/s prefill, "
              f"load {metrics(cold)['load_s']:.1f}s", file=sys.stderr)

    warm_large = [run_once(model, large, sampler) for _ in range(warm_runs)]
    warm_small = [run_once(model, small, sampler) for _ in range(warm_runs)]

    sampler.stop()

    res = residency(model)
    large_m = [metrics(r) for r in warm_large]
    small_m = [metrics(r) for r in warm_small]
    cold_m = metrics(cold)

    def med(rows, key):
        return statistics.median([r[key] for r in rows]) if rows else 0.0

    result = {
        "label": label,
        "model": model,
        "kv": kv_type,
        "prompt_tokens_small": int(med(small_m, "prompt_tokens")),
        "prompt_tokens_large": int(med(large_m, "prompt_tokens")),
        "decode_tps": round(med(large_m, "decode_tps"), 1),
        "prefill_tps_large": round(med(large_m, "prefill_tps"), 1),
        "prefill_tps_small": round(med(small_m, "prefill_tps"), 1),
        "load_s": round(cold_m["load_s"], 1),
        "cold_prefill_tps": round(cold_m["prefill_tps"], 1),
        "peak_vram_mb": round(sampler.peak, 0) if sampler.peak else None,
        "baseline_vram_mb": baseline_mb,
        "vram_delta_mb": (round(sampler.peak - baseline_mb, 0)
                          if sampler.peak and baseline_mb is not None else None),
        "residency": res,
        "runs": warm_runs,
    }
    result["verdict"], result["reasons"] = verdict(result)
    if verbose:
        print(f"  verdict: {result['verdict']}", file=sys.stderr)
        for reason in result["reasons"]:
            print(f"    - {reason}", file=sys.stderr)
    return result


def verdict(r):
    reasons = []
    res = r.get("residency") or {}
    if not res:
        reasons.append("model not listed in /api/ps after a run")
    elif res.get("resident", 0) < RESIDENT_FRACTION:
        reasons.append(f"spilled to CPU ({res.get('percent_gpu')}% GPU)")
    if r["decode_tps"] < MIN_DECODE_TPS:
        reasons.append(f"decode {r['decode_tps']} tok/s < {MIN_DECODE_TPS}")
    if r["prefill_tps_large"] < MIN_PREFILL_TPS:
        reasons.append(f"prefill@2048 {r['prefill_tps_large']} tok/s < {MIN_PREFILL_TPS}")
    if r["load_s"] > MAX_LOAD_S:
        reasons.append(f"load {r['load_s']}s > {MAX_LOAD_S}s")
    return ("works" if not reasons else "smoke-only"), reasons


def fallback_commands(model, kv_type):
    if kv_type != "f16":
        return []
    return [
        "# 4B spilled at f16 KV. Restart the server with a quantised KV cache,",
        "# then re-run this model:",
        "export OLLAMA_KV_CACHE_TYPE=q8_0",
        "ollama serve    # restart the server with the variable set",
        f"python3 scripts/measure_stack.py --models {model}",
    ]


# ---------------------------------------------------------------- output

def markdown_row(r):
    res = r.get("residency") or {}
    peak = f"{r['peak_vram_mb'] / 1024:.2f}" if r.get("peak_vram_mb") else "n/a"
    delta = f"{r['vram_delta_mb'] / 1024:.2f}" if r.get("vram_delta_mb") is not None else "n/a"
    pct = f"{res.get('percent_gpu')}%" if res else "n/a"
    model = r["model"].split("/")[-1]
    return (f"| {r['label']} | {model} | {r['kv']} | {r['decode_tps']} | "
            f"{r['prefill_tps_large']} | {peak} | {delta} | {r['load_s']} | "
            f"{pct} | {r['verdict']} |")


HEADER = ("| Machine | Model | KV | Decode tok/s | Prefill tok/s @2048 | "
          "Peak VRAM (GB) | VRAM delta (GB) | Load (s) | Resident | Verdict |")


def main(argv=None):
    global HOST
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True, help="machine label for the table row")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--kv", default="f16", help="KV cache dtype in use (f16/q8_0)")
    parser.add_argument("--warm-runs", type=int, default=WARM_RUNS)
    parser.add_argument("--out", help="write full JSON to this path")
    args = parser.parse_args(argv)

    HOST = args.host if args.host.startswith("http") else "http://" + args.host

    version = wait_for_server()
    if not version:
        print(f"error: no Ollama server at {HOST}. Start it with: ollama serve",
              file=sys.stderr)
        return 2

    digests = model_digests()
    missing = [m for m in args.models if m not in digests]
    if missing:
        print("error: model(s) not pulled: " + ", ".join(missing), file=sys.stderr)
        for m in missing:
            print(f"  ollama pull {m}", file=sys.stderr)
        return 2

    info = machine_info()
    info["ollama"] = version.get("version")
    info["digests"] = {m: digests.get(m) for m in args.models}
    info["env"] = {
        "OLLAMA_NUM_PARALLEL": _env("OLLAMA_NUM_PARALLEL"),
        "OLLAMA_MAX_LOADED_MODELS": _env("OLLAMA_MAX_LOADED_MODELS"),
        "OLLAMA_CONTEXT_LENGTH": _env("OLLAMA_CONTEXT_LENGTH"),
        "OLLAMA_KV_CACHE_TYPE": _env("OLLAMA_KV_CACHE_TYPE"),
        "OMP_NUM_THREADS": _env("OMP_NUM_THREADS"),
    }

    print("## machine", file=sys.stderr)
    for key, value in info.items():
        if key != "digests":
            print(f"  {key}: {value}", file=sys.stderr)

    results = []
    for model in args.models:
        results.append(measure_model(model, args.label, args.kv, args.warm_runs))

    payload = {"machine": info, "results": results}
    print("\n" + HEADER)
    for r in results:
        print(markdown_row(r))

    fallbacks = []
    for r in results:
        if r["verdict"] != "works":
            fallbacks.extend(fallback_commands(r["model"], r["kv"]))

    if fallbacks:
        print("\n## fallback")
        for line in fallbacks:
            print(line)

    print("\n## json")
    print(json.dumps(payload, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


def _env(name):
    return os.environ.get(name)


if __name__ == "__main__":
    sys.exit(main())
