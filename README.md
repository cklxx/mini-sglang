# mini-sglang

A minimal, streaming-first implementation of an sglang-style text generation stack with clear API / Engine / Backend separation. Benchmark comparisons now contrast streaming mini-sglang against a plain Hugging Face `generate()` baseline (non-sglang) rather than exposing a non-streaming HTTP mode.

### One-line quickstart (auto installs + modern small model)

> Prefer `uv`? A `pyproject.toml` is now committed so you can run `uv sync` to
> install deps from the project metadata instead of the requirements file.

If you have [uv](https://github.com/astral-sh/uv) installed, you can bootstrap dependencies, download a modern small instruct model, and run streaming **and** traditional generation side by side with a single command (CPU/Mac friendly):

```bash
uv run python one_click_compare.py "Hello mini-sglang"
```

No UV? The same script will auto-install uv (unless you set `AUTO_INSTALL_UV=0`) and fall back to `pip` if needed. You can also skip installation by passing `--no-bootstrap` if your environment is already set up:

```bash
python one_click_compare.py "Hello mini-sglang"
```

To benchmark a short chat with multiple user turns (history preserved for both
streaming and vanilla baselines), pass several prompts plus `--multi-turn`:

```bash
uv run python one_click_compare.py --multi-turn \
  "Hello mini-sglang" \
  "What's different about streaming?" \
  "Give me a TL;DR"
```

If you run `python one_click_compare.py` with no prompt, it will default to the preset benchmark suite (short, long, and two-turn scenarios) and print a throughput comparison table.
For realism, the default benchmark uses `max_new_tokens=512` when no prompt is provided.

Readable INFO logs narrate every prefill/decode step, so learners can follow the entire pipeline end to end.

## Project layout
```
requirements.txt
config.py           # (sglang global config analogue; source file: [global_config.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/global_config.py)) model choice + device selection
backend/model_backend.py  # (sglang backend analogue; source file: [model_runner.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py)) model + tokenizer + KV cache helpers
engine/engine.py    # (sglang engine analogue; source file: [entrypoints/engine.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/engine.py)) prefill + decode orchestration
api/server.py       # (sglang API analogue; source file: [entrypoints/http_server.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py)) FastAPI server exposing POST /generate
cli_demo.py         # Minimal client to stream tokens in a terminal (sglang CLI source: [cli/generate.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/cli/generate.py))
benchmark.py        # Streaming vs vanilla generate() comparison helper (benchmarks in [bench_serving.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_serving.py))
one_click_compare.py  # All-in-one bootstrap + streaming vs traditional demo
```

## Usage
Install dependencies (CPU/MPS/CUDA torch resolved by pip environment). To avoid downloading large GPU wheels on CPU-only machines you can point pip at the CPU index (Linux) or rely on the default Mac wheels:
```bash
# CPU-only install
pip install --index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Or use the default index (may pull CUDA wheels on Linux)
pip install -r requirements.txt

# Or install via uv using the committed pyproject
uv sync
```

The one-click script mirrors this behavior: on Linux it defaults to the CPU index unless you set `ALLOW_CUDA_TORCH=1`, and you can override the index explicitly with `TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu uv run ...`. On macOS the default pip wheels cover CPU and MPS. If uv is missing, the script will attempt to install it automatically so the remaining steps still run with a single command.

CLI demo (streaming-only):
```bash
python cli_demo.py

# Multi-turn chat mode (keeps prior turns in context)
python cli_demo.py --multi-turn
```

End-to-end smoke test with the default model (streaming only):
```bash
python smoke_test.py --max-new-tokens 32 "Hello from mini-sglang"
```
The streaming run exercises the full prefill + decode path and streams tokens to stdout. For non-sglang baselines, use the comparison or benchmark scripts below.

Run a quick benchmark comparing streaming vs. vanilla Hugging Face `generate()` (non-sglang baseline):
```bash
python benchmark.py --max-new-tokens 64 "Benchmarking mini-sglang"
```
The benchmark prints token counts, wall-clock duration, and throughput for both modes so you can evaluate normal inference speed versus the step-by-step streaming loop.

### One-click side-by-side script (auto-downloads small instruct model)

If you just want a single command that initializes everything, downloads a small instruct model, and prints streaming vs. traditional outputs with readable logs, run the quickstart above or:

```bash
python one_click_compare.py "Hello mini-sglang"
```

Flags:

* `--max-new-tokens`: token budget for both modes (defaults to 128)
* `--model`: override the default modern small model
* `--multi-turn`: treat multiple prompts as sequential chat turns
* `--compile-model`: try `torch.compile` for extra speed (best effort)
* `--warmup-tokens` / `--no-warmup`: control a short warmup generation to amortize cold start
* `--no-optimizations`: disable torch perf knobs and warmup
* `--no-bootstrap`: skip auto-install if deps are pre-installed

Performance knobs (inspired by sglang defaults):
- TF32/benchmark flags for CUDA and tuned matmul precision (set via env `MATMUL_PRECISION`, `ENABLE_SDP=0` to disable flash/mem-efficient attention on CUDA).
- `torch.compile` opt-in (`--compile-model` or `COMPILE_MODEL=1`, `COMPILE_MODE` to adjust mode).
- Autocast + inference_mode around forward/generate to cut Python/dispatch overhead.
- One-shot warmup run before benchmarks (skip with `--no-warmup` or `--no-optimizations`).
- Decode input buffer reuse to avoid per-step tensor allocs (`DECODE_BUFFER=0` to disable).
- Token and prefill KV LRU caches to skip repeat tokenization and prefill (`TOKEN_CACHE_SIZE`, `PREFILL_CACHE_SIZE`).
- Server startup warmup to amortize first-request latency.

Logs at INFO level narrate every prefill/decode stream chunk and the traditional `generate()` call so beginners can follow the full flow. Summaries include token counts and throughput for both modes.

Start HTTP server:
```bash
uvicorn api.server:app --reload --port 8000
```

Send a streaming request (SSE-style text chunks):
```bash
curl -N -X POST \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "stream": true}' \
  http://localhost:8000/generate
```

Responses are JSON lines streamed chunk-by-chunk, e.g.:
```
{"text_delta": " sample"}
...
{"event": "done"}
```

The HTTP API is streaming-only; for non-sglang baselines use `benchmark.py` or `one_click_compare.py`.

Benchmark the server path (TTFB + throughput):
```bash
python server_benchmark.py --max-new-tokens 128
```

One-click bench (sglang vs HF streaming vs HTTP server):
```bash
python one_click_bench.py --max-new-tokens 128
```
To stream via HF baseline on the server, set `mode="hf"` in the request body.

Multi-device CUDA (round robin):
- Enabled by default when multiple CUDA GPUs exist (`ENABLE_MULTI_DEVICE=0` to force single device).
- The server will instantiate one engine per GPU and round-robin requests across them, warming each on startup.

### Learning-friendly logs

The CLI, smoke test, and FastAPI server enable INFO-level logging by default. Each generation prints:

* Model/backend setup (model name, device, EOS token id)
* Prefill call with sequence length
* Every decode step with the token id and decoded text snippet
* Streamed chunks as they are sent to the client

Use these logs to follow the complete prefill → decode → stream lifecycle step-by-step.

Set `MODEL_NAME` env var to load a different HuggingFace causal LM (default: `Qwen/Qwen2.5-0.5B-Instruct`).

### Example benchmark result (Apple M1 Pro, 32GB RAM, Python 3.12.8, MPS)

Default benchmark suite (3 chat turns, max_new_tokens=512 when no prompt is given) on `Qwen/Qwen2.5-0.5B-Instruct`:

```
Streaming chat summary: throughput=37.16 tok/s, duration=10.387s
Baseline HF streaming chat summary: throughput=17.69 tok/s, duration=21.825s
Comparison: streaming faster by 11.438s with x2.10 throughput
```

Why sglang-style streaming is faster here:
- Prefill + decode loop runs in-process with explicit KV reuse and minimal Python overhead between steps.
- HF streaming baseline re-enters the generator loop and Python callback machinery per token (TextIteratorStreamer), adding per-token overhead even though it also uses KV cache.
- Both use the same model and device (Apple M1 Pro via MPS); differences are from orchestration, not model quality.
