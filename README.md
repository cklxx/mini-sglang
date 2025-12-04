# mini-sglang

A minimal, streaming-first implementation of an sglang-style text generation stack with clear API / Engine / Backend separation. Benchmarks compare this mini stack to a plain Hugging Face streaming baseline. For the full upstream project and docs, see [sglang on GitHub](https://github.com/sgl-project/sglang).

### Quickstart

Install dependencies from `requirements.txt` or the committed `pyproject.toml` and run the concurrent benchmark:

```bash
# Concurrent benchmark (short/medium/long prompts; defaults need no flags)
python bench_suite.py
```

Defaults use the model in `MODEL_NAME`. Tune repeat/warmup/concurrency via env vars if needed.

### Performance levers (enabled by default unless noted)

- **Caches**: prefix/prefill KV caches with radix longest-prefix lookup, LRU/LFU eviction, token budgets, page spill (`PAGE_TOKEN_BUDGET`), and manual seeding via `insert_prefix`. Chunked prefill is optional (`CHUNKED_PREFILL=1`); static KV (`ENABLE_STATIC_KV=1`) defaults on for CUDA using `StaticCache` with auto-fallback.
- **Scheduling & backpressure**: round robin / fsfs / random / cache-aware dispatch (`SCHEDULER_MODE`), inflight caps (`MAX_INFLIGHT_TOTAL`, `MAX_INFLIGHT_PER_ENGINE`), adaptive `max_new_tokens` downscale under load, async prefix warmup (`WARM_PREFIXES`, `ASYNC_PREFILL_QUEUE`).
- **Fast paths**: chunked prefill optional (`CHUNKED_PREFILL=1`). Experimental CUDA graph capture on CUDA: enabled by default (`ENABLE_CUDA_GRAPH=1`), captures prefill for the first prompt length (or explicit `PREFILL_GRAPH_SEQ_LEN`) up to `PREFILL_GRAPH_MAX_LEN` (default 2048), and auto-falls-back on errors or mismatched lengths. Goal: extend to decode/flash-attn without HF shortcuts.
- **Model loading**: tensor parallel (`TENSOR_PARALLEL_SIZE`), dtype override (`MODEL_DTYPE`/`TORCH_DTYPE`), attention impl (`ATTN_IMPL`, 默认CUDA上启用 `flash_attention_2` 可通过 `ENABLE_FLASH_ATTENTION` 控制), torch.compile (`COMPILE_MODEL`/`COMPILE_MODE`), aggressive context-safe `max_new_tokens` capping.
- **Observability & control**: GET `/metrics` for scheduler/inflight/cache stats and rolling latency/throughput; per-request logs include cache hit/miss counters. Optional ZMQ control channel (`ZMQ_CONTROL=1`, `ZMQ_CONTROL_ENDPOINT`) supports metrics/warm commands.

## Project layout
```
requirements.txt
config.py              # (sglang global config analogue) model choice + device selection
backend/model_backend.py  # (sglang backend analogue) model + tokenizer + KV cache helpers
backend/cache.py         # cache utilities (prefill/prefix LRU + radix index)
engine/engine.py       # (sglang engine analogue) prefill + decode orchestration
api/server.py          # (sglang API analogue) FastAPI server exposing POST /generate
cli_demo.py            # Minimal client to stream tokens in a terminal
bench_suite.py         # Concurrent benchmark: mini-sglang vs HF streaming (single entrypoint)
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

MPS MLX backend (optional):
- Install `mlx` + `mlx-lm` (`pip install mlx mlx-lm` on Apple Silicon).
- Set `BACKEND_IMPL=mlx` (auto-selected on MPS when MLX is available).

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

Run the concurrent benchmark comparing mini-sglang and the HF streaming baseline across short/medium/long prompts with repeats:
```bash
python bench_suite.py
```
Defaults: repeat=3, warmup=1, concurrency=4, mixed prompts on. Tune via env `BENCH_REPEAT`/`BENCH_WARMUP`/`BENCH_CONCURRENCY`/`BENCH_MIXED_PROMPTS` as needed.

Logs at INFO level narrate every prefill/decode stream chunk so beginners can follow the full flow. Summaries include token counts and throughput for each path.
The bench suite defaults to a longer workload (long case uses `max_new_tokens=1024`, `repeat=3`)
so GPU runs have enough work to outweigh startup overhead. Tweak the `BENCH_*` env vars to
shorten quick tests.

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

The HTTP API is streaming-only.

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

Set `MODEL_NAME` env var to load a different HuggingFace causal LM (default: `rd211/Qwen3-0.6B-Instruct`). If you previously pulled another Qwen checkpoint, remove the cached folder so the new default can be re-downloaded cleanly. Use `MODEL_LOCAL_DIR` to point at a pre-downloaded path if you want to avoid hub downloads.

### Example benchmark result (Apple M1 Pro, 32GB RAM, Python 3.12.8, MPS)

Default benchmark suite (longer decode: max_new_tokens=1024, repeat=3 runs) on `rd211/Qwen3-0.6B-Instruct`:

```
Streaming chat summary: throughput=37.16 tok/s, duration=10.387s
Baseline HF streaming chat summary: throughput=17.69 tok/s, duration=21.825s
Comparison: streaming faster by 11.438s with x2.10 throughput
```

Why sglang-style streaming is faster here:
- Prefill + decode loop runs in-process with explicit KV reuse and minimal Python overhead between steps.
- HF streaming baseline re-enters the generator loop and Python callback machinery per token (TextIteratorStreamer), adding per-token overhead even though it also uses KV cache.
- Both use the same model and device (Apple M1 Pro via MPS); differences are from orchestration, not model quality.
