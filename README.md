# mini-sglang

A minimal, streaming-first implementation of an sglang-style text generation stack with clear API / Engine / Backend separation. Benchmarks now compare real `sglang`, this mini stack, and a plain Hugging Face streaming baseline in both local and HTTP server modes. For the full upstream project and docs, see [sglang on GitHub](https://github.com/sgl-project/sglang).

### Quickstart

Install dependencies from `requirements.txt` or the committed `pyproject.toml` and run the two benchmark scripts:

```bash
# Local mode: sglang Runtime vs mini-sglang vs HF streaming
python local_bench.py --max-new-tokens 128 "Hello mini-sglang"

# Server mode: sglang HTTP endpoint vs the FastAPI demo (sglang + HF modes)
python server_bench.py --max-new-tokens 128 "Hello mini-sglang"
```

Both scripts default to the model in `MODEL_NAME` and allow overrides via `--model`. The server benchmark will start a local `sglang.Runtime` if you do not supply `--sglang-url`.

### Performance levers (enabled by default)

- **Prefix caching**: reuse prefill KV for repeated prompts (`PREFIX_CACHE=1`, `PREFIX_CACHE_SIZE=16`).
- **Longest-prefix matching**: prefix cache picks the deepest matching prefix with a radix-style lookup (O(prefix length)) instead of scanning the cache.
- **Prefix cache controls**: skip caching overly long prompts (`PREFIX_CACHE_MAX_TOKENS`, default 4096), cap total cached tokens (`PREFIX_CACHE_TOKEN_BUDGET`, default 65536), optionally switch eviction policy (`PREFIX_CACHE_POLICY=lru|lfu`, default lru), and allow manual pre-insertion via `ModelBackend.insert_prefix(...)`. Prefix entries can spill into a KV page manager when `PAGE_TOKEN_BUDGET>0`.
- **Prefill cache controls**: LRU by entry count (`PREFILL_CACHE_SIZE`, default 8) plus an optional total token budget (`PREFILL_CACHE_TOKEN_BUDGET`, default 65536).
- **Scheduler modes**: pick generation engine via round robin (`SCHEDULER_MODE=rr`, default), lowest-load-first (`fsfs`), random (`random`), or cache-aware (`cache_aware`) which selects the engine with the deepest prefix cache hit and breaks ties by lower load.
- **Backpressure limits**: cap concurrency with `MAX_INFLIGHT_TOTAL` (all engines) and `MAX_INFLIGHT_PER_ENGINE` (per device). Requests block until capacity frees.
- **Prefix warmup**: optionally prefill and cache common prompts at server startup via `WARM_PREFIXES="prompt1||prompt2"`.
- **Async prefill queue (opt-in)**: set `ASYNC_PREFILL_QUEUE=1` to run background prefix inserts so prompts start warming while decode threads stream.
- **Cache stats**: per-request logs include prefill/prefix cache hit/miss counters for quick observability.
- **Load dtype control**: optionally set `MODEL_DTYPE`/`TORCH_DTYPE` to `fp16`/`bf16`/`fp32` (or `auto` to pick float16 on CUDA/MPS) when loading the model.
- **CUDA graphs for decode**: optionally capture the 1-token decode step (`ENABLE_DECODE_CUDA_GRAPH=1`, default) to reduce Python overhead on CUDA.
- **Attention implementation override**: set `ATTN_IMPL`/`ATTN_IMPLEMENTATION` to `flash_attention_2`, `sdpa`, or `eager` to force a specific attention kernel when supported by the model.
- **Metrics endpoint**: GET `/metrics` returns scheduler mode, inflight counts, and cache hit/miss stats (prefix + prefill).
- **Adaptive max_new_tokens under load**: when `ADAPTIVE_MAX_NEW_TOKENS=1`, downscale `max_new_tokens` if inflight requests exceed a threshold (`ADAPTIVE_MAX_INFLIGHT_THRESHOLD`, default pool size; `ADAPTIVE_MAX_NEW_TOKENS_FACTOR`, default 0.8).
- **KV page manager (opt-in)**: set `PAGE_TOKEN_BUDGET` (and optional `PAGE_SIZE_TOKENS`) to store prefix cache KV in a paged store with token-budget eviction.
- **Dynamic cache guard**: CUDA graph capture auto-disables when a model uses DynamicCache; set `FORCE_LEGACY_CACHE=1` to ask HF to use the legacy cache and allow graphs on supported models.
- **Fast decode mode**: set `ENGINE_FAST_DECODE=1` to use the HF streaming/generate path inside mini-sglang, eliminating Python per-token overhead for maximum throughput (defaults on for non-CUDA devices).
- **Tensor parallel loading**: shard the model across available CUDA devices automatically (`TENSOR_PARALLEL_SIZE` defaults to GPU count).
- **Torch compilation**: wrap the model with `torch.compile` unless disabled (`COMPILE_MODEL=0` to opt out, optional `COMPILE_MODE`).
- **CUDA graphs for prefill**: capture and replay prefill shapes on CUDA by default (`ENABLE_CUDA_GRAPH=1`, `CUDA_GRAPH_MAX_SEQ_LEN=512`).
- **Chunked prefill (opt-in)**: break long prompts into smaller prefill chunks to lower peak memory and improve graph reuse on very long prompts (`CHUNKED_PREFILL=1`, `PREFILL_CHUNK_SIZE=512`).
- **Aggressive max_new_tokens capping**: automatically cap `max_new_tokens` to the model context window minus a small margin (`AGGRESSIVE_MAX_NEW_TOKENS=1`, `MAX_CONTEXT_MARGIN=16`).

## Project layout
```
requirements.txt
config.py              # (sglang global config analogue) model choice + device selection
backend/model_backend.py  # (sglang backend analogue) model + tokenizer + KV cache helpers
backend/cache.py         # cache utilities (prefill/prefix LRU + radix index)
engine/engine.py       # (sglang engine analogue) prefill + decode orchestration
api/server.py          # (sglang API analogue) FastAPI server exposing POST /generate
cli_demo.py            # Minimal client to stream tokens in a terminal
local_bench.py         # Local benchmark: sglang Runtime vs mini-sglang vs HF streaming
server_bench.py        # HTTP benchmark: sglang server vs mini-sglang FastAPI (sglang + HF modes)
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

Run the local benchmark comparing sglang Runtime, mini-sglang, and the HF streaming baseline:
```bash
python local_bench.py --max-new-tokens 64 "Benchmarking mini-sglang"
```
The benchmark prints token counts, wall-clock duration, and throughput for all three so you can evaluate orchestration overhead.

Logs at INFO level narrate every prefill/decode stream chunk so beginners can follow the full flow. Summaries include token counts and throughput for each path.

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

The HTTP API is streaming-only; for non-sglang baselines use `local_bench.py` or `server_bench.py`.

Benchmark the server path (TTFB + throughput):
```bash
python server_bench.py --max-new-tokens 128
```
The server benchmark prints three views side-by-side so you can see how the streaming FastAPI path compares to the HF streaming baseline **and** a real `sglang` server (started automatically when no URL is supplied).

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

Set `MODEL_NAME` env var to load a different HuggingFace causal LM (default: `Qwen/Qwen2.5-0.5B-Instruct`). If huggingface.co
is unreachable, the loader will auto-fallback to the same repo on 魔搭 ModelScope (override with `MODELSCOPE_MODEL_NAME` or use
`MODEL_LOCAL_DIR` to point at a pre-downloaded path).

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
