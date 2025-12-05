# mini-sglang

A minimal, streaming-first implementation of an sglang-style text generation stack with clear API / Engine / Backend separation. Benchmarks compare this mini stack to a plain Hugging Face streaming baseline. For the full upstream project and docs, see [sglang on GitHub](https://github.com/sgl-project/sglang).

### Quickstart

Install dependencies from `requirements.txt` or the committed `pyproject.toml` and run the concurrent benchmark:

```bash
# Concurrent benchmark (short/medium/long prompts; defaults need no flags)
python bench_suite.py
```

Defaults use the model in `MODEL_NAME`. Tune repeat/warmup/concurrency via env vars if needed. Pick backend via `BACKEND_IMPL=torch|hf|sgl|mlx` (auto-selects sgl on CUDA, mlx on MPS, torch otherwise).

### Performance levers (enabled by default unless noted)

- **Caches**: prefix/prefill KV caches with radix longest-prefix lookup plus LRU eviction and token budgets; manual seeding via `insert_prefix`.
- **Scheduling & backpressure**: round robin / fsfs / random / cache-aware dispatch (`SCHEDULER_MODE`), inflight caps (`MAX_INFLIGHT_TOTAL`, `MAX_INFLIGHT_PER_ENGINE`), adaptive `max_new_tokens` downscale under load, async prefix warmup (`WARM_PREFIXES`, `ASYNC_PREFILL_QUEUE`).
- **Backends**: HF torch backend for CPU/MPS, sgl_kernel Qwen3 backend on CUDA, MLX backend on MPS, and an HF baseline mode for plain streaming; pick via `BACKEND_IMPL` or device.
- **Observability & control**: GET `/metrics` for scheduler/inflight/cache stats and rolling latency/throughput; per-request logs include cache hit/miss counters. Optional ZMQ control channel (`ZMQ_CONTROL=1`, `ZMQ_CONTROL_ENDPOINT`) supports metrics/warm commands.

## Project layout
```
requirements.txt
config.py              # (sglang global config analogue) model choice + device selection
backend/hf/backend.py   # CPU/MPS HF backend with prefix/prefill cache
backend/sglang/backend.py  # CUDA sgl_kernel backend using Qwen3Model
backend/mlx/backend.py  # MLX backend for MPS
backend/cache.py        # cache utilities (prefill/prefix LRU + radix index)
engine/engine.py        # prefill + decode orchestration
api/server.py           # FastAPI server exposing POST /generate
cli_demo.py            # Minimal client to stream tokens in a terminal
bench_suite.py         # Concurrent benchmark (single entrypoint)
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

Flash-attn install note: the wheel often builds from source on Py3.12 and needs torch visible
to the build backend. If you hit `ModuleNotFoundError: No module named 'torch'` while building
flash-attn, use:
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel ninja packaging
pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
CUDA_HOME=/usr/local/cuda MAX_JOBS=$(nproc) \
  pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir
pip install -r requirements.txt --no-deps --no-build-isolation
PIP_NO_BUILD_ISOLATION=1 pip install -r requirements.txt  # installs transformers/fastapi/uvicorn/etc.
pip install numpy  # optional, removes torch numpy warning
```
For `uv`, add to `pyproject.toml`:
```
[tool.uv.extra-build-dependencies]
flash-attn = ["torch"]
```
then install torch in the venv first and run `UV_BUILD_ISOLATION=0 uv sync`.

sgl_kernel (CUDA-only, arch-specific): upstream wheels now work directly on SM89+ GPUs (Ada/Hopper/
Blackwell). Install from PyPI alongside torch; this repo no longer vendors the source or ships helper
scripts. For older architectures without a matching wheel, follow upstream sgl_kernel build docs to
compile from source. If the backend falls back to HF/torch, logs include the import/CUDA reason.

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

### Z-Image image generation (MPS/CUDA/CPU)

`z_image_mps.py` mirrors the [z-image-mps](https://github.com/ivanfioravanti/z-image-mps) CLI so you can
generate images with `Tongyi-MAI/Z-Image-Turbo` locally. The script auto-downloads the checkpoint to
`~/.cache/mini-sglang/z-image-turbo` (override with `Z_IMAGE_MODEL_DIR`/`--model-dir`) and picks MPS ->
CUDA -> CPU automatically.

```
# Quick run (defaults to the Hanfu prompt)
python z_image_mps.py

# Custom prompt and aspect ratio
python z_image_mps.py -p "Cyberpunk night market, neon haze" --aspect 16:9

# Deterministic seeds + multiple images + FlashAttention2 on CUDA
python z_image_mps.py -p "Nordic fjord at dawn" --num-images 2 --seed 123 \
  --attention-backend flash2
```

Useful flags: `--device` to force mps/cuda/cpu, `--steps`/`--guidance-scale`/`--height`/`--width`,
`--compile` (torch.compile on the DiT transformer), `--cpu-offload` (CUDA only), `--model` to point
at a different repo, and `--model-dir` to reuse a local checkpoint.

### Learning-friendly logs

The CLI, smoke test, and FastAPI server enable INFO-level logging by default. Each generation prints:

* Model/backend setup (model name, device, EOS token id)
* Prefill call with sequence length
* Every decode step with the token id and decoded text snippet
* Streamed chunks as they are sent to the client

Use these logs to follow the complete prefill -> decode -> stream lifecycle step-by-step.

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
