# Configuration

mini-sglang uses environment variables for most runtime toggles. Defaults are chosen to be safe
and CPU-friendly.

## Model

- `MODEL_NAME`: Hugging Face model id (default: `rd211/Qwen3-0.6B-Instruct`)
- `MODEL_LOCAL_DIR`: optional local path override (avoids hub downloads)
- `TRUST_REMOTE_CODE`: set to `1` to pass `trust_remote_code=True` to transformers loaders

## Backend selection

- `BACKEND_IMPL`: `auto|torch|sgl|mlx|hf` (default: `auto`)
- `TP_SIZE`, `DP_SIZE`: sglang global server args (only used when sglang runtime is present)

## Engine / streaming

- `MAX_NEW_TOKENS_DEFAULT`: code constant in `config.py` (change in code if needed)
- `FAST_STREAM_DECODE`: `1` streams per-token deltas and decodes full text at the end
- `DECODE_LOG_STRIDE`: log every N decode steps (default: 256)

## Multi-device and scheduling (`multi_device.py`)

- `ENABLE_MULTI_DEVICE`: `1` enables multi-GPU pool when multiple CUDA GPUs exist
- `SCHEDULER_MODE`: `rr|fsfs|random|cache_aware`
- `MAX_INFLIGHT_TOTAL`: global inflight cap (0 disables)
- `MAX_INFLIGHT_PER_ENGINE`: per-engine inflight cap (0 disables)
- `ADAPTIVE_MAX_NEW_TOKENS`: `1` enables load-based downscale
- `ADAPTIVE_MAX_INFLIGHT_THRESHOLD`: inflight threshold to trigger downscale
- `ADAPTIVE_MAX_NEW_TOKENS_FACTOR`: scale factor (default: 0.8)
- `ASYNC_PREFILL_QUEUE`: `1` enables background prefix warming queue
- `METRIC_WINDOW`: rolling window size for latency/throughput averages

## Cache knobs (HF/torch + sgl backends)

- `TOKEN_CACHE_SIZE`: tokenizer decode cache size (default: 32; HF backend)
- `PREFILL_CACHE_SIZE`: exact-match prefill cache size (default: 4)
- `PREFILL_CACHE_TOKEN_BUDGET`: token budget for prefill cache (default: 32768)
- `PREFIX_CACHE`: `1` enables prefix cache (default: 1)
- `PREFIX_CACHE_SIZE`: prefix cache size (default: 8)
- `PREFIX_CACHE_MAX_TOKENS`: maximum prompt length to insert (default: 2048)
- `PREFIX_CACHE_TOKEN_BUDGET`: token budget for prefix cache (default: 32768)

## MLX backend knobs

- `MLX_CACHE_DIR`: cache dir (default: `~/.cache/mini_sglang/mlx`)
- `MLX_TRUST_REMOTE_CODE`: `1` enables MLX remote code trust
- `MLX_PREFILL_STEP_SIZE`: chunk size for long-prefill (default: 2048)
- `MLX_MAX_KV_SIZE`: hard KV cap (0 disables)
- `AGGRESSIVE_MAX_NEW_TOKENS`: MLX-only; clamps output more aggressively under context limits

## Server

- `SERVER_STREAM_LOG_STRIDE`: log every N streamed chunks
- `WARM_PREFIXES`: `prompt1||prompt2||...` seeds prefix cache on startup
- `ZMQ_CONTROL`: `1` enables optional ZMQ control server
- `ZMQ_CONTROL_ENDPOINT`: default `tcp://127.0.0.1:5557`

## VLM (chat with images)

Used by `POST /v1/chat/completions` when messages include `image_url` segments.

- `VLM_BACKEND`: `auto|hf|mlx` (default: `auto`, prefers MLX on MPS when available)
- `VLM_MODEL_NAME`: model id for VLM requests (defaults to `MODEL_NAME` when unset)
- `VLM_IMAGE_TOKEN`: placeholder inserted into the prompt for each image (default: `<image>`)
- `VLM_IMAGE_HTTP_TIMEOUT_S`: timeout for `http(s)` image fetch (default: 5)
- `VLM_MAX_IMAGE_BYTES`: size cap for downloaded/decoded images (default: 10485760)
- `ALLOW_FILE_IMAGE_URL`: `1` enables `file://` and local path image URLs (default: 0)
- `VLM_STREAM_LOG_STRIDE`: VLM streaming log stride (default: 32)

Note: enabling `ALLOW_FILE_IMAGE_URL=1` allows remote callers to request local files via the API;
only use it in trusted environments.

MLX VLM backend notes:

- Requires Apple Silicon and MLX installed (`pip install mlx`).
- Requires a compatible MLX VLM package (commonly exposed as `mlx_vlm`) and an MLX-format VLM model.

## Image generation

Used by `POST /v1/images/generations`.

- `IMAGE_MODEL_ID`: default model id when `backend=diffusers` and request `model` is unset

## Benchmarks (`bench_suite.py`)

- `BENCH_REPEAT`, `BENCH_WARMUP`, `BENCH_CONCURRENCY`
- `BENCH_MIXED_PROMPTS`: `1` injects random suffixes into half the prompts
- `BENCH_HF_CONCURRENCY`: baseline concurrency (default: 1)
- `BENCH_LOG_LEVEL`: logging level (default: `INFO`)

## Runtime

- `MATMUL_PRECISION`: forwarded to `torch.set_float32_matmul_precision(...)` when set
- `ENABLE_SDP`: `1` enables torch SDPA when available (default: 1)

## Z-Image helper (`z_image_mps.py`)

- `Z_IMAGE_MODEL_ID`: default `Tongyi-MAI/Z-Image-Turbo`
- `Z_IMAGE_MODEL_DIR`: local model dir (falls back to `MODEL_LOCAL_DIR`)
- `Z_IMAGE_CACHE_DIR`: default `~/.cache/mini-sglang`

These env vars are also used by `POST /v1/images/generations` (diffusion image generation).

## Video generation

Used by `POST /v1/videos/generations`.

- `VIDEO_MODEL_ID`: default model id for video generation when request `model` is unset

Encoding notes:

- `response_format=frames_b64_png` needs only Pillow (frames are returned as base64 PNGs).
- `response_format=b64_mp4` requires `imageio` + `numpy` to be installed.
