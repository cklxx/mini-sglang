# Repository Guidelines

## Project Structure & Module Organization
- `api/` is the FastAPI streaming surface (`api/server.py`).
- `backend/` holds model + tokenizer setup, KV/prefix caches, and backend implementations:
  - `backend/hf/`: torch/transformers backend (CPU/MPS/CUDA fallback) + HF streaming baseline.
  - `backend/sglang/`: CUDA sgl_kernel-backed backend (best-effort, falls back when unavailable).
  - `backend/mlx/`: MLX backend for Apple Silicon.
  - `backend/diffusion/`: image/video generation backends (diffusers; Z-Image, generic pipelines).
  - Shared cache helpers live in `backend/cache.py`.
- `engine/engine.py` orchestrates prefill/decode and streaming callbacks.
- `multi_device.py` is the engine pool: scheduling, backpressure, multi-GPU routing, metrics.
- `ipc/zmq_control.py` exposes an optional ZMQ control plane for metrics/warm commands.
- `docs/` contains architecture/config/dev/benchmark documentation.
- Root scripts: `cli_demo.py`, `smoke_test.py`, `bench_suite.py`, `config.py`, `z_image_mps.py`.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt` (or `uv sync`).
- Quick sanity: `python3 smoke_test.py --max-new-tokens 32 "Hello mini-sglang"` exercises prefill+decode+streaming.
- Benchmarks: `python3 bench_suite.py` (concurrent mini-sglang vs HF baseline).
- FastAPI server: `uvicorn api.server:app --reload --port 8000`; stream via curl: `curl -N -X POST -H "Content-Type: application/json" -d '{"prompt":"hi","stream":true}' http://localhost:8000/generate`.
- Lint/type (CI parity): `ruff check .` and `mypy --ignore-missing-imports --install-types --non-interactive .`. Install hooks once with `git config core.hooksPath .githooks`.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indent, line length 100 (see `pyproject.toml`). Type hints required; keep functions small with explicit ownership.
- Use `logging` over prints; prefer pure helpers in `backend/` and `engine/`. Snake_case for functions/vars, PascalCase for classes, UPPER_SNAKE for env flags.
- Keep performance switches in `config.py`; avoid duplicating scheduler/cache logic.

## Testing Guidelines
- No formal unit suite yet; rely on `smoke_test.py` for e2e coverage and the benchmarks for perf regressions. Add targeted tests or reproducible scripts when touching cache, scheduler, or control-plane code.
- When adding features, include a reproduction command (e.g., `python3 bench_suite.py`) and verify multi-device paths when relevant (`ENABLE_MULTI_DEVICE=1`).

## Commit & Pull Request Guidelines
- Commit messages are short and imperative (e.g., `Improve local bench defaults`). Group mechanical changes separately when possible.
- Before PRs, ensure `ruff` and `mypy` pass and include key logs/metrics: throughput/TTFB deltas for engine/cache/backpressure changes, and curl/CLI traces for API changes. Link issues and note config/env flags touched.
- PR descriptions should call out compatibility assumptions (device, model size, flash-attn availability) and any new env vars or defaults affecting scripts.

## Configuration & Safety Notes
- Model selection defaults to `MODEL_NAME=rd211/Qwen3-0.6B-Instruct`; override via `MODEL_LOCAL_DIR` to avoid hub downloads. Device is chosen in `config.get_device()` (MPS → CUDA → CPU).
- Key toggles: `BACKEND_IMPL`, `SCHEDULER_MODE`, `MAX_INFLIGHT_TOTAL`, `MAX_INFLIGHT_PER_ENGINE`, `ENABLE_MULTI_DEVICE`, `PREFILL_CACHE_*`, `PREFIX_CACHE_*`, `WARM_PREFIXES`, `ASYNC_PREFILL_QUEUE`, `VLM_BACKEND`. Keep fallbacks safe for CPU-only runs and document new flags in `docs/CONFIGURATION.md`.
