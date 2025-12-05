# Agent Guidelines

1) Code must be absolutely elegant with crystal-clear ownership: each function/module does one thing, avoids duplication, and has no hidden side effects.
2) CI parity locally: install git hooks via `git config core.hooksPath .githooks` so ruff + mypy run on every commit; commits that fail lint/type checks must not be made.
3) Prioritize tackling the core, challenging performance paths (such as CUDA Graph, flash-attn, native scheduling/KV management, etc.) instead of masking the real issues with simple, roundabout “quick” workarounds.

# Repository Guidelines

## Project Structure & Module Organization
- `backend/` holds model + tokenizer setup, KV/prefix caches, and HF streaming compatibility; backends live under `backend/hf`, `backend/sglang`, and `backend/mlx`, with shared cache helpers in `backend/cache.py`.
- `engine/engine.py` orchestrates prefill/decode, scheduling, and backpressure; `multi_device.py` spreads work across GPUs.
- `api/server.py` is the FastAPI streaming surface; `ipc/zmq_control.py` exposes optional metrics/warm commands.
- Root scripts: `cli_demo.py`, `smoke_test.py`, `local_bench.py`, `server_bench.py`, `config.py`.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt` (or `uv sync`).
- Quick sanity: `python smoke_test.py --max-new-tokens 32 "Hello mini-sglang"` exercises prefill+decode+streaming.
- Benchmarks: `python local_bench.py --max-new-tokens 128 "prompt"` and `python server_bench.py --max-new-tokens 128`.
- FastAPI server: `uvicorn api.server:app --reload --port 8000`; stream via curl: `curl -N -X POST -H "Content-Type: application/json" -d '{"prompt":"hi","stream":true}' http://localhost:8000/generate`.
- Lint/type (CI parity): `ruff check .` and `mypy --ignore-missing-imports --install-types --non-interactive .`. Install hooks once with `git config core.hooksPath .githooks`.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indent, line length 100 (see `pyproject.toml`). Type hints required; keep functions small with explicit ownership.
- Use `logging` over prints; prefer pure helpers in `backend/` and `engine/`. Snake_case for functions/vars, PascalCase for classes, UPPER_SNAKE for env flags.
- Keep performance switches in `config.py`; avoid duplicating scheduler/cache logic.

## Testing Guidelines
- No formal unit suite yet; rely on `smoke_test.py` for e2e coverage and the benchmarks for perf regressions. Add targeted tests or reproducible scripts when touching cache, scheduler, or control-plane code.
- When adding features, include a reproduction command (e.g., `python local_bench.py ...`) and verify multi-device paths when relevant (`ENABLE_MULTI_DEVICE=1`).

## Commit & Pull Request Guidelines
- Commit messages are short and imperative (e.g., `Improve local bench defaults`). Group mechanical changes separately when possible.
- Before PRs, ensure `ruff` and `mypy` pass and include key logs/metrics: throughput/TTFB deltas for engine/cache/backpressure changes, and curl/CLI traces for API changes. Link issues and note config/env flags touched.
- PR descriptions should call out compatibility assumptions (device, model size, flash-attn availability) and any new env vars or defaults affecting scripts.

## Configuration & Safety Notes
- Model selection defaults to `MODEL_NAME=rd211/Qwen3-0.6B-Instruct`; override via `MODEL_LOCAL_DIR` to avoid hub downloads. Device is chosen in `config.get_device()` (MPS → CUDA → CPU).
- Key toggles: `SCHEDULER_MODE`, `MAX_INFLIGHT_TOTAL`, `PAGE_TOKEN_BUDGET`, `ENABLE_FLASH_ATTENTION`, `COMPILE_MODEL`, `ATTN_IMPL`, `ENABLE_MULTI_DEVICE`. Document new flags in `README.md` and keep fallbacks safe for CPU-only runs.
