# Contributing

Thanks for your interest in improving **mini-sglang**.

This repo is intentionally small and "readable-first": prefer changes that improve clarity and
correctness of the streaming + cache + scheduling path over feature breadth.

## Development setup

- Python: **3.10+** (recommended: 3.11)
- Install dependencies (pick one):
  - `uv sync` (recommended; uses `pyproject.toml` + `uv.lock`)
  - `pip install -r requirements.txt`

## Quick validation

- End-to-end smoke (prefill + decode + streaming):
  - `python3 smoke_test.py --max-new-tokens 32 "Hello from mini-sglang"`
- Concurrent benchmark (mini-sglang vs HF baseline):
  - `python3 bench_suite.py`
- Run server:
  - `uvicorn api.server:app --reload --port 8000`

## Lint & types (CI parity)

- Run:
  - `ruff check .`
  - `mypy --ignore-missing-imports --install-types --non-interactive .`
- Optional: install hooks once:
  - `git config core.hooksPath .githooks`

## What to include in a PR

- A short, scoped change (avoid drive-by refactors).
- A repro command (one of: `python3 smoke_test.py ...`, `python3 bench_suite.py`, `uvicorn ...`).
- If you touch cache/scheduler/streaming code, include:
  - before/after latency or throughput numbers (TTFB, tok/s) and the environment used.

## Code style

- Keep functions small and explicit; avoid hidden global state.
- Prefer `logging` over `print`.
- Type hints required for new/modified APIs.
- Line length: 100 (`pyproject.toml`).

## Design principles

- **Streaming-first**: avoid buffering entire outputs when a delta is available.
- **Separation of concerns**: API ↔ engine ↔ backend boundaries should stay sharp.
- **Safe defaults**: CPU-only runs should remain functional without extra flags.
