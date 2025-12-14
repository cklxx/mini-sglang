# Development

## Setup

- Python 3.10+
- Recommended:
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `uv sync`

## Common commands

- Smoke test (e2e streaming):
  - `python3 smoke_test.py --max-new-tokens 32 "Hello mini-sglang"`
- Benchmark suite:
  - `python3 bench_suite.py`
- Server:
  - `uvicorn api.server:app --reload --port 8000`
- Lint/type:
  - `ruff check .`
  - `mypy --ignore-missing-imports --install-types --non-interactive .`

## API examples

- Image generation (Z-Image):
  - `curl -sS -X POST -H "Content-Type: application/json" -d '{"prompt":"A cat","backend":"z_image","n":1,"response_format":"b64_json"}' http://localhost:8000/v1/images/generations`
- Video generation (diffusers; requires `VIDEO_MODEL_ID` or request `model`):
  - `curl -sS -X POST -H "Content-Type: application/json" -d '{"prompt":"A scenic drone shot","n":1,"response_format":"frames_b64_png","num_frames":14,"fps":8}' http://localhost:8000/v1/videos/generations`

## Repo hooks

Install local hooks (runs ruff + mypy on commit):

`git config core.hooksPath .githooks`

## Backend notes

- `BACKEND_IMPL=auto` chooses MLX on MPS (when installed), sgl_kernel on CUDA, torch otherwise.
- The CUDA sgl_kernel backend is best-effort; failures fall back to the HF/torch backend with a log
  message explaining why.
