# Architecture

mini-sglang is a small, streaming-first text generation stack inspired by sglang. The design goal
is to keep the control flow readable while still making performance levers (cache reuse, scheduler,
backend selection) explicit.

## Layering

```
Client (curl / bench / demo)
  └─ api/server.py (FastAPI, streaming)
      └─ multi_device.EnginePool (routing + backpressure)
          └─ engine.SGLangMiniEngine (prefill + decode + streaming callbacks)
              └─ backend.* (model + tokenizer + KV/cache implementations)
```

### API (`api/`)

- `POST /generate`: JSON request, streams JSON lines (`{"text_delta": ...}`).
- `POST /v1/chat/completions`: OpenAI-compatible SSE-like stream (supports VLM via `image_url`).
- `POST /v1/images/generations`: OpenAI-compatible image generation (Z-Image or generic diffusers).
- `POST /v1/videos/generations`: OpenAI-compatible video generation (diffusers; returns frames or mp4).
- `GET /metrics`: lightweight JSON metrics (scheduler, inflight, cache hit/miss).

The API layer is intentionally thin: it validates inputs, picks an engine from the pool, and
streams deltas produced by the engine. VLM and diffusion requests use dedicated in-process runners
to keep the text-generation engine path minimal.

### Pool & scheduling (`multi_device.py`)

`EnginePool` manages one engine per device and chooses an engine per request.

Key concerns:

- **Backpressure**: `MAX_INFLIGHT_TOTAL` / `MAX_INFLIGHT_PER_ENGINE`
- **Scheduling**: `SCHEDULER_MODE=rr|fsfs|random|cache_aware`
- **Adaptive budgets**: optional downscale of `max_new_tokens` under load
- **Async prefill**: optional background prefix warming queue (`ASYNC_PREFILL_QUEUE`)

### Engine (`engine/engine.py`)

`SGLangMiniEngine.run_generate()` is the core control flow:

1. Tokenize the prompt (or accept pre-tokenized ids from the pool).
2. Prefill once to build KV state and emit the first token.
3. Decode token-by-token, emitting deltas via the stream callback.

The engine does not own model weights; it only orchestrates backend calls and streaming semantics.

### Backends (`backend/`)

Backends hide model-specific details while exposing a common interface for:

- tokenization and decoding
- `prefill_forward(prompt_ids) -> (first_token_id, kv_cache)`
- `decode_forward(last_token_id, kv_cache) -> (next_token_id, kv_cache)`
- cache operations and metrics

Implemented backends:

- `backend/hf`: torch/transformers backend (CPU/MPS/CUDA fallback).
- `backend/sglang`: CUDA sgl_kernel-backed Qwen path (best-effort; falls back when unavailable).
- `backend/mlx`: Apple Silicon MLX backend (when installed).
- `backend/diffusion`: diffusion pipelines used by the images endpoint (currently Z-Image).

### Caches (`backend/cache.py`)

Two cache types are used by backends:

- **Prefill cache**: exact prompt match → first token + KV state (LRU with token budget).
- **Prefix cache**: longest-prefix match via a radix trie + LRU/LFU accounting (token budget).

Backends may additionally use a page-like KV manager to evict KV blobs by token budget.

## Extending the system

- Add a backend: implement the backend interface and wire it in `backend/factory.py`.
- Add a scheduler: extend `EnginePool.pick()` (keep backpressure invariants intact).
