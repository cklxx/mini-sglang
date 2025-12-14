# Benchmarks

The main entrypoint is `bench_suite.py`, which runs a concurrent workload against:

- mini-sglang (engine + backend)
- HF streaming baseline (TextIteratorStreamer)

## Run

`python3 bench_suite.py`

Common env vars:

- `BENCH_REPEAT=3`
- `BENCH_WARMUP=1`
- `BENCH_CONCURRENCY=4`
- `BENCH_MIXED_PROMPTS=1`
- `BENCH_LOG_LEVEL=INFO`

## Interpreting results

The suite prints per-workload summaries:

- `p50_ttfb` / `p95_ttfb`: time-to-first-byte (seconds)
- `tokens`: total tokens produced across runs
- `throughput`: tokens per second across the batch

Use warmup for GPU runs (graph/static cache effects) and keep logging at `INFO` only when you need
per-step traces.
