# Agent Guidelines

1) Code must be absolutely elegant with crystal-clear ownership: each function/module does one thing, avoids duplication, and has no hidden side effects.
2) CI parity locally: install git hooks via `git config core.hooksPath .githooks` so ruff + mypy run on every commit; commits that fail lint/type checks must not be made.
3) 优先追求核心、困难的性能路径（CUDA Graph、flash-attn、原生调度/KV 管理等），不要用简单绕路的“快速”替代方案掩盖问题。
