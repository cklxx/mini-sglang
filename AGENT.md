# Agent Guidelines

1) Code must be absolutely elegant with crystal-clear ownership: each function/module does one thing, avoids duplication, and has no hidden side effects.
2) CI parity locally: install git hooks via `git config core.hooksPath .githooks` so ruff + mypy run on every commit; commits that fail lint/type checks must not be made.
