# Security Policy

## Supported versions

This is an experimental research/engineering project. Only the `main` branch is supported.

## Reporting a vulnerability

If you believe you have found a security issue, please do **not** open a public issue with
reproduction details.

Preferred options:

1. If this repository is on GitHub, open a **GitHub Security Advisory** (private).
2. Otherwise, open an issue with the title prefixed by `[SECURITY]` and keep details minimal; a
   maintainer will follow up for reproduction details.

## Scope

- Remote execution paths: `api/server.py` request parsing and streaming.
- Dependency-driven issues: `transformers`, `torch`, `fastapi`, `uvicorn`, and optional backends.

