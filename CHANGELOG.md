# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

- Add VLM support to `POST /v1/chat/completions` via OpenAI `image_url` segments.
- Add MLX VLM backend option on MPS (`VLM_BACKEND=mlx|auto`).
- Add image generation endpoint `POST /v1/images/generations` (Z-Image and generic diffusers backends).
- Add video generation endpoint `POST /v1/videos/generations` (diffusers; frames or mp4).

## [0.1.0] - 2025-12-13

- Initial public release: streaming-first engine/backend split, caches, benchmarks, FastAPI server.
