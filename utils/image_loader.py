"""Utilities for loading images from OpenAI-style inputs.

This module is intentionally small and has no global side effects. It is used by the
FastAPI server to support VLM requests that carry `image_url` inputs.
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Tuple
from urllib.parse import unquote, urlparse

import httpx


def _split_data_url(url: str) -> Tuple[str, bytes]:
    if not url.startswith("data:"):
        raise ValueError("not a data url")
    header, sep, payload = url.partition(",")
    if not sep:
        raise ValueError("invalid data url")
    is_base64 = header.endswith(";base64")
    if not is_base64:
        return header, unquote(payload).encode("utf-8")
    try:
        return header, base64.b64decode(payload, validate=False)
    except Exception as exc:
        raise ValueError(f"invalid base64 data url: {exc}") from exc


def _import_pil() -> Any:
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Pillow is required for VLM image inputs; install `pillow`."
        ) from exc
    return Image


def load_pil_image(
    url: str,
    *,
    timeout_s: float,
    max_bytes: int,
    allow_file: bool,
) -> object:
    """Load an image for VLM inference.

    Supported schemes:
    - `data:image/...;base64,...`
    - `http(s)://...` (downloaded with a timeout and size cap)
    - `file://...` (disabled by default; enable via `allow_file=True`)
    """

    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")

    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()

    if url.startswith("data:"):
        _, raw = _split_data_url(url)
        if len(raw) > max_bytes:
            raise ValueError(f"image too large (bytes={len(raw)} max={max_bytes})")
        Image = _import_pil()
        return Image.open(BytesIO(raw)).convert("RGB")

    if scheme in {"http", "https"}:
        with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            raw = resp.content
        if len(raw) > max_bytes:
            raise ValueError(f"image too large (bytes={len(raw)} max={max_bytes})")
        Image = _import_pil()
        return Image.open(BytesIO(raw)).convert("RGB")

    if scheme == "file":
        if not allow_file:
            raise ValueError("file:// image urls are disabled on the server")
        path = Path(parsed.path).expanduser()
        if not path.is_file():
            raise ValueError(f"file not found: {path}")
        raw = path.read_bytes()
        if len(raw) > max_bytes:
            raise ValueError(f"image too large (bytes={len(raw)} max={max_bytes})")
        Image = _import_pil()
        return Image.open(BytesIO(raw)).convert("RGB")

    if scheme == "":
        if not allow_file:
            raise ValueError("path image urls are disabled on the server")
        path = Path(url).expanduser()
        if not path.is_file():
            raise ValueError(f"file not found: {path}")
        raw = path.read_bytes()
        if len(raw) > max_bytes:
            raise ValueError(f"image too large (bytes={len(raw)} max={max_bytes})")
        Image = _import_pil()
        return Image.open(BytesIO(raw)).convert("RGB")

    raise ValueError(f"unsupported image url scheme: {scheme or '<none>'}")


def extract_openai_image_url(segment: dict[str, object]) -> Optional[str]:
    """Return a URL string from an OpenAI-style `image_url` content segment."""

    raw = segment.get("image_url")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        candidate = raw.get("url")
        if isinstance(candidate, str):
            return candidate
    return None
