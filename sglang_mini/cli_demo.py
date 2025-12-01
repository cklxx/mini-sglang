"""Simple CLI demo to run sglang-mini without HTTP.

This acts like a minimal client on top of sglang's API/engine stack so you
can see streaming tokens in a terminal.
"""
from __future__ import annotations

import logging

from sglang_mini.backend.model_backend import ModelBackend
from sglang_mini.config import MAX_NEW_TOKENS_DEFAULT, MODEL_NAME, get_device
from sglang_mini.engine.engine import SGLangMiniEngine


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )

    prompt = input("Enter a prompt: ")
    backend = ModelBackend(model_name=MODEL_NAME, device=get_device())
    engine = SGLangMiniEngine(
        backend=backend, max_new_tokens_default=MAX_NEW_TOKENS_DEFAULT
    )

    def stream_callback(text_delta: str) -> None:
        print(text_delta, end="", flush=True)

    engine.run_generate(prompt=prompt, max_new_tokens=None, stream_callback=stream_callback)
    print()


if __name__ == "__main__":
    main()
