"""End-to-end smoke test for sglang-mini generation.

Think of this as the smallest possible integration run, mirroring how sglang
validates prefill + decode correctness.
"""
from __future__ import annotations

import argparse
import logging
from typing import Optional

from backend.model_backend import ModelBackend
from config import MAX_NEW_TOKENS_DEFAULT, MODEL_NAME, get_device
from engine.engine import SGLangMiniEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a full prefill+decode cycle")
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default="Hello, sglang-mini!",
        help="Prompt to feed the model",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        dest="max_new_tokens",
        help="Override the default token budget for the generation loop",
    )
    return parser.parse_args()


def main(prompt: str, max_new_tokens: Optional[int]) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )

    backend = ModelBackend(model_name=MODEL_NAME, device=get_device())
    engine = SGLangMiniEngine(
        backend=backend, max_new_tokens_default=MAX_NEW_TOKENS_DEFAULT
    )

    print(f"Using model={MODEL_NAME} on device={backend.device}")
    print(f"Prompt: {prompt!r}\n")

    def stream_callback(text_delta: str) -> None:
        print(text_delta, end="", flush=True)

    final_text = engine.run_generate(
        prompt=prompt, max_new_tokens=max_new_tokens, stream_callback=stream_callback
    )

    print("\n\n--- generation summary ---")
    print(f"prompt: {prompt!r}")
    print(f"generated text: {final_text!r}")


if __name__ == "__main__":
    args = parse_args()
    main(prompt=args.prompt, max_new_tokens=args.max_new_tokens)
