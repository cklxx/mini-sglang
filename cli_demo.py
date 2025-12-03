"""Simple CLI demo to run sglang-mini without HTTP.

This acts like a minimal client on top of sglang's API/engine stack so you
can see streaming tokens in a terminal. Enable ``--multi-turn`` to keep the
conversation history in context across turns (a tiny analogue of sglang's
chat-style loops).
"""
from __future__ import annotations

import argparse
import logging

from backend.model_backend import ModelBackend
from config import MAX_NEW_TOKENS_DEFAULT, MODEL_NAME, get_device
from engine.engine import SGLangMiniEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream tokens from mini-sglang")
    parser.add_argument(
        "--multi-turn",
        action="store_true",
        help="Keep prior turns in the prompt so you can chat turn-by-turn",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        dest="max_new_tokens",
        default=None,
        help="Optional token budget override (defaults to config setting)",
    )
    return parser.parse_args()


def build_chat_prompt(history: list[tuple[str, str]], user_turn: str) -> str:
    """Format a multi-turn conversation into a single prompt string."""

    segments: list[str] = []
    for prev_user, prev_bot in history:
        segments.append(f"User: {prev_user}\nAssistant: {prev_bot}")
    segments.append(f"User: {user_turn}\nAssistant:")
    return "\n".join(segments)


def run_single_turn(engine: SGLangMiniEngine, max_new_tokens: int | None) -> None:
    prompt = input("Enter a prompt: ")

    def stream_callback(text_delta: str) -> None:
        print(text_delta, end="", flush=True)

    engine.run_generate(
        prompt=prompt, max_new_tokens=max_new_tokens, stream_callback=stream_callback
    )
    print()


def run_multi_turn(engine: SGLangMiniEngine, max_new_tokens: int | None) -> None:
    print("Enter messages to chat. Press Enter on an empty line to exit.\n")
    history: list[tuple[str, str]] = []

    while True:
        user_turn = input("User: ").strip()
        if not user_turn:
            break

        prompt = build_chat_prompt(history, user_turn)
        print("Assistant: ", end="", flush=True)

        chunks: list[str] = []

        def stream_callback(text_delta: str) -> None:
            chunks.append(text_delta)
            print(text_delta, end="", flush=True)

        response = engine.run_generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            stream_callback=stream_callback,
        )
        history.append((user_turn, response))
        print()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )

    backend = ModelBackend(model_name=MODEL_NAME, device=get_device())
    engine = SGLangMiniEngine(
        backend=backend, max_new_tokens_default=MAX_NEW_TOKENS_DEFAULT
    )

    if args.multi_turn:
        run_multi_turn(engine, max_new_tokens=args.max_new_tokens)
    else:
        run_single_turn(engine, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
