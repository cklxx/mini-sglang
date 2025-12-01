"""Model backend that wraps the transformer model and tokenizer.

It mirrors sglang's model-backend layer in a tiny form: loading the
tokenizer/model, exposing prefill/decode with KV cache reuse, and hiding
device/KV details from the engine.
"""
from __future__ import annotations

import logging
from typing import Any, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


class ModelBackend:
    """Backend that handles model/tokenizer loading and forward passes."""

    def __init__(self, model_name: str, device: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        self.device = device
        eos_id = (
            self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is not None
            else self.tokenizer.sep_token_id
        )
        if eos_id is None:
            eos_id = self.tokenizer.pad_token_id
        if eos_id is None:
            eos_id = 0
        self.eos_token_id = eos_id

        logger.info(
            "Model backend ready: model=%s device=%s eos_token_id=%s",
            model_name,
            device,
            self.eos_token_id,
        )

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def tokenize(self, prompt: str) -> List[int]:
        """Convert prompt text to token ids."""
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        logger.info("Tokenized prompt into %d tokens", len(token_ids))
        return token_ids

    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token ids back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Streaming-friendly forward passes (prefill + decode)
    # ------------------------------------------------------------------

    def prefill_forward(self, prompt_ids: List[int]) -> Tuple[int, Any]:
        """Run the prefill (first) forward pass with KV cache enabled."""
        input_ids = torch.tensor(prompt_ids, device=self.device).unsqueeze(0)
        logger.info(
            "Running prefill_forward with sequence length=%d on device=%s",
            input_ids.shape[1],
            self.device,
        )
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, use_cache=True)
        logits = outputs.logits[:, -1, :]
        first_token_id = torch.argmax(logits, dim=-1).item()
        kv_cache = outputs.past_key_values
        logger.info(
            "Prefill produced first_token_id=%d (text=%r)",
            first_token_id,
            self.decode_tokens([first_token_id]),
        )
        return first_token_id, kv_cache

    def decode_forward(self, last_token_id: int, kv_cache: Any) -> Tuple[int, Any]:
        """Run a single decode step using the existing KV cache."""
        input_ids = torch.tensor([[last_token_id]], device=self.device)
        logger.info(
            "Running decode_forward with last_token_id=%d (text=%r)",
            last_token_id,
            self.decode_tokens([last_token_id]),
        )
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, past_key_values=kv_cache, use_cache=True
            )
        logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(logits, dim=-1).item()
        new_kv_cache = outputs.past_key_values
        logger.info(
            "Decode step produced next_token_id=%d (text=%r)",
            next_token_id,
            self.decode_tokens([next_token_id]),
        )
        return next_token_id, new_kv_cache

    # ------------------------------------------------------------------
    # Batched/non-streaming generation
    # ------------------------------------------------------------------
    def generate_greedy(self, prompt_ids: List[int], max_new_tokens: int) -> List[int]:
        """Run a full greedy generation in a single call for benchmarking.

        Returns only the newly generated token ids (excluding the prompt).
        """

        input_ids = torch.tensor(prompt_ids, device=self.device).unsqueeze(0)
        logger.info(
            "Running generate_greedy with seq_len=%d max_new_tokens=%d on %s",
            input_ids.shape[1],
            max_new_tokens,
            self.device,
        )
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        full_ids: List[int] = outputs[0].tolist()
        generated_only = full_ids[len(prompt_ids) :]
        logger.info(
            "generate_greedy produced %d tokens (text preview=%r)",
            len(generated_only),
            self.decode_tokens(generated_only[:5]),
        )
        return generated_only
