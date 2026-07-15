#!/usr/bin/env python3
"""Shared Qwen3.5-4B reader — the one place the hard-won runtime recipe lives.

Qwen3.5 is the production local model for the canon path (story segmentation + world
recognition + name judging). Three non-obvious nuances, learned the hard way (see the
EMP write-up and memory `plain-text-not-json`):

  1. It is a VISION-LANGUAGE model — load it with `mlx_vlm`, NOT `mlx_lm`. mlx_lm
     "loads" it but mis-drives the generation into garbage / Devanagari.
  2. It is a REASONING model that IGNORES a `/no_think` prompt prefix — disable the
     thinking block via `enable_thinking=False` in the chat template, or it burns the
     token budget on a reasoning monologue and returns nothing usable.
  3. Ask for PLAIN TEXT, never JSON. Every JSON request failed; plain lines work and
     don't waste a reasoning model's tokens. We do all JSON shaping on our side.

`make_reader()` returns a `gen(prompt, max_tokens) -> str` closure with the SAME
contract as `story_segment.make_reader` (the Gemma reader), so it is drop-in anywhere a
reader is expected. Greedy (temp 0); note mlx_vlm generation isn't fully deterministic
on Metal, so outputs can wobble a hair run-to-run.
"""

MODEL_ID = "mlx-community/Qwen3.5-4B-MLX-4bit"


def make_reader(model_id=MODEL_ID):
    """Load Qwen3.5-4B once; return gen(prompt_text, max_tokens=256) -> raw string."""
    from mlx_vlm import load, generate
    model, processor = load(model_id)
    tok = getattr(processor, "tokenizer", processor)

    def gen(prompt_text, max_tokens=256):
        fmt = tok.apply_chat_template([{"role": "user", "content": prompt_text}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
        res = generate(model, processor, fmt, max_tokens=max_tokens, temperature=0.0, verbose=False)
        return (getattr(res, "text", res) or "").strip()
    return gen
