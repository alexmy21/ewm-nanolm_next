#!/usr/bin/env python
"""Capture real LLM token-ID streams for the ewm-nanolm demo.

Loads DeepSeek-R1-Distill-Qwen-1.5B (cached on this machine) on the RTX 3060,
tokenizes a few prompts, runs greedy generation, and saves prompt/generated
token-ID streams plus decoded text to captures/llm_capture.npz.

Run:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
        /home/alexmy/.conda/envs/ewm-nanolm/bin/python scripts/capture_llm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
CAPTURES = PROJECT / "captures"
CAPTURES.mkdir(exist_ok=True)

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
PROMPTS = [
    "What is the capital of France?",
    "Explain gravity in one sentence.",
    "Write a haiku about rain.",
]

MAX_NEW_TOKENS = 64


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL_ID} on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    prompts: list[str] = []
    prompt_ids: list[np.ndarray] = []
    generated_ids: list[np.ndarray] = []
    generated_texts: list[str] = []

    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        in_ids = inputs["input_ids"][0].cpu().numpy()
        out_ids = out[0].cpu().numpy()
        new_ids = out_ids[len(in_ids):]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)

        prompts.append(prompt)
        prompt_ids.append(in_ids)
        generated_ids.append(new_ids)
        generated_texts.append(text)

        print(f"prompt: {prompt}")
        print(f"  prompt ids : {in_ids.tolist()}")
        print(f"  generated  : {len(new_ids)} tokens -> {text[:100]!r}")

    np.savez(
        CAPTURES / "llm_capture.npz",
        prompts=np.asarray(prompts, dtype=object),
        prompt_ids=np.asarray(prompt_ids, dtype=object),
        generated_ids=np.asarray(generated_ids, dtype=object),
        generated_texts=np.asarray(generated_texts, dtype=object),
    )
    print(f"saved: {CAPTURES / 'llm_capture.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
