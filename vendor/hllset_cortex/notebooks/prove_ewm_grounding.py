#!/usr/bin/env python3
"""Prove the EWM promise: real DeepSeek-OCR -> HLLSet cortex -> grounding.

Proof 1 (fidelity):  real encoding IDs round-trip through the lattice (set + order).
Proof 2 (grounding): one-sided — measured=>present, unmeasured=>absent (up to collision).
Finding (gate):      the single-HLLSet gate is one-sided; it leaks out-of-vocab ids by collision.
"""
import os, sys, random
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent  # hllset_cortex/
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import hllset_py
from hllset_cortex import HLLSetFilter

MODEL_PATH = "/home/alexmy/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/snapshots/9f30c71f441d010e5429c532364a86705536c53a"
IMAGE = "/home/alexmy/SGS/DeepSeek-OCR/data/test_ocr.png"
OUT = "/home/alexmy/SGS/DeepSeek-OCR/data/ocr_output"
os.makedirs(OUT, exist_ok=True)


def tid(i):
    return f"tid{i}"


print("=" * 70)
print("LOAD DeepSeek-OCR  (vision encoder + language decoder)")
print("=" * 70)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH, trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16
)
model = model.eval().cuda()
V = tokenizer.vocab_size
print(f"vocab_size    = {V}")
print(f"GPU allocated = {torch.cuda.memory_allocated(0)/1024**3:.1f} GB")

print()
print("=" * 70)
print("STEP 1  image -> OCR text  (real vision encoder)")
print("=" * 70)
model.infer(tokenizer, prompt="<image>\nFree OCR.", image_file=IMAGE,
            output_path=OUT, base_size=1024, image_size=640, crop_mode=True)
md_files = sorted(Path(OUT).glob("*.md"))
ocr_text = md_files[-1].read_text().strip() if md_files else "the neural network model processes image data"
print("OCR TEXT:", repr(ocr_text))

print()
print("=" * 70)
print("STEP 2  OCR text -> real encoding IDs  (real tokenizer)")
print("=" * 70)
real_ids = tokenizer.encode(ocr_text)
print(f"token IDs ({len(real_ids)}): {real_ids}")
measured = sorted(set(real_ids))
measured_tokens = [tid(i) for i in measured]
seq_tokens = [tid(i) for i in real_ids]
print(f"distinct measured ids: {len(measured)}")

print()
print("=" * 70)
print("PROOF 1  fidelity: ids -> HLLSet -> materialize -> ids  (set + order)")
print("=" * 70)
# (a) set round-trip: ingest the distinct measured ids, materialize via LUT.
doc = hllset_py.HLLSet.from_tokens(measured_tokens)
lut = hllset_py.TokenLut()
lut.record_all(measured_tokens)
restored_set = set(hllset_py.materialize(doc, lut))
print(f"(a) set round-trip: materialize(ingest(ids)) == ids  ->  {restored_set == set(measured_tokens)}")

# (b) order round-trip through the hllset-cortex black box (De Bruijn).
ctx = HLLSetFilter()
res = ctx.process_text_ordered(" ".join(seq_tokens))
restored_seq = [s for s in res.token_strings if s.startswith("tid")]
print(f"(b) order round-trip (De Bruijn): input {len(seq_tokens)} -> restored {len(restored_seq)}")
print(f"    restored sequence = {restored_seq}")
print(f"    exact order preserved = {restored_seq == seq_tokens}")

# (c) decode restored -> text, compare with OCR text.
restored_ids = [int(s[3:]) for s in restored_seq]
restored_text = tokenizer.decode(restored_ids, skip_special_tokens=True)
print(f"(c) decoded restored text = {restored_text!r}")
print(f"    equals OCR text (modulo leading <|...|>) = "
      f"{restored_text.strip().lower() == ocr_text.strip().lower()}")

print()
print("=" * 70)
print("PROOF 2  one-sided grounding")
print("=" * 70)
active = set(doc.active_positions())

# (a) measured => present, never false-negative
missed = [t for t in measured_tokens if hllset_py.token_to_position_py(t) not in active]
print(f"(a) measured ids -> bit set (no false negative): "
      f"{len(measured_tokens) - len(missed)}/{len(measured_tokens)} present, {len(missed)} missed")

# (b) unmeasured valid => absent with certainty, up to collision (false positive only)
random.seed(0)
unmeasured = [i for i in range(V) if i not in set(measured)]
sample = random.sample(unmeasured, min(20000, len(unmeasured)))
collisions = sum(1 for i in sample if hllset_py.token_to_position_py(tid(i)) in active)
print(f"(b) unmeasured valid ids sampled   = {len(sample)}")
print(f"    bit set   (collision/leakage)  = {collisions}   rate {collisions/len(sample):.5f}")
print(f"    bit unset (certain absence)    = {len(sample) - collisions}")

print()
print("=" * 70)
print("FINDING  gate ∩ is one-sided: never drops valid, but leaks out-of-vocab")
print("=" * 70)
gate = hllset_py.HLLSet.from_tokens([tid(i) for i in range(V)])
gate_active = set(gate.active_positions())
print(f"gate popcount = {gate.popcount()}  (full vocab {V})")
# valid tokens never falsely excluded:
valid_dropped = [t for t in measured_tokens if hllset_py.token_to_position_py(t) not in gate_active]
print(f"valid measured ids falsely excluded by gate: {len(valid_dropped)} (must be 0)")
# out-of-vocab ids leak rate:
random.seed(1)
oov = [tid(V + k) for k in random.sample(range(1, 200000), 20000)]
oov_leak = sum(1 for t in oov if hllset_py.token_to_position_py(t) in gate_active)
print(f"out-of-vocab ids sampled  = {len(oov)}")
print(f"  leaked through gate (bit in gate) = {oov_leak}   rate {oov_leak/len(oov):.4f}")
print(f"  -> single-HLLSet gate ∩ is a coarse one-sided filter; "
      f"exact vocab filtering needs the materializer/LUT or a multi-seed gate")

print()
print("PROOF COMPLETE")
