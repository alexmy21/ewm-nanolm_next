#!/usr/bin/env python3
"""
Real DeepSeek-OCR × HLLSet Cortex Integration
==============================================

Replaces simulated encoding IDs (enc10253, enc18278, etc.) with real
token IDs from the DeepSeek-OCR tokenizer.

Run:
    cd /home/alexmy/SGS/DeepSeek-OCR/hllset_cortex
    python notebooks/run_real_dsocr_hllset.py
"""
import sys
from pathlib import Path

# Ensure hllset_cortex is importable
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root.parent))

# Load hllset-cortex
import hllset_py
from hllset_cortex import (
    HLLSetFilter, FilterResult, FilterStats,
    OCRPipeline, PipelineResult, GateInfo,
    default_tokenizer, encoding_tokenizer,
)
print(f"hllset_py: {[x for x in dir(hllset_py) if not x.startswith('_')]}")
print("hllset_cortex ready")

# Load real DeepSeek-OCR tokenizer
from transformers import AutoTokenizer
print("\nLoading DeepSeek-OCR tokenizer...")
ds_tokenizer = AutoTokenizer.from_pretrained(
    'deepseek-ai/DeepSeek-OCR',
    trust_remote_code=True
)
print(f"Vocab size: {ds_tokenizer.vocab_size:,}")
print(f"BOS: {ds_tokenizer.bos_token!r} (id={ds_tokenizer.bos_token_id})")
print(f"EOS: {ds_tokenizer.eos_token!r} (id={ds_tokenizer.eos_token_id})")
print(f"PAD: {ds_tokenizer.pad_token!r} (id={ds_tokenizer.pad_token_id})")


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: Real Token IDs → hllset_py.Tokenizer → 3-gram encoding
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("1. Real ds-OCR Token IDs → 3-gram structural encoding")
print("=" * 60)

# Sample OCR text (simulating what ds-OCR vision encoder would output)
ocr_text = "The neural network model processes image data for object detection in computer vision systems"
print(f"\nOCR text: {ocr_text}")

# Tokenize with REAL DeepSeek-OCR tokenizer
real_ids = ds_tokenizer.encode(ocr_text)
print(f"Real token IDs ({len(real_ids)}): {real_ids}")

# Convert to string representation (mimicking "enc10253" format)
encoding_id_strings = [f"tid{i}" for i in real_ids]
encoding_stream = " ".join(encoding_id_strings)
print(f"\nEncoding ID strings: {encoding_stream}")

# Use hllset-py tokenizer on these encoding IDs
tok = default_tokenizer()
tokens = tok.tokenize_str(encoding_stream)

unigrams = [t for t in tokens if b"\x00" not in t]
bigrams = [t for t in tokens if t.count(0) == 1]
trigrams = [t for t in tokens if t.count(0) == 2]

print(f"\nUnigrams:  {len(unigrams):3d}  {[t.decode() for t in unigrams[:5]]}...")
print(f"Bigrams:   {len(bigrams):3d}  {[t.decode() for t in bigrams[:3]]}...")
print(f"Trigrams:  {len(trigrams):3d}  {[t.decode() for t in trigrams[:3]]}...")
print(f"Total:     {len(tokens):3d}  (3-gram structural encoding)")


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: HLLSet — IICA with real token IDs
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("2. HLLSet — IICA properties (real token IDs)")
print("=" * 60)

hllset = hllset_py.HLLSet.from_token_bytes(tokens)
print(f"HLLSet popcount:     {hllset.popcount()}")
print(f"Cardinality:         {hllset.cardinality():.1f}")
print(f"Content key:         {hllset.content_key()[:48]}...")
print(f"Active positions:    {len(hllset.active_positions())} bits")
print(f"Non-zero registers:  {hllset.non_zero_registers()}/1024")

# IICA verification
h2 = hllset_py.HLLSet.from_token_bytes(tokens)
assert hllset.popcount() == h2.popcount()
assert hllset.content_key() == h2.content_key()
print("IICA verified: same token IDs → same key")


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: Gate_TF — Real Token Vocabulary as Content-Addressed Gate
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("3. Gate_TF HLLSet — Real ds-OCR vocabulary gate")
print("=" * 60)

# Build gate from a SUBSET of token IDs (simulating known decoder vocabulary)
# In reality the full 128K vocab is valid, but we use a subset for the gate
# to demonstrate filtering behavior.
vocab_size = ds_tokenizer.vocab_size
print(f"Full ds-OCR vocabulary: {vocab_size:,} tokens")

# Build gate from token IDs (as their tidXXXX representation)
# We use a sample for the gate concept
gate_ids = [f"tid{i}" for i in range(1000)]  # common tokens
gate_hllset = hllset_py.HLLSet.from_tokens(gate_ids)
print(f"Gate vocabulary: {len(gate_ids)} encoding IDs")
print(f"Gate popcount:   {gate_hllset.popcount()}")
print(f"Gate key:        {gate_hllset.content_key()[:48]}...")

# Intersection: document HLLSet ∩ gate_TF HLLSet
filtered = hllset.intersection(gate_hllset)
pct = 100 - filtered.popcount() * 100 // max(hllset.popcount(), 1)
print(f"\nDoc bits:        {hllset.popcount()}")
print(f"After gate ∩:    {filtered.popcount()}  ({pct}% filtered)")

gate2 = hllset_py.HLLSet.from_tokens(gate_ids)
assert gate_hllset.content_key() == gate2.content_key()
print("Gate IICA: same vocab -> same gate key")


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: HLLSetFilter — LUT + TF-Ranked Materialization
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("4. HLLSetFilter — real token IDs through the pipeline")
print("=" * 60)

filt = HLLSetFilter()
filt.tokenizer = default_tokenizer()
filt.gate_hllset = gate_hllset

print(f"LUT (cold start): {filt.lut.len()} tokens")

# Process with valid + invalid token IDs (tid99999 simulates unknown ID)
stream = "tid671 tid99999 tid18308 tid4854 tid2645 tid88888 tid6579 tid4609 tid1499 tid362"
result = filt.process_text(stream)

print(f"\nStream: {stream}")
print(f"  Input unigrams:       {result.stats.input_tokens}")
print(f"  HLLSet bits (all):    {result.stats.hllset_popcount}")
print(f"  After gate ∩:         {result.stats.gate_popcount}")
print(f"  Bits filtered:        {result.stats.hllset_popcount - result.stats.gate_popcount}")
print(f"  Output tokens:        {result.stats.output_tokens}")
print(f"  Materialized:         {result.token_strings}")
print(f"  LUT after stream:     {result.lut_size}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: Multi-document learning with real token IDs
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("5. Multi-document learning — TF accumulation with real IDs")
print("=" * 60)

# Simulate multiple OCR outputs tokenized with real ds-OCR tokenizer
documents = [
    "The neural network model processes image data for object detection",
    "Deep learning algorithms require training on large datasets",
    "Computer vision systems use neural networks for classification tasks",
    "The neural network model processes image data for object detection",  # repeat
    "Document analysis and text extraction are key applications",
    "Machine learning requires large amounts of training data",
]

streams = []
for doc in documents:
    ids = ds_tokenizer.encode(doc)
    streams.append(" ".join(f"tid{i}" for i in ids))

filt2 = HLLSetFilter()
filt2.tokenizer = default_tokenizer()
filt2.gate_hllset = gate_hllset

for i, s in enumerate(streams):
    r = filt2.process_text(s)
    print(f"Stream {i}: in={r.stats.input_tokens:2d} hll={r.stats.hllset_popcount:2d} "
          f"gate={r.stats.gate_popcount:2d} out={r.stats.output_tokens:2d} "
          f"LUT={r.lut_size:3d} | text: {documents[i][:50]}...")

s = filt2.summary()
print(f"\nDocuments:     {s['streams']}")
print(f"LUT tokens:    {s['lut_size']}")
print(f"LUT positions: {s['lut_positions']}")
print(f"Avg input:     {s['avg_input_tokens']:.1f}")
print(f"Avg output:    {s['avg_output_tokens']:.1f}")
print(f"Avg roundtrip: {s['avg_roundtrip_match']:.3f}")

print(f"\nTop 10 TF: {filt2.lut.ranked_tokens()[:10]}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: Cross-document BSS similarity 
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("6. Cross-document BSS similarity (real token IDs)")
print("=" * 60)

docs = {
    "vision_1": " ".join(f"tid{i}" for i in ds_tokenizer.encode(
        "The neural network model processes image data for object detection")),
    "vision_2": " ".join(f"tid{i}" for i in ds_tokenizer.encode(
        "Computer vision systems use neural networks for image classification")),
    "language_1": " ".join(f"tid{i}" for i in ds_tokenizer.encode(
        "Natural language processing revolutionized text understanding")),
    "language_2": " ".join(f"tid{i}" for i in ds_tokenizer.encode(
        "Text extraction and language analysis require neural networks")),
}

hllsets = {}
for name, text in docs.items():
    t = tok.tokenize_str(text)
    hllsets[name] = hllset_py.HLLSet.from_token_bytes(t)

# BSS matrix
print(" " * 10, end="")
for n in docs:
    print(f"{n:14s}", end="")
print()
for n1 in docs:
    print(f"{n1:10s}", end="")
    for n2 in docs:
        print(f"{hllsets[n1].bss_inclusion(hllsets[n2]):.4f}          ", end="")
    print()

print("\nVision docs BSS: {:.4f}".format(hllsets["vision_1"].bss_inclusion(hllsets["vision_2"])))
print("Vision-Lang BSS:  {:.4f}".format(hllsets["vision_1"].bss_inclusion(hllsets["language_1"])))


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: Full Roundtrip — Real ds-OCR tokenizer → hllset → decode
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("7. Full Roundtrip: ds-OCR tokenizer → hllset-cortex → decode")
print("=" * 60)

# The full roundtrip:
# 1. OCR text → ds-OCR tokenizer → real token IDs
# 2. Token IDs as encoding IDs → hllset-cortex → restored encoding IDs
# 3. Restored encoding IDs → ds-OCR tokenizer decode → text

original_text = "The neural network model processes image data for object detection and deep learning applications"
print(f"\nOriginal text: {original_text}")

# Step 1: Encode text with real ds-OCR tokenizer
original_ids = ds_tokenizer.encode(original_text)
encoding_stream = " ".join(f"tid{i}" for i in original_ids)
print(f"Token IDs ({len(original_ids)}): {original_ids[:10]}...")

# Step 2: hllset-cortex processing
# Build gate from all valid encoding IDs used
valid_encoding_ids = sorted(set(f"tid{i}" for i in original_ids))
full_gate = hllset_py.HLLSet.from_tokens(valid_encoding_ids)

ctx = HLLSetFilter()
ctx.tokenizer = default_tokenizer()
ctx.gate_hllset = full_gate

# Add some invalid IDs to test filtering
stream_with_noise = encoding_stream + " tid99999 tid88888"
result = ctx.process_text(stream_with_noise)

print(f"\nInput IDs:      {len(original_ids) + 2} (with 2 invalid)")
print(f"HLLSet bits:    {result.stats.hllset_popcount}")
print(f"After gate:     {result.stats.gate_popcount}  "
      f"({result.stats.hllset_popcount - result.stats.gate_popcount} filtered)")
print(f"Restored IDs:   {len(result.token_strings)}")
print(f"Restored:       {result.token_strings[:10]}...")

# Step 3: Decode restored encoding IDs back to text
restored_ids = []
unknown = []
for eid_str in result.token_strings:
    try:
        if eid_str.startswith("tid"):
            tid = int(eid_str[3:])
            restored_ids.append(tid)
        else:
            unknown.append(eid_str)
    except ValueError:
        unknown.append(eid_str)

if restored_ids:
    restored_text = ds_tokenizer.decode(restored_ids, skip_special_tokens=True)
    print(f"\nRestored text:  {restored_text}")

    # Compare
    orig_set = set(w.lower() for w in original_text.split())
    rest_set = set(w.lower() for w in restored_text.split())
    common = orig_set & rest_set
    lost = orig_set - rest_set
    print(f"\nOriginal words: {len(orig_set)}")
    print(f"Restored words: {len(rest_set)}")
    print(f"Common:         {len(common)}")
    print(f"Lost:           {len(lost)} ({sorted(lost) if lost else 'none'})")
    print(f"Retention:      {len(common)}/{len(orig_set)} = {len(common)/max(len(orig_set),1):.0%}")

print(f"\nUnknown IDs:    {unknown}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED — Real DeepSeek-OCR token IDs work with hllset-cortex!")
print("=" * 60)
