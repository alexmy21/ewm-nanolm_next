#!/usr/bin/env python
"""Build the ewm-nanolm notebook (5th EWM front end: traditional LLM)."""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)

KERNELSPEC = {
    "display_name": "Python 3 (ewm-nanolm)",
    "language": "python",
    "name": "python3",
}


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


SETUP = '''\
import sys
import pathlib

ROOT = pathlib.Path.cwd()
if not (ROOT / "ewm_nanolm").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

_vendor_src = ROOT / "vendor" / "hllset_cortex" / "src"
if _vendor_src.exists() and str(_vendor_src) not in sys.path:
    sys.path.insert(0, str(_vendor_src))

import sys as _sys
print("python:", _sys.executable)

import numpy as np

# If this kernel lacks the compiled hllset_py extension, append the env
# site-packages that have it (abi3 wheel, Python >= 3.10).
try:
    import hllset_py
except ModuleNotFoundError:
    _site_candidates = [
        pathlib.Path("/home/alexmy/.conda/envs/ewm-nanolm/lib/python3.10/site-packages"),
        pathlib.Path("/home/alexmy/.conda/envs/ewm-jepa/lib/python3.10/site-packages"),
    ]
    for _sp in _site_candidates:
        if (_sp / "hllset_py").exists() and str(_sp) not in sys.path:
            sys.path.append(str(_sp))
    import hllset_py

from hllset_cortex import HLLSetFilter
from hllset_cortex.domain import encoding_tokenizer, hllset_from_ids, tid
from hllset_cortex.temporal import drn

from ewm_nanolm import NanolmPipeline
from ewm_nanolm.pipeline import load_capture, synthetic_capture, ids_to_text

cap = load_capture()
print("capture:", "REAL LLM" if not cap["synthetic"] else "SYNTHETIC (real capture pending)")
print("prompts :", len(cap["prompts"]))
for p, pid in zip(cap["prompts"], cap["prompt_ids"]):
    print("  -", p, "|", len(pid), "tokens")
'''

cells = [
    md('''# ewm-nanolm — the EWM lattice inside a traditional LLM

**The 5th front end — and the anchor case.**  A causal LLM already speaks in
discrete token IDs, so the boundary needs **no quantizer**: the tokenizer's
output *is* the encoding stream.

```text
prompt ──► tokenizer ──► token IDs "tid671 tid18308 ..."
                           │
                           ▼  HLLSet lattice (IICA)
                gate_TF = the model's vocabulary (world view)
                LUT · DRN · BSS · temporal pyramid
                           │
                           ▼  materialize → restored token IDs
                           │
                           ▼
                        LLM ──► generated text
```

Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (fp16, RTX 3060).
The EWM layer itself is torch-free — it runs on the captured token streams.'''),
    code(SETUP),
    md('''## 1. The token-ID stream as measurement

The tokenizer's output is already the encoding stream.  `tid{n}` here is the
real BPE token ID of the Qwen tokenizer — the same `tid{n}` format used by
ds-hllset-cortex for DeepSeek-OCR.  No quantization step exists, because
none is needed.'''),
    code('''\
prompt_ids = cap["prompt_ids"][0]
print("prompt :", cap["prompts"][0])
print("ids    :", prompt_ids.tolist())
print("stream :", ids_to_text(prompt_ids)[:120], "...")
'''),
    md('''## 2. HLLSet — IICA over token IDs

Idempotent, immutable, content-addressed.  Two identical token streams are
the same object, by construction (Principle 2).'''),
    code('''\
import hllset_py as _hp
h1 = _hp.HLLSet.from_tokens(ids_to_text(prompt_ids).split())
h2 = _hp.HLLSet.from_tokens(ids_to_text(prompt_ids).split())
print("idempotent:", h1.content_key() == h2.content_key())
print("popcount  :", h1.popcount())
print("CID       :", h1.content_key())
'''),
    md('''## 3. gate_TF HLLSet — the vocabulary as world view

For a 151k-vocabulary LLM, the full-vocab gate would saturate the 32,768-bit
plate.  The EWM gate is therefore built from the **emergent vocabulary** —
the token IDs this conversation has actually measured.  That is the model's
*world view*: what it can express here.  A narrow gate = a restricted world
view (the cortical brake of `main.tex`).'''),
    code('''\
pipe = NanolmPipeline()
all_prompt_tids = sorted({int(i) for ids in cap["prompt_ids"] for i in ids})
pipe.set_gate(all_prompt_tids)
print("gate popcount:", pipe.gate.popcount())
print("gate CID     :", pipe.gate.content_key())
print("gate tids    :", len(all_prompt_tids))
'''),
    md('''## 4. HLLSetFilter — LUT + TF-ranked materialization

The LUT accumulates TF for every tid the model has seen — pre-gate,
monotonic, never reset.  Materialization returns the highest-TF token at each
active bit position (the Type-1 forward model).'''),
    code('''\
filt = HLLSetFilter()
filt.tokenizer = encoding_tokenizer()
stream = ids_to_text(cap["prompt_ids"][0])
res = filt.process_text(stream)
print("restored (gated set):", [t for t in res.token_strings if "\\x00" not in t])
print("stats               :", res.stats)
'''),
    md('''## 5. Full roundtrip — token stream through the lattice

The ewm-nanolm loop: tokenize → HLLSet lattice (gate ∩, LUT, De Bruijn
order restoration) → restored IDs.  Because the LLM case is natively
discrete, the roundtrip can be **exact** — same set, same order.'''),
    code('''\
pipe.set_gate(all_prompt_tids)
for i, ids in enumerate(cap["prompt_ids"]):
    result = pipe.roundtrip(ids.tolist())
    s = result.stats
    print(f"prompt {i}: {s.input_ids} tids ({s.unique_input_ids} unique) "
          f"-> gated {s.gated_ids} (set ret {s.set_retention:.3f}) "
          f"| ordered {s.restored_ids} (ret {s.ordered_retention:.3f})")
    print(f"          exact set: {s.exact_set_match} | exact order: {s.exact_order_match} "
          f"| HLLSet {s.hllset_popcount} -> gate {s.gate_popcount} | LUT {s.lut_size}")
'''),
    md('''## 6. The gate as world view — latent vocabulary

A gate built from prompt A only filters prompt B's tids from output; the LUT
keeps measuring them.  Expanding the gate (a wider world view) re-admits
them **at their earned TF** — no re-tokenization, no cold start.'''),
    code('''\
f2 = HLLSetFilter()
f2.tokenizer = encoding_tokenizer()
gate_a = hllset_from_ids(sorted({int(i) for i in cap["prompt_ids"][0]}))
f2.gate_hllset = gate_a

stream_b = ids_to_text(cap["prompt_ids"][1].tolist())
r1 = f2.process_text(stream_b)
out1 = [t for t in r1.token_strings if "\\x00" not in t]
b_tids = {f"tid{i}" for i in cap["prompt_ids"][1].tolist()}
filtered = sorted(b_tids - set(out1))
print("narrow gate restored:", len(out1), "of", len(b_tids), "prompt-B tids")
print("filtered (latent)   :", filtered[:8], "...")
print("LUT still knows them:", all(any(t == x for t, _ in f2.lut.ranked_tokens()) for x in filtered[:5]))

gate_ab = hllset_from_ids(sorted({int(i) for ids in cap["prompt_ids"] for i in ids}))
f2.gate_hllset = gate_ab
r2 = f2.process_text(stream_b)
out2 = [t for t in r2.token_strings if "\\x00" not in t]
print("wide gate restored  :", len(out2), "of", len(b_tids), "prompt-B tids")
'''),
    md('''## 7. Cross-prompt BSS — topical similarity

Prompts about different topics produce different HLLSets; the directed BSS
pair (τ = inclusion, ρ = exclusion) measures how much one prompt's
vocabulary overlaps another's.'''),
    code('''\
def bss(a, b):
    tau = a.intersection(b).popcount() / max(b.popcount(), 1)
    rho = a.difference(b).popcount() / max(b.popcount(), 1)
    return tau, rho

hs = {
    f"prompt {i}": hllset_py.HLLSet.from_tokens(ids_to_text(ids.tolist()).split())
    for i, ids in enumerate(cap["prompt_ids"])
}
base = hs["prompt 0"]
for name, hx in hs.items():
    tau, rho = bss(base, hx)
    print(f"{name:>10}: tau={tau:.3f} rho={rho:.3f}")
'''),
    md('''## 8. Holographic memory of text

The temporal pyramid accumulates prompt HLLSets across layers; the top is the
model's textual memory.  DRN decomposes each new prompt into Departed /
Retained / Novel against the previous state — the Noether balance, applied
to a language model's life.'''),
    code('''\
class Pyramid:
    """Configurable temporal pyramid over real HLLSets (top-carry guarded)."""
    def __init__(self, durations):
        self.durations = list(durations)
        self.layers = [hllset_py.HLLSet() for _ in range(len(durations) + 1)]
        self.counters = [0] * len(durations)

    @property
    def top(self):
        top = hllset_py.HLLSet()
        for layer in self.layers:
            top = top.union(layer)
        return top

    def step(self, s):
        self.layers[0] = self.layers[0].union(s)
        self.counters[0] += 1
        for i, d in enumerate(self.durations):
            if self.counters[i] < d:
                break
            self.layers[i + 1] = self.layers[i + 1].union(self.layers[i])
            self.layers[i] = hllset_py.HLLSet()
            self.counters[i] = 0
            if i + 1 < len(self.counters):
                self.counters[i + 1] += 1

    def layer_popcounts(self):
        return [layer.popcount() for layer in self.layers]

# Feed prompt + generated token streams as separate observations.
streams = [cap["prompt_ids"], cap["generated_ids"]]
pyramid = Pyramid([2, 2])
prev = hllset_py.HLLSet()
for chunk in streams:
    for ids in chunk:
        s_ep = hllset_py.HLLSet.from_tokens(ids_to_text(ids.tolist()).split())
        pyramid.step(s_ep)
        d = drn(s_ep, prev)
        print(f"layers={pyramid.layer_popcounts()} top={pyramid.top.popcount()} "
              f"| DRN R={d.r.popcount()} D={d.d.popcount()} N={d.n.popcount()}")
        prev = prev.union(s_ep)
'''),
    md('''## 9. Summary

**The anchor demo.**  The LLM is the only front end that needs no
quantizer: its token IDs *are* the encoding stream.  The lattice round-trips
them exactly (set and order), the gate-as-world-view dynamic holds (narrow
gate filters, LUT keeps auditing, wide gate re-admits at earned TF), and the
holographic memory accumulates every prompt the model has ever seen.

The chain is now complete: **nanoLM → DeepSeek-OCR → V-JEPA → NVIDIA VLA →
traditional LLM** — five front ends, one encoding-agnostic EWM boundary.

**To capture the real LLM streams** (one time):
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \\
    /home/alexmy/.conda/envs/ewm-nanolm/bin/python scripts/capture_llm.py
```
then re-run this notebook — `captures/llm_capture.npz` is picked up
automatically and the same cells run on real token streams.'''),
]

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": KERNELSPEC, "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = OUT / "01_nanolm_hllset_pipeline.ipynb"
path.write_text(json.dumps(notebook, indent=1))
print(f"wrote {path} ({len(cells)} cells)")
