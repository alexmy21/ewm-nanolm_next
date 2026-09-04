# HLLSet Cortex — Encoding Restoration for DeepSeek-OCR

A **reference implementation** for HLLSet Algebra applications built on
[hllset-next](https://github.com/alexmy21/hllset-next).
Receives encoding IDs from ds-OCR's vision encoder, processes them through
the HLLSet Algebra pipeline, and returns restored encoding IDs for the decoder.
It is also an **EWM grounding/search engine** for the LLM (STANDARD.md Part X):
grounding (hallucination diagnostics), page-granular search, and precedent
retrieval.

ds-ocr and hllset-cortex are **independent modules**. hllset-cortex never
sees real tokens — only encoding IDs and their hashes.

## Concept

```text
ds-OCR Encoder                        ds-OCR Decoder
      │                                      ▲
      │ encoding IDs        restored enc IDs │
      ▼                                      │
╔══════════════════════════════════════════════════╗
║              hllset-cortex (black box)           ║
║                                                  ║
║  encoding IDs → Tokenizer → HLLSet → gate ∩      ║
║    → TokenLut (TF) → materialize → restored IDs  ║
║                                                  ║
║  Results: Token-LUT, HLLSets, Lattice            ║
╚══════════════════════════════════════════════════╝
```

## Scenario: PDF Book → Holographic Memory

```text
page₁ → HLLSet₁ ─┐
page₂ → HLLSet₂ ─┤
  ...            ├─ ∪ → chapter₁ ─┐
page₁₀ → HLLSet₁₀┘                ├─ ∪ → book
                    chapter₂ ─────┘
                                      │
                                      ▼
                              temporal pyramid L₀→L₆
                              (holographic memory)
```

Each scanned page produces an HLLSet. Chapters are unions of page HLLSets.
The book is a union of chapters. After scanning, the book is committed to
the temporal pyramid creating holographic memory (STANDARD.md §4.2, §4.11).

## EWM Grounding & Search

Beyond encoding restoration, hllset-cortex is an **EWM grounding/search engine**
for the LLM (STANDARD.md Part X §10.7–10.9). The grounding layer (see
`notebooks/phase7_ewm_llm_loop.py`, `notebooks/search_page_level.py`, and
`01_ocr_hllset_pipeline_real.ipynb` §11) provides:

- **Grounding** — two hallucination diagnoses: *token* (the LUT flags an
  encoding that never arrived through ingestion) and *structural* (BSS ρ flags a
  response that departs from the context even when every token is known).
- **Search + localization** — page atoms (`o:`) and a document view (`v:`); a
  query is submitted to the lattice and ranked by BSS τ, resolving down to the
  page.
- **Precedents** — dive into the temporal-pyramid history to surface prior
  observations (pages *and* queries) as reference for decision-making.
- **Encoder bypass** — the exact-LUT gate (0 leak / 0 FN) filters
  out-of-vocabulary ids, so encoding IDs may come from any source.

The boundary is **encoding-agnostic**: two spaces (tokens ↔ HLLSets) connected
only by *ingest* (tokens → HLLSet) and *materialize* (HLLSet → tokens).

## Architecture

Built directly on hllset-next per STANDARD.md. Zero caal-llm code.

| Layer | Crate | Role |
| ------- | ------- | ------ |
| Python config | `domain.py` | Tokenizer configuration + `tid{n}` token definition + `hllset_from_ids` |
| Python pipeline | `filter.py`, `pipeline.py` | Filter orchestration, gate_TF HLLSet, BPE interface |
| Python grounding | `grounding.py` | Token + structural hallucination diagnostics (exact-LUT gate + BSS ρ), τ/ρ + R-link (Part X §10.7) |
| Python temporal | `temporal.py` | DRN decomposition, L0–L6 temporal pyramid, feeder (§4.2–4.3, §10.8) |
| Python search | `search.py` | Page-granular retrieval: page atoms (`o:`) + document view (`v:`), BSS τ/ρ + R-link (§4.4) |
| Python lattice | `lattice.py` | The EWM lattice: submits every observation (pages *and* queries) to LUT + DRN + pyramid; `precedents` dives into history for decision-making reference (§4.4) |
| Python conversation | `conversation.py` | Phase 8: exchange 1/2/3-HLLSet encoding, sparse adjacency matrix, 3-gram De Bruijn restore, prompt materialization, round-trip law |
| Rust bindings | `hllset_py` (crates/) | PyO3 wrapper: HLLSet, TokenLut, Tokenizer, materialize |
| Rust core (vendored) | hllset-core, hllset-dsl | HLLSet algebra, MurmurHash3, standard tokenizer |

## Quick Start

### One-time setup

```bash
# hllset-cortex (Rust crate + Python package)
bash setup.sh
```

### hllset-cortex only (simulated encoding IDs, no GPU)

```python
from hllset_cortex import HLLSetFilter, default_tokenizer

filt = HLLSetFilter()
filt.tokenizer = default_tokenizer()

# Process encoding ID streams from ds-ocr
result = filt.process_text("enc10253 enc18278 enc50690 enc10325 enc1805 enc6579")

print(f"Restored IDs: {result.token_strings}")
print(f"HLLSet key:  {result.hllset.content_key()[:40]}...")
```

### Full pipeline with real DeepSeek-OCR (GPU required)

```bash
# Requires: RTX 3060 12GB (or compatible GPU), conda env deepseek-ocr
conda activate deepseek-ocr
cd /home/alexmy/SGS/DeepSeek-OCR/hllset_cortex

# End-to-end: image → OCR → token IDs → hllset-cortex → decode
CUDA_VISIBLE_DEVICES=0 python notebooks/e2e_dsocr_hllset.py

# Extended notebook (original tests + real ds-OCR integration)
jupyter notebook notebooks/01_ocr_hllset_pipeline_real.ipynb
```

### Environment summary

| Environment | Packages | Purpose |
| ------------- | ---------- | --------- |
| `.venv` (hllset-cortex) | hllset-py, nbformat | HLLSet algebra only (no GPU needed) |
| `deepseek-ocr` (conda) | torch 2.4.0, transformers 4.46.3, vllm 0.6.3 | Full ds-OCR model (GPU required) |

Both environments have hllset-cortex installed (setup.sh installs into both).

## Notebooks

| Notebook | Description |
| ---------- | ------------- |
| `01_ocr_hllset_pipeline.ipynb` | Validation pipeline with simulated encoding IDs (9 tests) |
| `01_ocr_hllset_pipeline_real.ipynb` | **Extended**: all 9 tests + Section 10 (real DeepSeek-OCR) + Section 11 (grounding, search & localization) |
| `08_holographic_memory.ipynb` | Temporal pyramid: pages → chapters → books → holographic memory |
| `prove_ewm_grounding.py` | Real ds-OCR → HLLSet cortex → grounding (fidelity + one-sided grounding) |
| `phase7_ewm_llm_loop.py` | **Phase 7**: EWM↔LLM loop — exact-LUT gate, τ/ρ + R-link, DRN, L0–L6 pyramid, feeder |
| `phase8_conversation_context.py` | **Phase 8**: conversation context — exchange encoding, adjacency matrix, collision groups, 3-gram De Bruijn restore, round-trip law, prompt materialization |
| `search_page_level.py` | Page-granular search: query → HLLSet → submitted to the lattice, then BSS τ/ρ ranking resolves to the page |

## Key Properties (IICA)

Per STANDARD.md Part I:

- **Idempotent**: same encoding IDs → same HLLSet, every time
- **Immutable**: HLLSets never change once created
- **Content-Addressed**: HLLSet key = SHA1 of serialized bytes

## LUT Initialization Constraint

Per STANDARD.md Appendix D: The LUT starts cold (empty) and accumulates
TF through encoding stream ingestion. Never seed with equal-TF vocabulary.

## Real DeepSeek-OCR Integration

DeepSeek-OCR runs locally on RTX 3060 (12GB VRAM) using **Gundam mode**
(base_size=1024, image_size=640, crop_mode=True). Model footprint: 6.3 GB.

| Aspect | Simulated | Real ds-OCR |
| -------- | ----------- | ------------- |
| Encoding IDs | `enc10253 enc18278` | `tid671 tid18308` |
| ID source | Mock dict (`_encode_map`) | `AutoTokenizer` (128K BPE vocab) |
| Gate vocabulary | 30 mock IDs | 2008 real token IDs (subset) |
| OCR text | Hand-crafted | Vision encoder output from image |
| GPU required | No | Yes (RTX 3060, 6.3GB) |
| Roundtrip retention | 73% (simulated) | 82% (real, set semantics) |

### Encoding ID format

Real encoding IDs are DeepSeek-OCR BPE token IDs formatted as `tid{N}`:

```text
OCR text: "The neural network model"
  → tokenizer → [0, 671, 18308, 4854, 2645]
  → encoding IDs: "tid0 tid671 tid18308 tid4854 tid2645"
  → hllset-cortex (3-gram + MurmurHash3 + gate ∩ + LUT)
  → restored IDs → decoder → text
```

hllset-cortex is **encoding-agnostic** — whether `enc10253` (simulated) or
`tid671` (real), MurmurHash3 treats all encoding IDs as opaque byte sequences.

## Use Cases

### Document archive → Holographic memory

```text
PDF book → DeepSeek-OCR (page by page)
  → token IDs → hllset-cortex → HLLSet₁, HLLSet₂, ...
  → ∪ chapter HLLSets → ∪ book HLLSet
  → commit to temporal pyramid L₀→L₆
  → query by structural similarity (BSS)
```

### Cross-document similarity search

```text
doc₁ → HLLSet₁, doc₂ → HLLSet₂, ...
BSS(HLLSet₁, HLLSet₂) → related documents cluster
Gate ∩ filters irrelevant vocabulary
Shadow indexing: similar docs find each other
```

### Latent vocabulary activation

```text
Phase 1: Narrow gate (1000 token IDs) → some IDs survive LUT, filtered from output
Phase 2: Expanded gate (5000 token IDs) → previously filtered IDs instantly rankable
No cold start — TF earned during Phase 1 persists across gate changes
```

### Model upgrade without reindexing

```text
ds-OCR v2 released → new tokenizer vocabulary
  → rebuild gate_TF HLLSet from new vocab
  → old LUT still valid (encoding IDs are just hashes)
  → new encoding IDs accumulate alongside old ones
  → no migration needed
```

## Dependencies

- `hllset-py` — self-contained Rust PyO3 binding (vendored hllset-core + hllset-dsl)
- Python 3.10+
- **For real DeepSeek-OCR**: torch 2.4.0, transformers 4.46.3, vLLM 0.6.3 (optional)
- **Model**: `deepseek-ai/DeepSeek-OCR` (~6.4GB, downloaded from HuggingFace)

## Reference

- [STANDARD.md](docs/STANDARD.md) — governing development standard
- [IICA_PRINCIPLES.md](docs/IICA_PRINCIPLES.md) — IICA gate definition
- [DESIGN.md](DESIGN.md) — this module's design
- [hllset-next](https://github.com/alexmy21/hllset-next) — platform
