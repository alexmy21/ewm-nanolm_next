# HLLSet Cortex — Design Document

## Architecture

ds-ocr and hllset-cortex are **independent modules**. The boundary is
encoding IDs: ds-ocr produces them, hllset-cortex processes them, ds-ocr
consumes the restored IDs. hllset-cortex never sees real tokens.

```text
┌──────────────────────────────────────────────────────────────────┐
│                        DeepSeek-OCR                              │
│                                                                  │
│  Image → Vision Encoder (SAM+CLIP)                               │
│              │                                                   │
│              ▼                                                   │
│         Real tokens (text)                                       │
│              │                                                   │
│  ┌───────────┴──────────────────────────────────────────────┐    │
│  │              OCR Encoder (token → encoding ID)           │    │
│  └───────────┬──────────────────────────────────────────────┘    │
│              │                                                   │
│              │  encoding IDs (enc10253, enc18278, ...)           │
│              ▼                                                   │
│  ╔══════════════════════════════════════════════════════════╗    │
│  ║              hllset-cortex (black box)                   ║    │
│  ║                                                          ║    │
│  ║  encoding IDs → hllset_py.Tokenizer (standard pipeline)  ║    │
│  ║    → MurmurHash3 → HLLSet (32,768-bit fingerprint)       ║    │
│  ║      → ∩ gate_TF HLLSet (decoder vocabulary filter)      ║    │
│  ║        → TokenLut (monotonic TF, pre-gate)               ║    │
│  ║          → materialize (TF-ranked disambiguation)        ║    │
│  ║                                                          ║    │
│  ║  Results:                                                ║    │
│  ║    1. Token-LUT: encoding_id → hash_position (+ TF)      ║    │
│  ║    2. HLLSets: content-addressed structural fingerprints ║    │
│  ║    3. Lattice: pages → chapters → books (union ops)      ║    │
│  ╚══════════════════════════════════════════════════════════╝    │
│              │                                                   │
│              │  restored encoding IDs                            │
│              ▼                                                   │
│  ┌───────────┴──────────────────────────────────────────────┐    │
│  │              OCR Decoder (encoding ID → token)           │    │
│  └───────────┬──────────────────────────────────────────────┘    │
│              │                                                   │
│              ▼                                                   │
│         Real tokens (text) → output document/pdf/image           │
└──────────────────────────────────────────────────────────────────┘
```

## Two spaces, two morphisms

Tokens (encoding IDs) and HLLSets live in **different spaces** and never cross
directly:

- **Token space** — `tid{n}` encodings: input, response, and every interchange
  with the LLM (ds-ocr). The TokenLUT (reverse index + TF) and materialization
  live here.
- **HLLSet space** — 32,768-bit sketches: every *structural* operation (union,
  intersection, difference, popcount, BSS, R-link, DRN, the temporal pyramid).

The only morphisms between the spaces are:

```text
ingest      tokens → HLLSet   (hash + bootstrap: n-grams, multi-seed)
materialize HLLSet → tokens   (active positions → TokenLUT lookup)
```

So structural work is done only on HLLSets; any input/response/interchange with
the LLM is done only in tokens. A token never becomes a structural value except
through ingest; an HLLSet never becomes a token except through materialize.

## Component Roles

### OCR Encoder/Decoder (ds-ocr, external)

The ds-OCR model maps between real tokens (words) and encoding IDs.
hllset-cortex has **no access** to this mapping — it only sees encoding
IDs as opaque byte sequences. The encoder is simulated in the notebook
for demonstration; in production, it is the model's internal vocabulary.

### hllset_py.Tokenizer (standard hllset-dsl pipeline)

The canonical HLLSet Algebra tokenizer from hllset-dsl. Processes
encoding IDs through the standard pipeline:

```text
Bytes → [Pattern Match] → Tokens → [Normalize] → [N-grams] → [Boundary Pad]
```

- N-grams use NUL (`\x00`) separator — the standard HLLSet convention
- 3-gram encoding (1..3) provides structural fingerprinting
- Composable: pattern, normalizers, n-gram range, boundary padding
- Exposed as `hllset_py.Tokenizer` — configured via `domain.py`

### Gate (gate_TF HLLSet, content-addressed)

Built once from the decoder's valid encoding ID vocabulary. Applied via
HLLSet intersection at the lattice level — a **bit-level indirect filter**.

- Content-addressed: `h:<sha1>`, immutable (IICA)
- Intersection is probabilistic: invalid ID survives only if all its
  hash positions collide with valid IDs (~10⁻⁵ with 3-gram encoding)
- Gate never changes at runtime; rebuilt on vocabulary update
- Per STANDARD.md §2.2: akin to `system:global_1`

The sketch gate is a **tunable approximation**, not an exact filter. Its leak
has two sources, both tunable:

| Source | Tuning |
| --- | --- |
| Hash collision (64-bit MurmurHash3) | widen the hash 64 → 128 bits |
| Bootstrap order (3-gram / 3-seed) | increase n-grams / seeds |

The "~10⁻⁵" figure holds in the *sparse* multi-gram regime, where an invalid
ID must survive several independent position intersections. It breaks under
**saturation**: a full-vocabulary single-seed gate saturates the 32,768-bit
sketch, the collisions become correlated, and the leak rises to ~97% (the
STANDARD.md Appendix A finding). Membership in that regime therefore uses the
**exact-LUT forward map** (`grounding.py::exact_known`), not the sketch.

### HLLSet (32,768 bits, fixed)

The structural fingerprint. Every encoding ID is hashed via MurmurHash3
to a bit position in a 1,024 × 32 register array. Same ID → same
position (IICA). Multiple IDs may collide — the LUT resolves ambiguity.

- Size: 4,116 bytes
- Content-addressed: key = SHA1(bytes)
- Operations: AND, OR, XOR, popcount — single-cycle on FPGA

### TokenLut (dynamic, learned)

Maps bit positions back to encoding IDs. Starts empty (cold start),
accumulates encoding IDs + TF monotonically as streams are ingested.
Materialization selects highest-TF ID at each active bit position.

- Start: empty (cold start per STANDARD.md Appendix D)
- Growth: each stream adds new IDs, increments TF for seen IDs
- TF is monotonic CRDT — never decreases
- Convergence: after ~50-100 streams, disambiguation stabilizes

#### Materialization strategies

Three strategies in hllset-dsl, all accessible from Python:

| Strategy | Binding | How it works |
|----------|---------|--------------|
| `materialize` (InLUT) | `hllset_py.materialize()` | Each set bit → lookup in LUT → return candidates (TF-ranked, unordered) |
| `materialize_debruijn` | `hllset_py.materialize_debruijn()` | Build De Bruijn graph from bigrams, find Eulerian path — **order preserved** |
| `materialize_top_n` | `hllset_py.materialize_top_n()` | Top-N tokens by TF across all active positions |

```python
# Basic materialize (set-level, TF-ranked, no order)
result = hllset_py.materialize(hllset, lut)

# De Bruijn (ordered reconstruction):
tok = Tokenizer().lowercase().pad(b"<S>", b"</S>").ngrams(2, 2)
result = hllset_py.materialize_debruijn(hllset, lut, "<S>", "</S>")
# result: ['<S>', 'the', 'neural', 'network', 'model', ... '</S>']
```

The De Bruijn strategy uses boundary-padded bigrams to build a graph where
each bigram `a\0b` becomes an edge from `a` to `b`. The Eulerian path from
`<S>` to `</S>` reconstructs the original token sequence order.

### Lattice

Encoding streams form a lattice under union (OR) and intersection (AND):

```text
page₁ → HLLSet₁
page₂ → HLLSet₂
...
chapter₁ = ∪{page₁, page₂, ...}
book = ∪{chapter₁, chapter₂, ...}
```

The lattice enables structural queries: BSS similarity between pages,
R-link intersections between chapters, rank-based relevance over time.

### Document decomposition — the atom granularity

A document is still **one HLLSet** — but that HLLSet is the **union of its
parts** (pages): a *view* over the parts (`v:` prefix, ephemeral), not a
separately-persisted object. The parts are the persisted, content-addressed,
individually-addressable **atoms** (`o:` originals). What varies is only
**what the atoms are**:

| Decomposition | Project | Atom (persisted, `o:`) | View (union, `v:`) |
| --- | --- | --- | --- |
| **Embedding** | EWM-nanoLM (§10.5/10.6) | `embed-HLLSet(wte[t_i])` — a token's embedding neighborhood | `doc = ⋃ embed-HLLSet(wte[t_i])` |
| **Page** | hllset-cortex (ds-ocr) | `page_i → HLLSet_i` | `chapter = ⋃ pages`, `book = ⋃ chapters` |

This is what makes EWM a **semantic search engine**: because the atoms are
individually content-addressed, a match resolves **down to the page**, not
just to the whole document. The decomposition therefore sets the
**granularity of retrieval** — finer atoms ⇒ more precise search. The page
atom is the superior choice here: a scanned page is a natural, physical
measurement unit, so no synthetic embedding neighborhood (and no LLM) is
needed.

The rule (STANDARD.md §5.7 "bridges are not special"): **keep what works, and
adopt a domain-native atom when one serves better in a special environment.**
Embedding stays for general LLM use; page replaces it in OCR. Each new domain
may bring its own atom (image patches, audio frames, DNA k-mers) with zero
change to the downstream algebra — gate, LUT, grounding, DRN, and the
temporal pyramid all consume HLLSets regardless of atom.

## Usage Scenario: PDF Book Scanning

```text
1. Scan pages:
   page₁ → OCR encoder → encoding IDs → hllset-cortex → HLLSet₁
   page₂ → OCR encoder → encoding IDs → hllset-cortex → HLLSet₂
   ...

2. Build chapters:
   chapter₁ = HLLSet₁ ∪ HLLSet₂ ∪ ... ∪ HLLSet₁₀   (10 pages)
   chapter₂ = HLLSet₁₁ ∪ ... ∪ HLLSet₂₀

3. Build book:
   book = chapter₁ ∪ chapter₂ ∪ ... ∪ chapter_N

4. Commit to temporal pyramid (STANDARD.md §4.2):
   L₀ (second) → L₁ (minute) → ... → L₆ (year)
   → holographic memory (STANDARD.md §4.11)
```

At every level — page, chapter, book, temporal layer — the same five
operations apply: ∪, ∩, \\, popcount, key(). No new algebra needed.

## LUT Initialization Constraint

Per STANDARD.md Appendix D: Loading the LUT with equal-TF external
vocabulary causes possible random materialization (Jaccard ≈ 0.03).

**Rule:** The LUT may only contain encoding IDs whose TF reflects
actual experience. Three valid states:

| State | Vocabulary | TF | When |
| ------- | ----------- | ----- | ------ |
| Cold start | Empty | N/A | New system |
| Lattice-covered | From current HLLSet corpus | From materialization | Resume session |
| Donor transfer | From donor LUT | Copied from donor | Knowledge transfer |

The decoder vocabulary is never loaded INTO the LUT (it is compressed into HLLSet) — it acts only
as a GATE (gate_TF HLLSet) that filters invalid encoding IDs from generated HLLSets.

### Gate Intersection: Probabilistic Filtering

The gate_TF HLLSet intersection is **probabilistic**: an invalid encoding
ID may survive iff its hash positions collide with valid-vocabulary IDs.
With 3-gram encoding, the invalid ID must survive 3 independent position
intersections — ~10⁻⁵ for a 32,768-bit space. Effectively deterministic.

### Latent Vocabulary (TF Storage vs Gate Access)

The LUT accumulates TF from **all encoding IDs** — including those
filtered by the gate. This creates a **latent vocabulary**: IDs that are
currently "illegal" still accumulate experience. When the decoder
vocabulary expands (model upgrade), rebuilding the gate makes previously
illegal IDs legal — they materialize immediately at their earned TF.
No cold start penalty. In order to implements new entries into vocabulary, you should only "update" TF-gate HLLSet, but due to immutability of all HLLSets, you should generate new instance of TF-gate HLLSet and add it to the top layer of temporal pyramid.

This is the TF-vs-rank separation (STANDARD.md §3.1): **TF is stored
monotonically (pre-gate), rank is derived at query time (post-gate).**
The gate controls what's rankable, not what's storable.

## Rust Backend (hllset_py)

Self-contained PyO3 crate wrapping hllset-next crates:

```text
crates/hllset_py/
├── src/
│   ├── lib.rs        — module: HLLSet, TokenLut, Tokenizer, materialize
│   ├── hllset.rs     — HLLSet Python class (IICA, BSS, lattice ops)
│   ├── lut.rs        — TokenLut with monotonic TF (CRDT)
│   └── tokenizer.rs  — hllset-dsl Tokenizer Python wrapper
├── vendor/
│   ├── hllset-core/  — HLLSet, MurmurHash3, BSS, content addressing
│   └── hllset-dsl/   — Tokenizer, Pattern (standard pipeline)
├── Cargo.toml
└── pyproject.toml
```

Zero external Rust dependencies beyond crates.io.

## Dependencies

- **hllset-py**: Self-contained Rust PyO3 binding (vendored hllset-core + hllset-dsl)
- **Python 3.10+**: Pipeline integration layer
- **No GPU required** — HLLSet operations are CPU-only, 32K bit ops

## Relationship to DeepSeek-OCR

ds-ocr and hllset-cortex are **totally independent**:

| Concern | ds-ocr | hllset-cortex |
| --------- | -------- | --------------- |
| Token meaning | Knows words | Sees only encoding IDs |
| Encoding map | Owns token↔ID table | No access |
| Gate vocabulary | Provides valid ID set | Builds gate_TF HLLSet |
| Input | Images | Encoding ID streams |
| Output | Encoding IDs | Restored encoding IDs |

The integration point: ds-ocr encoder → encoding IDs → hllset-cortex →
restored IDs → ds-ocr decoder. hllset-cortex is a black box at this boundary.

### The encoder is bypassable — the voc-Gate is the correctness boundary

Because only encoding IDs cross the boundary (STANDARD.md §10.1) and EWM is
encoding-agnostic (§10.9 "one EWM, many LLMs"), the ds-ocr vision encoder is
**not required**: encoding IDs may come from *any* source (another tokenizer,
a simulated stream, a general LLM). The safety net that makes this safe is the
**vocabulary gate**:

- a *wrong* (out-of-vocabulary) id **can** land in the TokenLUT — the LUT is
  never gated, everything measured is stored (TF stored pre-gate);
- but it is **filtered at materialization** (rank derived post-gate), so it
  **never reaches the ds-ocr decoder**.

This guarantee is **exact only for the exact-LUT gate** (`grounding.py::
exact_known`, 0 leak / 0 false negatives). The single-HLLSet sketch gate
leaks ~97% of out-of-vocabulary ids under saturation (Appendix A), which is
why the exact-LUT forward map is the gate, not the sketch. With it, feeding
arbitrary ids is safe — invalid ids are measured but never expressed to the
decoder — so the real vision-encoder plumbing can be skipped.

### Real Encoding ID Format

With local DeepSeek-OCR, encoding IDs are **BPE token IDs** from the
model's 128,000-token vocabulary, formatted as `tid{N}` strings:

```text
Image → DeepSeek-OCR (vision encoder + LLM decoder)
  → OCR text: "The neural network model"
  → AutoTokenizer.encode() → [0, 671, 18308, 4854, 2645]
  → encoding IDs: "tid0 tid671 tid18308 tid4854 tid2645"
  → hllset-cortex → restored IDs → decoder → text
```

The simulated format (`enc10253`) and real format (`tid671`) are structurally
identical — hllset-cortex is encoding-agnostic. MurmurHash3 treats both as
opaque byte sequences.

| Format | Example | Source | Vocab size |
| -------- | --------- | -------- | ------------ |
| Simulated | `enc10253` | Mock `_encode_map` dict | 35 mock IDs |
| Real ds-OCR | `tid671` | `AutoTokenizer` (BPE) | 128,000 token IDs |

### RTX 3060 (12GB) Deployment

DeepSeek-OCR model runs locally in **Gundam mode**:

```python
# Gundam mode: base_size=1024, image_size=640, crop_mode=True
# Optimized for 12GB VRAM — MAX_CROPS=6, CROP_MODE=True
model.infer(tokenizer, prompt='<image>\nFree OCR.',
            image_file=path, output_path=out,
            base_size=1024, image_size=640, crop_mode=True)
```

| Metric | Value |
| -------- | ------- |
| Model size | 6.36 GB (safetensors) |
| GPU VRAM allocated | 6.3 GB |
| GPU VRAM free | ~5.7 GB (KV cache headroom) |
| Resolution mode | 640px tiles, dynamic cropping |
| OCR latency (simple image) | ~5 seconds |
| Supported input | PNG, JPG, PDF (via PyMuPDF) |

### Environment

Two isolated environments support different use cases:

| Env | Packages | Has ds-OCR? | Use case |
| ----- | ---------- | ------------- | ---------- |
| `.venv` | hllset-py, nbformat | No | HLLSet algebra, notebooks |
| `deepseek-ocr` (conda) | torch 2.4.0, transformers 4.46.3, vllm 0.6.3 | Yes | Full OCR + hllset pipeline |

`setup.sh` installs hllset-cortex into both environments.

## Use Cases

### Document Archive → Holographic Memory

```text
PDF book → DeepSeek-OCR (page by page)
  → token IDs → hllset-cortex → HLLSet₁, HLLSet₂, ...
  → ∪ chapter HLLSets → ∪ book HLLSet
  → commit to temporal pyramid L₀→L₆
  → query by BSS structural similarity
```

### Cross-Document Semantic Search

```text
corpus → each doc → HLLSet fingerprint
BSS(HLLSet_a, HLLSet_b) → similarity score
  → related documents cluster across languages and formats
  → gate_TF filters irrelevant vocabulary fragments
  → shadow indexing: similar docs discover each other
```

### Latent Vocabulary Activation

```text
Phase 1: Narrow gate (1000 IDs) → new IDs survive in LUT, filtered from output
Phase 2: Expanded gate (5000 IDs) → previously hidden IDs instantly rankable
Key property: TF earned during Phase 1 persists across gate changes
Per STANDARD.md §3.1: TF stored pre-gate, rank derived post-gate
```

### Model Upgrade Without Reindexing

```text
ds-OCR v2 → new tokenizer vocabulary → rebuild gate_TF HLLSet
  → old LUT still valid (encoding IDs are just hashed bytes)
  → new IDs accumulate alongside old ones in shared LUT
  → no cold start, no migration, no reindexing
```

## References

- [STANDARD.md](docs/STANDARD.md) — governing development standard
- [IICA_PRINCIPLES.md](docs/IICA_PRINCIPLES.md) — IICA gate definition
- hllset-next notebooks: `08_holographic_memory.ipynb` — temporal pyramid
