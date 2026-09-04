#!/usr/bin/env python3
"""
Append Section 11 (EWM grounding / search / localization) to the real-OCR
notebook, using the Phase-7 grounding layer (lattice.py, grounding.py,
search.py).

Run:
    cd /home/alexmy/SGS/DeepSeek-OCR/hllset_cortex
    ./.venv/bin/python notebooks/extend_notebook_grounding.py
"""
import nbformat as nbf
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent / "01_ocr_hllset_pipeline_real.ipynb"

nb = nbf.read(str(NOTEBOOK), as_version=4)

# Idempotency guard: skip if Section 11 already present.
if any("## 11. EWM Grounding" in "".join(c.get("source", [])) for c in nb.cells):
    print("Section 11 already present — nothing to do.")
    raise SystemExit(0)

new_cells = []


def md(source):
    new_cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    new_cells.append(nbf.v4.new_code_cell(source))


md("""---
## 11. EWM Grounding, Search & Localization

hllset-cortex is now an EWM **search engine**, not just a filter. The real
encoding IDs from §10 become **page atoms** (`o:`), the document a **view**
(`v:`), and a query is *submitted* to the lattice (measured), then:

- **search** ranks pages by BSS τ (coverage),
- **localization** resolves to the page,
- **hallucination diagnostics** flag *token* hallucination (unknown encodings,
  diagnosed by the LUT) and *structural* hallucination (BSS ρ departure).

This mirrors `notebooks/search_page_level.py` and
`notebooks/phase7_ewm_llm_loop.py`, but against the **real** DeepSeek-OCR
encoding IDs produced in §10.""")

md("""### 11.1 Build a multi-page lattice from the real encoding IDs

Split the single test image's token-id stream into contiguous "pages". A real
book would give one page per scanned image; here we chunk one stream to
exercise the page-granular machinery.""")

code("""from hllset_cortex import Lattice, hllset_from_ids, tid

PAGE = 4  # ids per page
pages = []
for k in range(0, len(real_ids), PAGE):
    chunk = list(dict.fromkeys(real_ids[k:k + PAGE]))  # dedupe, keep order
    pages.append((f"page_{k // PAGE}", chunk))

lattice = Lattice.from_pages(pages)
print(f"pages: {len(lattice.doc.pages)}, "
      f"view popcount={lattice.doc.view.popcount()}, "
      f"LUT={lattice.lut.len()} tokens, pyramid steps={lattice.pyramid.steps}")
for p in lattice.doc.pages:
    print(f"  {p.page_id:8} ids={p.hllset.popcount()}  key={p.key[:28]}...")""")

md("""### 11.2 Hallucination diagnostics — token vs structural

Two distinct diagnoses in two different spaces:
1. **Token hallucination** (token space): an encoding that never arrived
   through ingestion is unknown — the LUT flags it.
2. **Structural hallucination** (HLLSet space): even when every token is
   known, the response may depart from the *current* context — BSS ρ flags it.""")

code("""from hllset_cortex import GroundingConfig, grounding_report

cfg = GroundingConfig(tau_min=0.8, rho_max=0.2, structural_rho_max=0.2)

# (1) token hallucination — an out-of-vocabulary encoding never ingested
oov_id = ds_tokenizer.vocab_size + 12345
resp_unknown = [f"tid{i}" for i in real_ids[:3]] + [f"tid{oov_id}"]
rep = grounding_report(lattice.context, resp_unknown, lattice.lut, cfg)
print(f"unknown token        : {rep}")

# (2) structural hallucination — every token known, but departs from a window
sub_context = hllset_from_ids(real_ids[:2])          # the current window
resp_shift = [f"tid{real_ids[0]}", f"tid{real_ids[-1]}"]
rep2 = grounding_report(sub_context, resp_shift, lattice.lut, cfg)
print(f"known, out-of-window: {rep2}")""")

md("""### 11.3 Search + localization — resolve to the page

Query the lattice with a phrase's encoding IDs; the page covering it ranks top.""")

code("""if len(real_ids) >= 6:
    query_ids = list(dict.fromkeys(real_ids[4:6]))   # ids from page_1
    hits = lattice.search(query_ids)                  # submit + rank pages
    print(f"query ids: {[f'tid{i}' for i in query_ids]}")
    for h in hits:
        print(f"  {h}")
    if hits:
        print(f"localized to: {hits[0].page_id}")
else:
    print("real_ids too short for a multi-page query — skip.")""")

md("""### 11.4 Precedents — dive into history for decision-making reference

Beyond the shallow response-vs-context grounding, retrieve prior observations
(pages *and* past queries) that resemble a query, as reference.""")

code("""q = hllset_from_ids(real_ids[:2])
precs = lattice.precedents(q, top_k=4)
print("precedents (from history):")
for p in precs:
    print(f"  {p}")""")

nb.cells.extend(new_cells)
nbf.write(nb, str(NOTEBOOK))
print(f"Appended {len(new_cells)} cells -> {NOTEBOOK.name}")
