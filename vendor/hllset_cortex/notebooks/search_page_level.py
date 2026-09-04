#!/usr/bin/env python3
"""Page-level semantic search — query submitted to the lattice (EWM search).

Full flow: build a lattice from page atoms, then search by converting a query
to an HLLSet and **submitting it to the lattice** (LUT TF + DRN + temporal
pyramid), then ranking pages by BSS τ/ρ (R-link popcount kept for FPGA compat).

Demonstrates:
  - the query is *measured*, not just compared transiently: submitting it
    grows the LUT, decomposes it against the state (DRN), and steps the pyramid,
  - the **voc-Gate is the correctness boundary**: an out-of-vocabulary id lands
    in the LUT (measured) but is blocked by the exact-LUT gate, so it never
    reaches the decoder — this is why the ds-ocr encoder can be bypassed,
  - ranking resolves down to the page (coverage τ),
  - the ρ gate is the opt-in precision signal (clean vs noisy hit),
  - addressability (by_key) and IICA determinism.

No GPU required — `tid{n}` encoding IDs are simulated; the encoder is not
needed because only encodings cross the boundary (STANDARD.md §10.1, §10.9).
"""
import hllset_py

from hllset_cortex import Lattice, SearchConfig, exact_known, hllset_from_ids, tid

pages = [
    ("page_1", [1, 2, 3]),
    ("page_2", [4, 5, 6]),
    ("page_3", [10, 11]),                                   # clean answer (ρ=0)
    ("page_4", [10, 11, 50, 51, 52, 53, 54, 55]),           # noisy answer (ρ=1)
    ("page_5", [10]),                                       # partial answer (τ=0.5)
]
lattice = Lattice.from_pages(pages)

print("=" * 70)
print("STEP 1  document = page atoms (o:) + whole-document view (v:)")
print("=" * 70)
for p in lattice.doc.pages:
    print(f"  {p.page_id:8} popcount={p.hllset.popcount():3}  key={p.key[:28]}...")
print(f"  view (v:)  popcount={lattice.doc.view.popcount()}")
print(f"  LUT={lattice.lut.len()} tokens, pyramid steps={lattice.pyramid.steps}")
assert len({p.key for p in lattice.doc.pages}) == len(pages)
assert lattice.pyramid.steps == len(pages), "each page was submitted to the pyramid"

print()
print("=" * 70)
print("STEP 2  submit a query to the lattice (it is measured)")
print("=" * 70)
lut_before = lattice.lut.len()
query_hll, d = lattice.submit_query([10, 999])   # 10 is known, 999 is novel
print(f"  query HLLSet key = {query_hll.content_key()[:28]}...")
print(f"  {d}")
print(f"  LUT {lut_before} -> {lattice.lut.len()} tokens; "
      f"pyramid steps={lattice.pyramid.steps}")
assert d.n.popcount() == 1, "only the novel id 999 is in N"
assert d.r.popcount() == 1, "the known id 10 is retained (R)"
assert lattice.lut.len() == lut_before + 1, "the novel id was measured into the LUT"
assert lattice.lut.tf(tid(999)) > 0, "tid999 is now known to the LUT"

print()
print("=" * 70)
print("STEP 3  the voc-Gate is the correctness boundary (encoder bypassable)")
print("=" * 70)
gate = hllset_py.TokenLut()
gate.record_all([tid(i) for i in range(100)])   # the decoder's valid vocabulary
wrong = tid(5000)                                # an out-of-vocabulary id
lattice.submit_query([10, 5000])
print(f"  wrong token {wrong} measured into LUT: {lattice.lut.tf(wrong) > 0}")
print(f"  voc-Gate admits it to the decoder?  {exact_known(gate, wrong)}")
assert lattice.lut.tf(wrong) > 0, "wrong token lands in the LUT (LUT is never gated)"
assert not exact_known(gate, wrong), "wrong token is blocked by the voc-Gate"

print()
print("=" * 70)
print("STEP 4  search ranks by BSS τ (coverage), resolves to the page")
print("=" * 70)
hits = lattice.search([10, 11])   # submit + rank
for h in hits:
    print(f"  {h}")
assert {h.page_id for h in hits} == {"page_3", "page_4", "page_5"}
assert hits[0].page_id == "page_3"
assert abs(hits[0].tau - 1.0) < 0.01 and hits[0].rho < 0.01
assert hits[0].weight == 2, "R-link popcount still reported (FPGA compat)"
assert hits[-1].page_id == "page_5" and abs(hits[-1].tau - 0.5) < 0.01

print()
print("=" * 70)
print("STEP 5  ρ is the opt-in precision gate (tighten rho_max)")
print("=" * 70)
tight = lattice.search([10, 11], SearchConfig(rho_max=0.5))
print("  rho_max=1.0 (off) ->", [h.page_id for h in hits])
print("  rho_max=0.5       ->", [h.page_id for h in tight])
assert {h.page_id for h in tight} == {"page_3", "page_5"}

print()
print("=" * 70)
print("STEP 6  whole-document view vs page granularity")
print("=" * 70)
q_clean = hllset_from_ids([10, 11])   # the answer is entirely in the document
doc_coverage = lattice.doc.view.bss_inclusion(q_clean)
print(f"  doc view covers the query: tau = {doc_coverage:.3f} (says 'it is here')")
print(f"  page search says:          'it is on page_3'")
assert abs(doc_coverage - 1.0) < 0.01

print()
print("=" * 70)
print("STEP 7  addressability + determinism (IICA)")
print("=" * 70)
recovered = lattice.doc.by_key(hits[0].key)
assert recovered is not None and recovered.page_id == "page_3"
print(f"  by_key({hits[0].key[:16]}...) -> {recovered.page_id}")
again = lattice.search([10, 11])
assert [h.key for h in again] == [h.key for h in hits]
print("  re-running the query returns the same ranked keys")

print()
print("=" * 70)
print("STEP 8  precedents: dive into the history for decision-making reference")
print("=" * 70)
# Beyond the shallow response-vs-context grounding: surface prior observations
# (pages *and* past queries) that resemble this query, as reference.
precs = lattice.precedents(q_clean, top_k=5)
for p in precs:
    print(f"  {p}")
assert precs[0].weight == 2 and precs[0].label == "page_3"
assert any(p.label.startswith("query_") for p in precs), "past queries are precedents too"

print()
print("SEARCH (query submitted to lattice) COMPLETE — all assertions passed")
