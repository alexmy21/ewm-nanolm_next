#!/usr/bin/env python3
"""Phase 7 — the EWM<->LLM loop on the ds-ocr substrate (STANDARD.md Part X).

Runs the full exchange protocol §10.3–10.8 end-to-end on ``tid{n}`` encoding
IDs (the real ds-ocr currency), using the real hllset_py lattice — no numpy
simulation, no GPU required.

    1. gate           exact-LUT forward map vs. the leaky single-seed sketch
    2. grounding      hallucination test (exact-LUT) + τ/ρ + R-link
    3. DRN            R/D/N decomposition with content-addressed R-link
    4. pyramid        L0..L6 temporal pyramid (monotonic, bit-lossless top)
    5. feeder         emergent vocabulary (persistent LUT tokens)

The ``tid{n}`` streams here are simulated for a deterministic, GPU-free proof;
the same loop accepts real ds-ocr encoding IDs (see ``prove_ewm_grounding.py``
for the real vision-encoder path) with zero code change.
"""
import hllset_py

from hllset_cortex import (
    GroundingConfig,
    TemporalPyramid,
    drn,
    emergent_vocab,
    exact_known,
    grounding_report,
    hallucinated_positions,
)


def tid(i: int) -> str:
    """Encoding ID ``i`` in the real ds-ocr format (§10.3)."""
    return f"tid{i}"


def hllset_of(ids) -> hllset_py.HLLSet:
    return hllset_py.HLLSet.from_tokens([tid(i) for i in ids])


V = 50257  # contiguous encoding-id range (representative; ds-ocr real vocab ~128k)

print("=" * 70)
print("STEP 1  gate: exact-LUT forward map vs the single-seed sketch")
print("=" * 70)
# Valid vocabulary = tid{0..V-1}.  The exact gate is a reverse index (TokenLut).
gate_lut = hllset_py.TokenLut()
gate_lut.record_all([tid(i) for i in range(V)])
gate_sketch = hllset_py.HLLSet.from_tokens([tid(i) for i in range(V)])

# Out-of-vocabulary ids (just past the vocabulary).
oov = [tid(V + k) for k in range(20000)]

exact_leak = sum(1 for t in oov if exact_known(gate_lut, t))
exact_fn = sum(1 for i in range(V) if not exact_known(gate_lut, tid(i)))
sketch_active = set(gate_sketch.active_positions())
sketch_leak = sum(
    1 for t in oov if hllset_py.token_to_position_py(t) in sketch_active
)
print(f"exact-LUT gate : OOV leak {exact_leak}/20000 ({exact_leak/20000:.4f}), "
      f"false negatives {exact_fn}/{V}")
print(f"single-seed ∩  : OOV leak {sketch_leak}/20000 ({sketch_leak/20000:.4f})")
assert exact_leak == 0, "exact-LUT gate must have zero OOV leak"
assert exact_fn == 0, "exact-LUT gate must have zero false negatives"
assert sketch_leak / 20000 > 0.5, "single-seed sketch should saturate (leak)"

print()
print("=" * 70)
print("STEP 2  ingest a small corpus -> measured context + LUT")
print("=" * 70)
# A few "pages" of encoding IDs, with overlap so some ids persist (feeder).
pages = [
    [0, 1, 2, 3, 4, 100, 101, 102],
    [2, 3, 4, 100, 101, 200, 201],
    [100, 101, 102, 200, 201, 300],
    [102, 200, 300, 301, 400],
]
measure_lut = hllset_py.TokenLut()
context = hllset_py.HLLSet()
for page in pages:
    context = context.union(hllset_of(page))
    measure_lut.record_all([tid(i) for i in page])

measured_ids = sorted({i for page in pages for i in page})
print(f"measured context: popcount={context.popcount()}, "
      f"distinct ids={len(measured_ids)}, LUT={measure_lut.len()} tokens")

print()
print("=" * 70)
print("STEP 3  grounding: token hallucination (LUT) + structural hallucination (BSS ρ)")
print("=" * 70)
cfg = GroundingConfig(tau_min=0.8, rho_max=0.2)

# Response A — fully measured (echoes measured ids): grounded.
resp_a = [tid(i) for i in [100, 101, 200, 300]]
report_a = grounding_report(context, resp_a, measure_lut, cfg)
print(f"A (measured)  : {report_a}")
assert report_a.grounded and report_a.tau == 1.0 and report_a.rho == 0.0
assert report_a.flagged == []

# Response B — emits never-measured ids: hallucination flagged.
resp_b = [tid(i) for i in [100, 101]] + [tid(V + 7), tid(V + 8)]
report_b = grounding_report(context, resp_b, measure_lut, cfg)
print(f"B (novel)     : {report_b}")
assert not report_b.grounded and report_b.rho > 0.0
assert set(report_b.flagged) == {tid(V + 7), tid(V + 8)}
assert report_b.r_link_popcount <= report_a.r_link_popcount

# Position-level hallucination test on the response HLLSet.
resp_b_hll = hllset_of([100, 101, V + 7, V + 8])
flagged_pos = hallucinated_positions(resp_b_hll, measure_lut)
print(f"  hallucinated positions in B: {len(flagged_pos)} (never-measured)")
assert flagged_pos, "novel ids must yield never-measured positions"

# Response C — every token is known to the LUT, but one id departs from the
# *current* context: token-grounded, structurally novel.  The two diagnoses
# are distinct — the LUT flags token hallucination, BSS ρ flags structural.
sub_context = hllset_of([100, 101])   # the current window
resp_c = [tid(100), tid(200)]         # 200 is known, but not in the window
report_c = grounding_report(sub_context, resp_c, measure_lut, cfg)
print(f"C (known, out-of-window): {report_c}")
assert report_c.flagged == [], "no token hallucination: both tokens are known"
assert report_c.structural_rho > 0.0, "structural hallucination: 200 departs"

print()
print("=" * 70)
print("STEP 4  DRN: R/D/N decomposition + content-addressed R-link")
print("=" * 70)
h_prev = context  # state after the corpus
s_t = hllset_of([102, 200, 300, 500])  # a later observation (500 is new)
d = drn(s_t, h_prev)
print(f"  {d}")
# R ∪ D == H_prev  and  R ∪ N == S(t)  (bit-exact, via content key)
assert d.r.union(d.d).content_key() == h_prev.content_key()
assert d.r.union(d.n).content_key() == s_t.content_key()
assert d.n.popcount() == 1, "only the novel id 500 should be in N"
# IICA: the same R-link recomputes to the same content key.
assert drn(s_t, h_prev).r_link_key == d.r_link_key

print()
print("=" * 70)
print("STEP 5  temporal pyramid: monotonic, bit-lossless top")
print("=" * 70)
pyramid = TemporalPyramid(durations=[2, 2, 2])
all_union = hllset_py.HLLSet()
prev_top = 0
for page in pages:
    s = hllset_of(page)
    all_union = all_union.union(s)
    pyramid.step(s)
    top = pyramid.top
    assert top.popcount() >= prev_top, "top must never shrink (Noether)"
    prev_top = top.popcount()
    assert top.content_key() == all_union.content_key(), "top must be bit-lossless"
print(f"  after {pyramid.steps} steps: layers={pyramid.layer_popcounts()}, "
      f"top={pyramid.top.popcount()}")
assert pyramid.top.content_key() == context.content_key(), "top == union of all pages"

print()
print("=" * 70)
print("STEP 6  feeder: emergent vocabulary (persistent LUT tokens)")
print("=" * 70)
vocab_before = set(emergent_vocab(measure_lut, min_tf=1))
# Re-ingest one page to raise TF of its ids (recertification -> monotonic feeder).
measure_lut.record_all([tid(i) for i in pages[0]])
vocab_after = set(emergent_vocab(measure_lut, min_tf=1))
print(f"  emergent vocab: {len(vocab_before)} -> {len(vocab_after)} tokens")
assert vocab_before == vocab_after, "feeder vocab must be stable under re-ingest"
assert len(vocab_before) == len(measured_ids), "every measured id is in the feeder"
# Every feeder token is content-addressed (deterministic hash position).
for t in list(vocab_after)[:5]:
    assert measure_lut.position_of(t) is not None

print()
print("PHASE 7 LOOP COMPLETE — all assertions passed")
