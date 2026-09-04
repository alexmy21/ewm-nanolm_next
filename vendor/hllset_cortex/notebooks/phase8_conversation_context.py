#!/usr/bin/env python3
"""Phase 8 — conversation context over HLLSet on the ds-ocr substrate.

Runs the conversation context model (hllset-context, mirrored in
``hllset_cortex.conversation``) end-to-end on ``tid{n}`` encoding IDs using
the real hllset_py lattice — no numpy, no GPU.

    1. exchange encoding    1/2/3-HLLSet layers with START/END padding
    2. context building     top unions + sparse adjacency matrix (1-HLLSet space)
    3. collision groups     same <reg,zeros> -> different tokens in two exchanges
    4. De Bruijn restore    3-gram graph, all START->END paths
    5. round-trip law       1-hll exact; 2/3-hll exact (perfect LUT)
    6. prompt               materialize-to-prompt + predicted continuations

The ``tid{n}`` streams here are simulated for a deterministic, GPU-free proof;
the same code accepts real ds-ocr encoding IDs with zero change.
"""

import hllset_py

from hllset_cortex.conversation import (
    END,
    START,
    ConversationContext,
    Exchange,
    bit_position,
    build_prompt,
    build_prompt_from_restored,
    roundtrip_context,
    roundtrip_exchange,
    roundtrip_exchange_with_extra_tokens,
)


def tid(i: int) -> str:
    return f"tid{i}"


print("=" * 70)
print("STEP 1  exchange encoding: 1-HLLSet ∪ 2-HLLSet ∪ 3-HLLSet")
print("=" * 70)
ex_user = Exchange.from_ids("user", [0, 671, 18308])
ex_asst = Exchange.from_ids("assistant", [671, 18308, 4854])

manual_union = ex_user.hll_1.union(ex_user.hll_2).union(ex_user.hll_3)
print(f"user exchange : 1={ex_user.hll_1.popcount()} 2={ex_user.hll_2.popcount()} "
      f"3={ex_user.hll_3.popcount()} all={ex_user.hll.popcount()}")
assert ex_user.hll.content_key() == manual_union.content_key()
assert not ex_user.is_empty()
assert len(ex_user.ngrams(2)) == 4  # _START_\0tid0, tid0\0tid671, tid671\0tid18308, tid18308\0_END_
assert len(ex_user.ngrams(3)) == 3
assert ex_user.ngrams(1) == [tid(0), tid(671), tid(18308)]

single = Exchange.from_ids("user", [42])
assert len(single.ngrams(2)) == 2 and len(single.ngrams(3)) == 1

print("STEP 1 OK\n")

print("=" * 70)
print("STEP 2  context building: top unions + adjacency matrix")
print("=" * 70)
ctx = ConversationContext.build([ex_user, ex_asst])

manual_top = ex_user.hll.union(ex_asst.hll)
print(f"context       : {ctx}")
assert ctx.top.hll.content_key() == manual_top.content_key()
assert ctx.len() == 2
assert ctx.matrix_index_is_vocabulary(), "matrix must be indexed over top.hll_1"

# "tid671 -> tid18308" follows in both exchanges.
row, col = bit_position(tid(671)), bit_position(tid(18308))
assert ctx.matrix.get(row, col) == 2, ctx.matrix.get(row, col)
# the matrix is indexed over the 1-HLLSet union + the two boundary bits.
vocab = {r * 32 + z for r, z in ctx.top.hll_1.active_positions()}
vocab.add(bit_position(START))
vocab.add(bit_position(END))
assert ctx.matrix.index_is_subset_of(vocab)
rows, cols, cells = ctx.matrix.shape()
print(f"matrix shape  : {rows} rows x {cols} cols, {cells} cells")
print("STEP 2 OK\n")

print("=" * 70)
print("STEP 3  collision groups: same <reg,zeros>, different tokens")
print("=" * 70)


def find_collision_pair(n: int):
    seen = {}
    for i in range(n):
        token = tid(i)
        pos = hllset_py.token_to_position_py(token)
        if pos in seen:
            return seen[pos], token
        seen[pos] = token
    return None


pair = find_collision_pair(4000)
assert pair is not None, "a collision pair must exist within tid0..tid4000"
t_a, t_b = pair
print(f"collision pair: {t_a} and {t_b} -> {hllset_py.token_to_position_py(t_a)}")

coll_ctx = ConversationContext.build([
    Exchange.from_tokens("user", [t_a]),
    Exchange.from_tokens("assistant", [t_b]),
])
origins = coll_ctx.shared_bit_origins(0, 1)
assert any(not o.same_token for o in origins), origins
same_bit = next(o for o in origins if not o.same_token)
print(f"shared bit {same_bit.bit}: user={same_bit.from_a} assistant={same_bit.from_b} "
      f"(same_token={same_bit.same_token})")

anchors = Exchange.from_ids("user", [0, 671, 18308]).bit_anchors(bit_position(tid(671)))
assert anchors is not None and anchors.left == [tid(0)] and anchors.right == [tid(18308)]
print(f"bit anchors for {tid(671)}: left={anchors.left} right={anchors.right}")
print("STEP 3 OK\n")

print("=" * 70)
print("STEP 4  3-gram De Bruijn restore: all START->END paths")
print("=" * 70)
restored = ctx.restore()
print(f"restored      : {restored}")
print(f"paths         : {restored.paths}")
# The two exchanges share the middle 2-gram "tid671\0tid18308", so the
# restore also enumerates hybrid paths (path switching). All paths are made
# of true edges; the two original token sets must be among them.
assert restored.len() >= 2, restored
got = [set(path) for path in restored.paths]
assert {tid(0), tid(671), tid(18308)} in got, got
assert {tid(671), tid(18308), tid(4854)} in got, got
assert 0.0 <= restored.confidence <= 1.0

# Overlapping chains that share a middle 2-gram node -> hybrid paths exist,
# but every enumerated path is made of true edges.
hybrid_ctx = ConversationContext.build([
    Exchange.from_ids("user", [1, 2, 3, 4]),
    Exchange.from_ids("assistant", [5, 2, 3, 6]),
])
hybrid_restored = hybrid_ctx.restore()
print(f"hybrid restore: {hybrid_restored}")
assert hybrid_restored.len() >= 2
print("STEP 4 OK\n")

print("=" * 70)
print("STEP 5  round-trip composition law")
print("=" * 70)
for ids in ([0, 671, 18308], [0, 0, 0], [7, 7, 8], [1, 2, 3, 4]):
    report = roundtrip_exchange(Exchange.from_ids("user", ids))
    print(f"roundtrip {ids}: {report}")
    assert report.all_exact(), (ids, report)

report_ctx = roundtrip_context(hybrid_ctx)
print(f"roundtrip hybrid ctx: {report_ctx}")
assert report_ctx.all_exact(), report_ctx

# Noisy LUT: thousands of distractor n-grams must not break the law.
distractors = [f"dx{i}\0dy{i}\0dz{i}" for i in range(2000)]
noisy = roundtrip_exchange_with_extra_tokens(ex_user, distractors)
print(f"roundtrip noisy LUT ({len(distractors)} distractors): {noisy}")
assert noisy.hll_1_exact and noisy.hll_2_exact and noisy.hll_3_exact, noisy
print("STEP 5 OK\n")

print("=" * 70)
print("STEP 6  materialize-to-prompt")
print("=" * 70)
prompt_ctx = ConversationContext.build([
    Exchange.from_ids("user", [0, 1, 2]),
    Exchange.from_ids("assistant", [2, 3, 4]),
    Exchange.from_ids("user", [1, 2]),
])
prompt = build_prompt(prompt_ctx)
print(prompt)
print("-" * 70)
assert "user: tid0 tid1 tid2" in prompt
assert "assistant: tid2 tid3 tid4" in prompt
assert prompt.rstrip().endswith("assistant:")
assert "[context: 3 exchanges" in prompt
assert "[top follow links:" in prompt
assert "[predicted continuations:" in prompt

restored_prompt = build_prompt_from_restored(restored, "tid9 tid10")
assert "user: tid9 tid10" in restored_prompt
assert restored_prompt.rstrip().endswith("assistant:")
print("STEP 6 OK\n")

print("=" * 70)
print("ALL PHASE 8 PROOFS GREEN")
print("=" * 70)
