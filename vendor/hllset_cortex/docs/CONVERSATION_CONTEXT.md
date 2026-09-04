# Conversation Context over HLLSet — Integration Design for hllset-cortex

> **Status:** Prototype implemented in `src/hllset_cortex/conversation.py`;
> proof script `notebooks/phase8_conversation_context.py`.
> **Upstream model:** `hllset-next/crates/hllset-context` (Rust), whose
> algorithms this module mirrors on the `hllset_py` binding.
> **Empirical status:** the round-trip composition law is empirically proven
> (55 tests in the Rust crate); a rigorous proof is future work.

---

## 1. What this brings

The conversation context model treats a user↔LLM conversation as:

```text
ConversationContext {
    exchanges : [Exchange],          // one per user/assistant turn
    top       : { hll_1, hll_2, hll_3, hll },
    matrix    : SparseAdjacencyMatrix
}
```

- **Exchange** — each turn is encoded as `1-HLLSet ∪ 2-HLLSet ∪ 3-HLLSet`
  with `_START_`/`_END_` padding. 1-grams are vocabulary; padded 2-grams are
  follow-edges; padded 3-grams are De Bruijn edges.
- **top.hll_1** — the union of the exchanges' 1-HLLSets. This is the
  **vocabulary space** the matrix is indexed over (plus the two constant
  boundary bits). The 2-/3-HLLSets are *structural equivalents*: they
  contribute edges and order, never their own bit positions as matrix
  indices.
- **matrix** — sparse follow-frequency counts `(row_bit, col_bit) → count`
  accumulated from every padded 2-gram; token-level counts are kept in
  parallel for prompt materialization.
- **restore** — a 3-gram De Bruijn walk (nodes = 2-grams, edges = 3-grams)
  enumerates every START→END path, recovering the exchanges from the top
  unions + a LUT alone.
- **prompt** — the context is materialized into a role-labelled text prompt
  with matrix-predicted continuations, ready for any LLM.
- **round-trip law** — for `hllA → materialize → {tokens} → ingest → hllB`:
  `1-hllA = 1-hllB` exactly; `2-hllA = 2-hllB` and `3-hllA = 3-hllB` with
  high probability (exactly, when the LUT is perfect).

## 2. Mapping onto hllset-cortex

hllset-cortex already has the right shape: `Lattice` submits every
observation (pages *and* queries) the same way, `grounding.py` validates
LLM responses in two spaces, and `phase7_ewm_llm_loop.py` runs the
EWM↔LLM protocol. The conversation model slots in as the **exchange layer
above the lattice**:

```text
                        ┌────────────────────────────────────────────┐
                        │        conversation.py (NEW)               │
                        │                                            │
  user turn (ids) ───▶ Exchange ──▶ ConversationContext            │
                        │  │           ├─ top unions (1/2/3-HLLSet)  │
                        │  │           ├─ adjacency matrix           │
                        │  │           └─ history of exchanges       │
                        │  │                                         │
  assistant turn ◀─────┤  └─▶ restore (3-gram De Bruijn)           │
                        │        └─▶ prompt materialization         │
                        └───────────────┬────────────────────────────┘
                                        │ HLLSets + TokenLut
                                        ▼
                       ┌────────────────────────────────────────────┐
                       │        Lattice (existing)                  │
                       │  ingest → DRN → context ∪ → L0–L6 pyramid  │
                       │  grounding (exact-LUT + BSS ρ)             │
                       │  search + precedents                       │
                       └────────────────────────────────────────────┘
```

Specific integration points:

| Existing piece | Conversation-model counterpart | How they meet |
| --- | --- | --- |
| `Lattice.context` (single union) | `ContextTop.hll` + per-layer unions | `ConversationContext` can submit each exchange to `Lattice.ingest`, keeping the measured state identical |
| `Lattice.history` (pages/queries) | `ConversationContext.exchanges` | Exchanges are the conversation-shaped history; precedents work unchanged |
| `grounding_report` (response vs context) | `suggest_continuations`, `matrix` | Grounding validates a response; the matrix proposes it. Same two-space boundary |
| `materialize_debruijn` (bigram, single path) | 3-gram De Bruijn restore (multi-path) | The prototype's restore handles the union of several padded exchanges and returns **all** START→END paths |
| `hllset_from_ids` (1-gram only) | `Exchange` (1/2/3-gram layers) | `Exchange.from_ids` is the 3-layer replacement |
| `emergent_vocab` (feeder) | `vocabulary_index`, `TokenIndex` | Both read the LUT; the conversation model adds collision groups |

## 3. Data model (Python prototype)

All classes live in `hllset_cortex.conversation` and use only `hllset_py`
primitives (`HLLSet`, `TokenLut`, `token_to_position_py`). Tokens are the
project's `tid{n}` encoding strings.

```python
Exchange(role, tokens)            # hll_1 / hll_2 / hll_3 / hll (+ ngrams())
Exchange.from_ids(role, ids)      # [0, 671] → ["tid0", "tid671"]

ContextTop                        # hll_1, hll_2, hll_3, hll (+ tz_histogram)
SparseAdjacencyMatrix             # bit counts + token counts (+ walks, unfolding)
TokenIndex                        # bit → collision group of candidate tokens
UnfoldedCell                      # one token-resolved matrix cell
SharedBitOrigin                   # same bit, per-exchange tokens (collision case)
ConversationContext               # build / add_exchange / restore / to_prompt / roundtrip
RestoredConversation              # paths + confidence (+ to_prompt)
RoundTripReport                   # per-layer exactness + Jaccard
```

## 4. Algorithms

### 4.1 Exchange encoding

```text
tokens      = [tid(a), tid(b), tid(c)]
1-HLLSet    = { hash(tid(a)), hash(tid(b)), hash(tid(c)) }
2-HLLSet    = { hash(_START_\0tid(a)), hash(tid(a)\0tid(b)),
                hash(tid(b)\0tid(c)), hash(tid(c)\0_END_) }
3-HLLSet    = { hash(_START_\0tid(a)\0tid(b)),
                hash(tid(a)\0tid(b)\0tid(c)),
                hash(tid(b)\0tid(c)\0_END_) }
exch-HLLSet = 1-HLLSet ∪ 2-HLLSet ∪ 3-HLLSet
```

NUL (`\0`) is the n-gram separator — the standard convention the binding
already uses for bigrams.

### 4.2 3-gram De Bruijn restore (multi-path)

1. Build a LUT from every n-gram of every exchange (the "perfect LUT").
2. Gather candidates per layer by scanning the layer's active positions and
   looking up **all** LUT tokens there, keeping only tokens that split into
   exactly `n` parts.
3. Cross-validate each 3-gram `a\0b\0c`: `a`,`b`,`c` must be known 1-grams
   with bits in `top.hll_1` (`_START_`/`_END_` exempt); `a\0b` and `b\0c`
   must have bits in `top.hll_2`.
4. Build the De Bruijn graph: node = 2-gram, edge = 3-gram.
5. DFS from every START node (2-gram whose first part is `_START_`) with a
   per-path used-edge set; every END-reaching path is an exchange.
6. Collapse each path of 2-grams back into tokens (emit the last token of
   each node, strip boundaries).

The existing `hllset_py.materialize_debruijn` is bigram-only and returns a
single greedy path; the prototype restores from 3-grams and returns **all**
paths, which is what a conversation union requires.

### 4.3 Round-trip composition law

```text
hllA ──materialize──▶ {tokens} ──ingest──▶ hllB
   1-hllA = 1-hllB                        (exact)
   2-hllA = 2-hllB, 3-hllA = 3-hllB       (high probability)
```

The prototype re-ingests **all** restored paths and compares the per-layer
unions (bit-exact via content keys). With a perfect LUT the structural
layers are exact as well — even for repeated tokens and hybrid paths —
because every enumerated path is made of true edges.

## 5. EWM↔LLM loop changes (phase 7 → phase 8)

The current loop:

```text
ingest pages → context ∪ LUT
query → HLLSet → submit to lattice
LLM response → grounding_report(context, response, lut)
```

With the conversation model:

```text
for each turn:
    ex = Exchange.from_ids(role, ids)        # 1/2/3-HLLSet layers
    ctx.add_exchange(ex)                     # top unions + matrix
    lattice.ingest(ex.hll, ids, label=turn)  # existing DRN + pyramid

response hints  = ctx.suggest_continuations(last_turn, k)   # matrix rule
prompt          = ctx.to_prompt()             # LLM interchange
response_ids    = llm(prompt)
report          = grounding_report(ctx.top.hll, response_ids, lut)  # unchanged
```

The two hallucination diagnoses are untouched — the conversation model adds
the **generation side** (matrix-predicted continuations + restored prompt)
while grounding keeps validating whatever the LLM produces.

## 6. Production rollout notes

1. **Python first.** `conversation.py` is the production orchestration path
   for now; it uses only binding primitives and the algorithms are
   bit-identical to the proven Rust crate.
2. **Rust acceleration later.** When matrix ops or restore become hot, port
   `hllset-context` into `hllset_py` (vendor its sources as is already done
   for `hllset-core`/`hllset-dsl`) and expose `Exchange`, `AdjacencyMatrix`,
   `debruijn3`, and `RoundTripReport` as PyO3 classes. The Python module's
   public API is designed to be a drop-in for those bindings.
3. **Gate interaction.** `Exchange` layers are gated exactly like the
   existing pipeline: intersect each layer with `gate_TF` before
   materialization; the exact-LUT gate remains the correctness boundary.
4. **Persistence.** Exchanges are content-addressed (`hll.content_key()`);
   the matrix is a derived, monotonic accumulator. Persist exchanges as
   atoms and rebuild the matrix on load (or keep it in memory).
5. **Rigor proof.** The composition law is empirically proven; a formal
   proof (collision model + path-enumeration completeness) is planned
   follow-up work.
