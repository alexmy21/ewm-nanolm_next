# IICA: The Gate Definition for HLLSet Algebra Nesting

> **Date:** July 21, 2026
> **Status:** Architectural principle — re-affirming what's already in the code
> **Refs:** `UNIVERSAL_BRIDGE.md`, `TF_VS_RANK.md`, `DIMENSIONAL_NESTING.md`

---

## 0. The Principle

Every relation that links parts of the HLLSet Algebra — HLLSets, sub-algebras,
domains, nested spaces — must satisfy three properties simultaneously:

| Property | Definition | Why it matters |
|---|---|---|
| **I**dempotency | f(x) = f(x) every time | Same input produces same output, regardless of when, where, or how many times |
| **I**mmutability | f(x) = y is fixed; y never changes once computed | No state. No mutation. No temporal dependency |
| **C**ontent **A**ddressability | If a = b then f(a) = f(b); the output IS the address | Deterministic. Given the content, you can find it again |

These are not implementation details. They are the **gate definition**.
Any morphism that connects one HLLSet Algebra component to another must
be IICA-compliant. Any morphism that doesn't satisfy IICA is not a valid
connection.

---

## 1. Hash Functions Are the Canonical IICA Builders

A hash function h: X → Y is the canonical IICA morphism:

- **Idempotent**: h("山") always produces the same 64-bit value
- **Immutable**: the mapping is a mathematical function; it doesn't have state
- **Content-Addressed**: the hash IS the address of the content in bit space

MurmurHash3 is the current implementation. SHA-1 is used for content keys
(`h:<sha1>`). Both satisfy IICA. Any hash function that satisfies IICA can
be substituted.

The IICA properties of the hash function are what make the LUT monotonic
(TF never decreases), HLLSets idempotent (same tokens → same bitmap),
and content keys deterministic (same bitmap → same key).

---

## 2. Composition Preserves IICA

The critical property: **composition of IICA morphisms is IICA**.

```
h_n ∘ h_{n-1} ∘ h_{n-2} ∘ ... ∘ h_1

If each h_i satisfies IICA, then the composition satisfies IICA.
```

Proof sketch:

- Idempotency: each h_i is idempotent → the composition is idempotent
- Immutability: each h_i is a pure function → the composition is a pure function
- Content Addressability: each h_i is deterministic → the composition is deterministic

This is why nested spaces work. You don't need a new theory for each level
of nesting. You just need a composition of IICA-compliant hash functions.

---

## 3. The Chain We Already Use

Every step in the current pipeline is an IICA composition:

```
Real tokens (Chinese characters)
  │
  ▼ h_1: murmurhash3(token)
Token hashes in LUT (64-bit integers)
  │
  ▼ h_2: hash_to_position(hash) → (register, trailing_zeros)
Bit positions in HLLSet (32,768-bit bitmap)
  │
  ▼ h_3: format("reg:{r}:tz:{tz}") → murmurhash3 → set bit in target HLLSet
Bridge HLLSet in target bit space (Re-Representation)
  │
  ▼ h_4: CAAL LUT → murmurhash3 → (register, trailing_zeros)
Materialized tokens in target domain
  │
  ▼ h_5: target tokenizer → murmurhash3 → ...
... and so on, for any depth of nesting
```

Each arrow is an IICA morphism. The composition is IICA. There is no
limit to the nesting depth — you can bridge from domain A to domain B
to domain C, and the IICA chain holds.

---

## 4. G1/G2/G3 as the Universal Nesting Anchors

The global accumulators (G1, G2, G3) are the CRDT union of all n-grams
ever ingested at each level. They are themselves IICA-compliant:

- **Idempotent**: union(A, A) = A
- **Immutable**: once ingested, the bit is set; it never clears
- **Content-Addressed**: G1's content key identifies exactly which 1-grams
  have ever been seen

G1/G2/G3 serve as the **anchor points** for bridging between spaces.
Any HLLSet can be rank-correlated against G1/G2/G3 to find its position
in the lattice, regardless of which domain it came from.

---

## 5. What This Means for Implementation

1. **No new algebra is needed for nesting.** Every bridge, every sub-lattice,
   every cross-domain projection is a composition of existing IICA morphisms.
   We don't need a bridge API that's different from the hash API.

2. **The gate definition constrains what we can build.** Any new mechanism
   (temporal pyramid, flashlight forecast, rank blending) must itself be
   IICA-compliant to connect to the lattice. This is a design constraint,
   not a limitation — it's what keeps the system composable.

3. **Hash function choice is the only implementation decision.** Different
   hash functions trade off speed, collision resistance, and bit-space
   utilization. But any IICA-compliant hash function works. The algebra
   doesn't care which one you use.

4. **Chinese characters are not special.** They happen to be an IICA-perfect
   token base — fixed set, no morphology, deterministic. But any domain
   with non-inflectable tokens (image patches, sensor readings, byte n-grams)
   can be addressed the same way. The architecture is domain-universal
   because IICA is domain-universal.

---

## 5. Vocabulary Compression: Sub-Lattice Gates

When a sub-lattice has a **smaller vocabulary** than its parent (e.g.,
I Ching uses a subset of Chinese characters), the gate hash function
must compress the parent vocabulary into the sub-vocabulary space.

This is not a bug — it's the mechanism that makes sub-lattices useful.
The hash function naturally maps any input to a fixed bit space. If the
I Ching vocabulary covers a subset of bit positions, then bridging through
that hash function automatically projects the scene into the I Ching's
limited vocabulary:

```text
General Chinese scene (any ~80K chars)
  → murmurhash3 → bit positions in 32,768-bit space
  → I Ching vocabulary LUT (subset of ~200 chars)
  → only bit positions present in I Ching vocab "resonate"
  → compression: wide vocabulary → narrow sub-vocabulary
```

The IICA composition chain guarantees this compression is deterministic.
Same scene → same compressed projection → same hexagram, every time.

---

## 6. Closing the Loop: Disambiguation (R-R → R)

The gate into the sub-lattice is one-way compression. To close the loop,
we need the reverse path — **disambiguation** — which expands the
sub-lattice's output back into the parent vocabulary:

```text
R → R-R (bridge: compress into sub-lattice)
  → I Ching consult → navigate → guidance text (in sub-lattice Chinese)
  → CAAL LUT materialize (still sub-lattice vocabulary)
  → R-R → R (disambiguate: tokenize guidance with SOURCE tokenizer)
  → H_src_guidance (in parent bit space, full vocabulary)
  → FEED BACK: H_src_guidance ∪ H_src_next
```

Disambiguation is itself an IICA morphism: the guidance text (Chinese
characters from the I Ching's limited vocabulary) is tokenized by the
source-domain tokenizer (murmurhash3, source vocabulary). The resulting
H_src_guidance lives in the parent bit space — the full 32,768-bit
space with the full Chinese vocabulary.

**Structure transferred**: scene → I Ching (compressed), guidance → parent (expanded).
**Statistics NOT transferred**: each universe's LUT/TF/ranks accumulate independently.

The full cycle: `R → R-R → (sub-lattice deliberation) → R-R → R`.
IICA at every step. Compression in, expansion out. The latch closes.

---

## 7. Relation to LLMs

Traditional LLMs violate IICA at the foundation:

| IICA Property | Traditional LLM | HLLSet Algebra |
|---|---|---|
| Idempotency | ✗ Token embeddings drift with training | ✓ Same token → same hash → same bit |
| Immutability | ✗ Weights update; representations change | ✓ Bits never clear; TF only increases |
| Content Addressability | ✗ Embeddings are learned, not derived | ✓ h(x) IS the address of x |

The embedding approach is a consequence of starting with inflectable tokens
(Indo-European word pieces). When your tokens morph, you can't content-address
them — you need a learned mapping to capture "run/runs/running" as related.

CAAL inverts this: start with an IICA-compliant token base, and the entire
algebra follows. No embeddings. No gradient descent. Just compositions of
hash functions, every one of which satisfies IICA.

---

## 8. Bridges Are Not Special

The universal bridge, vocabulary compression, nested sub-lattices,
disambiguation — none of these require new algebra. They are all the
**same ingestion pattern with different token definitions:**

```text
                            ┌─ Source tokens → H_src
                            │
Any input ──→ Tokenizer ────┼─ Bit-position labels → H_bridge
                            │
                            ├─ I Ching vocabulary → H_hexagram
                            │
                            └─ Guidance text → H_src_guidance
                                     │
                                     ▼
                            Same murmurhash3
                            Same HLLSet::from_tokens()
                            Same five operations
```

What changes is the **token definition** — what string you feed to the
hash function. The core (`hllset-core`) doesn't need a `bridge` module.
It doesn't need to know about sub-lattices, compression, or disambiguation.
It just needs `from_tokens()` and the five lattice operations.

The architecture emerges from **how you tokenize**, not from new code.
Every bridge, every nested space, every feedback loop is a composition
of tokenizer choice + core operations. The IICA properties guarantee
that the composition holds.

This is why the `caal-llm` POC didn't need to change anything in
`hllset-core`. It only needed to be careful about **which tokenizer
feeds which LUT**, and let the same five operations do the rest.

---

## 9. Fire and Forget: No Coordination Needed

The full pipeline is a composition of pure functions connected by
monotonic CRDT state. No concurrency primitive is required:

```text
Every step is f(input) → output:
  compress:    char → murmurhash3 % 38 → I Ching char    (pure)
  ingest:      tokens → HLLSet::from_tokens               (pure)
  LUT.record:  token → TF += 1                            (monotonic CRDT)
  globals:     HLLSet → G1 ∪ G2 ∪ G3                     (associative, commutative, idempotent)
  consult:     H_iching → argmax BSS(H_iching, hex_i)     (pure)
  navigate:    hex → argmax R-weight(hex, j)              (pure)
  disambiguate: text → tokenizer → H_src_guidance         (pure)
  feedback:    H_src_next ∪ H_guidance                    (CRDT merge — order-independent)
```

- **No locks**: LUT.TF is monotonic (only increases); concurrent increments
  from different cycles don't conflict — max(TF_a, TF_b) is always correct.
- **No scheduling**: each cycle is independent. The feedback merge is
  union — it commutes. Cycle N's guidance merging into cycle N+1 doesn't
  depend on whether cycle N-1's guidance was already merged.
- **No backpressure**: the actuator (Fork 2) never waits for the strategic
  path (Fork 1). Fire and forget.

The only "state" is CRDT accumulators — union globals, monotonic TF
counters — that are correct under any merge order. The architecture
doesn't need a scheduler because there's nothing to schedule.

