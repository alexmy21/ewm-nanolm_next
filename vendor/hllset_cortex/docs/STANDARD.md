# HLLSet Development Standard

> **Status:** Authoritative — supersedes all per-topic dev docs listed in §0.2
> **Date:** July 25, 2026
> **Scope:** hllset-next (core library) + caal-llm (reference application)
>
> This document consolidates all development documents into a single, internally
> consistent standard. It resolves contradictions, aligns with the current code
> state, and provides the definitive architectural reference for both projects.

---

## 0. Preliminaries

### 0.1 How to Read This Document

The standard is organized in **conceptual layers**, each building on the previous:

| Part | Layer | Question it answers |
| ------ | ------- | --------------------- |
| I | Foundation | What properties must every operation satisfy? |
| II | Protocol | How is data stored, addressed, and retrieved? |
| III | Core Concepts | What is TF? What is rank? How do they relate? |
| IV | Architecture | How does the system evolve, learn, and steer itself? |
| V | Universal Bridge | How do different domains connect? |
| VI | Self-Ingestion | How does the system observe its own development? |
| VII | Status Matrix | What is actually implemented in code? |
| VIII | caal-llm Guide | How to build a reference application on this foundation? |

Every concept carries a **status marker**:

| Marker | Meaning |
| -------- | --------- |
| `[IMPL]` | Implemented in hllset-next code; tests exist; API is stable |
| `[PART]` | Partially implemented; some pieces exist, some are stubs |
| `[SPEC]` | Fully specified in this standard; no hllset-next implementation yet |
| `[INACC]` | Inaccessible to caal-llm — depends on hllset-next crates that aren't usable yet (e.g., path-dependency blocked) |

Where the predecessor documents disagreed, this standard **resolves** the
disagreement with explicit reasoning (marked **Resolution:**).

### 0.2 Source Documents (Superseded)

This standard consolidates and supersedes the following documents. After
adoption, individual docs remain as historical records — new design work
references this standard exclusively.

| Document | Primary contribution |
| ---------- | --------------------- |
| `HLPP.md` | Formal algebraic protocol specification |
| `TF_VS_RANK.md` | TF vs Rank separation principle |
| `DIMENSIONAL_NESTING.md` | D_P = N + 2 scaling theorem |
| `UNIVERSAL_BRIDGE.md` | Two-pass re-representation and domain LUTs |
| `SELF_REPROGRAMMING_ARCHITECTURE.md` | ("the bible") All architectural concepts |
| `HLLSET_NEXT_REVIEW.md` | Code-vs-docs gap analysis |
| `IICA_PRINCIPLES.md` | IICA gate definition and composition |
| `IICA_STATISTICS_CONSTRAINT.md` | Statistics are not transferable across bridges |

### 0.3 Project Roles

| Project | Role | This Standard's Authority |
| --------- | ------ | --------------------------- |
| **hllset-next** (`/home/alexmy/SGS/SGS_lib/fractal_manifold/hllset-next/`) | Core library — defines the algebra, storage protocol, and rank framework | **Definitive.** The standard is written *from* this project. Refactoring is possible as an exception; new code must comply. |
| **caal-llm** (`/home/alexmy/SGS/SGS_lib/caal-llm/`) | Reference application — demonstrates CAAL (Chinese as Assembly Language) LLM on hllset-next | **Prescriptive.** Part VIII defines the redesigned architecture. caal-llm must conform to this standard. |

---

## Part I: Foundation — The IICA Gate

### 1.1 The Three Properties

`[IMPL]` Every operation that connects HLLSet Algebra components must satisfy three
properties simultaneously. This is the **gate definition** — any morphism that
violates IICA is not a valid connection.

| Property | Definition | Consequence |
| ---------- | ----------- | ------------- |
| **I**dempotency | f(x) = f(f(x)) | Same input → same output, regardless of when/where/how many times |
| **I**mmutability | f(x) = y is fixed; y never changes once computed | No state, no mutation, no temporal dependency |
| **C**ontent-**A**ddressability | If a = b then f(a) = f(b); the output IS the address | Deterministic; given the content, you can find it again |

### 1.2 Hash Functions as Canonical IICA Builders

`[IMPL]` A hash function h: X → Y is the canonical IICA morphism:

- **Idempotent:** h("山") always produces the same value
- **Immutable:** The mapping is a mathematical function; it has no state
- **Content-Addressed:** The hash IS the address of the content in bit space

**Current implementation:** MurmurHash3 for bit-position hashing; SHA-1 for
content keys (`h:<sha1>`). Any IICA-compliant hash function can be substituted.

The IICA properties of the hash function are what make the LUT monotonic
(TF never decreases), HLLSets idempotent (same tokens → same bitmap),
and content keys deterministic (same bitmap → same key).

### 1.3 Composition Preserves IICA

`[IMPL]` **Composition of IICA morphisms is IICA.**

```text
h_n ∘ h_{n-1} ∘ h_{n-2} ∘ ... ∘ h_1

If each h_i satisfies IICA, then the composition satisfies IICA.
```

This is the theorem that makes nested spaces work. You don't need a new theory
for each level of nesting. You just need a composition of IICA-compliant hash
functions.

### 1.4 The IICA Pipeline

`[IMPL]` Every step in the current pipeline is an IICA composition:

```text
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

---

## Part II: The HLPP Protocol

### 2.1 Algebraic Specification

`[PART]` The HLLSet Lattice Persistence Protocol (HLPP) defines the formal
interface between lattice computation and persistent storage. The algebraic
specification is the source of truth. Every interface binding (Rust, HTTP, Lua,
Forth) must satisfy these laws.

#### Sorts (Types)

```text
Bytes     = byte sequence (opaque)
SHA1      = 40-char hex string
UUID      = 32-char hex string (canonical, no dashes)
Name      = UTF-8 string matching [a-zA-Z_][a-zA-Z0-9_]*
Prefix    = { o, h, r, d, n, t, v }
CID       = Prefix ":" SHA1                       -- content-addressed ID
TmpID     = "u:" UUID                             -- user-assigned temporal
          | "system:" Name                        -- named global
Key       = CID | TmpID
HLLSet    = ⟨magic:8, version:4, M:4, B:4, regs:M×B⟩
TFVec     = ⟨N:4, vals:N×f64⟩                     -- N = M × B = 32768
Commit    = ⟨ts:u64, s:CID, h:CID, d:CID, r:CID, n:CID⟩
```

#### Operations (Signatures)

```text
── CA Operations ──────────────────────────────────────────────

  PUT   : CID × Bytes → Unit
  GET   : CID → Option[Bytes]
  HAS   : CID → Bool
  LIST  : Prefix → List[CID]
  PIN   : CID → Unit
  UNPIN : CID → Unit
  GC    : Unit → List[CID]

── Temporal Operations ────────────────────────────────────────

  GET_TMP : TmpID → Option[Bytes]
  PUT_TMP : TmpID × Bytes → Unit
  CAS_TMP : TmpID × Bytes × Bytes → Bool
```

#### Laws (Invariants)

```text
── IICA ───────────────────────────────────────────────────────

  LAW put-get:     ∀ cid, bytes :
                     PUT(cid, bytes); GET(cid) = Some(bytes)

  LAW idempotent:  ∀ cid, bytes :
                     PUT(cid, bytes); PUT(cid, bytes) = PUT(cid, bytes)

  LAW sha1-match:  ∀ cid = prefix:sha1, bytes :
                     PUT(cid, bytes) ⇒ sha1 = SHA1(bytes)

── Monotonicity ───────────────────────────────────────────────

  LAW pin-idempotent:   ∀ cid : PIN(cid); PIN(cid) = PIN(cid)
  LAW unpin-idempotent: ∀ cid : UNPIN(cid); UNPIN(cid) = UNPIN(cid)
  LAW gc-pin-safe:      ∀ cid : PIN(cid) ⇒ cid ∉ GC()

── Temporal ───────────────────────────────────────────────────

  LAW tmp-put-get:  ∀ k, bytes : PUT_TMP(k, bytes); GET_TMP(k) = Some(bytes)
  LAW cas-correct:  ∀ k, old, new :
                      GET_TMP(k) = Some(old) ⇒ CAS_TMP(k, old, new) = true
  LAW cas-reject:   ∀ k, old, cur, new : cur ≠ old ⇒
                      CAS_TMP(k, old, new) = false

── Commit Chain ────────────────────────────────────────────────

  LAW commit-link:  ∀ commit c = ⟨ts, s, h, d, r, n⟩ stored at CID cid :
                      GET_HEAD() = Some(prev) ⇒ h = prev
```

#### Derived Operations

```text
PUT_HLL  : HLLSet → Unit
         = let bytes = serialize(HLLSet) in
           PUT("h:" + SHA1(bytes), bytes)

PUT_TF   : TFVec → Unit
         = PUT_TMP("system:tf", serialize(TFVec))

GET_TF   : Unit → Option[TFVec]
         = match GET_TMP("system:tf") { Some(b) ⇒ deserialize(b), None ⇒ None }

PUT_HEAD : CID → Unit
         = PUT_TMP("system:head", ascii_bytes(cid))

GET_HEAD : Unit → Option[CID]
         = match GET_TMP("system:head") { Some(b) ⇒ parse_cid(b), None ⇒ None }

COMMIT   : HLLSet^5 → CID
         = let c = Commit(now(), s, h, d, r, n) in
           let cid = "t:" + SHA1(json(c)) in
           PUT(cid, json(c));
           PUT_HEAD(cid)
```

### 2.2 Object Namespaces

`[PART]` **Current code status:** Only `h:` and `c:` prefixes are implemented in
`hllset-core/src/core/content_addr.rs`. The full taxonomy is specified below —
new code must use these prefixes.

| Namespace | Identity | Replication | Meaning |
| ----------- | ---------- | ------------- | --------- |
| `o:` | SHA1 (40 hex) | K≥3 (source) | Original HLLSet — from tokenizer, immutable |
| `h:` | SHA1 (40 hex) | K=1 (cache) | Standard HLLSet — any operation result |
| `r:` | SHA1 (40 hex) | K=1 | Retained HLLSet (R-link / intersection) |
| `d:` | SHA1 (40 hex) | K=1 | Departed HLLSet (difference) |
| `n:` | SHA1 (40 hex) | K=1 | New HLLSet (difference) |
| `t:` | SHA1 (40 hex) | K=2 | Commit object — CA by content |
| `v:` | SHA1 (40 hex) | none | View HLLSet — ephemeral, not persisted |
| `l:` | SHA1 (40 hex) | K=1 | LLM context — human annotation bridging prompts to code |
| `u:` | UUID (32 hex) | K=1 | User-assigned temporal identifier |
| `system:` | Fixed name | K=1 | Named global (tf, head, global_N) |

**System Keys:**

| Key | Type | Updated by | Description |
| ----- | ------ | ----------- | ------------- |
| `system:tf` | `TFVec` | Ingestion only | Global TF vector (32,768 × f64). Monotonic CRDT. |
| `system:tf_0` | `TFVec` | Second boundary | L0 TF snapshot |
| `system:tf_1` | `TFVec` | Minute boundary | L1 TF snapshot |
| `system:tf_2` | `TFVec` | Hour boundary | L2 TF snapshot |
| `system:tf_3` | `TFVec` | Day boundary | L3 TF snapshot |
| `system:tf_4` | `TFVec` | Week boundary | L4 TF snapshot |
| `system:tf_5` | `TFVec` | Month boundary | L5 TF snapshot |
| `system:tf_6` | `TFVec` | Year boundary | L6 TF snapshot |
| `system:head` | `CID` (string) | Commit | Latest commit CID — chain tip |
| `system:global_1` | `HLLSet` | Operation | System-wide aggregation #1 |
| `system:global_2` | `HLLSet` | Operation | System-wide aggregation #2 |
| `system:global_3` | `HLLSet` | Operation | System-wide aggregation #3 |
| `system:layer_0` | `HLLSet` | Ingestion (second) | L0 — current second, active S(t) |
| `system:layer_1` | `HLLSet` | Ingestion (minute) | L1 — completed seconds in current minute |
| `system:layer_2` | `HLLSet` | Ingestion (hour) | L2 — completed minutes in current hour |
| `system:layer_3` | `HLLSet` | Ingestion (day) | L3 — completed hours in current day |
| `system:layer_4` | `HLLSet` | Ingestion (week) | L4 — completed days in current week |
| `system:layer_5` | `HLLSet` | Ingestion (month) | L5 — completed weeks in current month |
| `system:layer_6` | `HLLSet` | Ingestion (year) | L6 — completed months in current year |

### 2.3 Wire Formats (Canonical)

`[PART]` **Status:** HLLSet wire format is implemented. TFVec and Commit wire
formats are not yet in `hllset-core`.

#### HLLSet (CA)

```text
Offset  Size    Field
0       8       Magic: "HLLSET\0\0"
8       4       Version: uint32 LE
12      4       M (register count): uint32 LE (= 1024)
16      4       B (bits per register): uint32 LE (= 32)
20      4096    Register array: 1024 × uint32 LE
4116    *       Optional metadata (currently empty, reserved)

Total: 4116 bytes fixed
CID: h:SHA1(bytes[0..4116])
```

#### TF Vector (Temporal)

```text
Offset  Size    Field
0       4       N (entry count): uint32 LE (= 32768)
4       262144  TF values: 32768 × float64 LE

Total: 262148 bytes fixed
Key: system:tf
```

#### Commit (CA)

```text
Compact JSON with canonical key ordering:
{"d":"<cid>","h":"<cid>","n":"<cid>","r":"<cid>","s":"<cid>","ts":<u64>}

CID: t:SHA1(json_bytes)
```

### 2.4 State Machine

`[SPEC]` The storage state machine is specified in algebraic form. The `Storage`
trait (§2.5) provides a partial implementation; the full state machine is not
yet implemented as a standalone component.

```text
States:

  S = ⟨ store:    Map[CID → Bytes],
        temporal: Map[TmpID → Bytes],
        pinned:   Set[CID] ⟩

Initial state: S₀ = ⟨∅, ∅, ∅⟩

Transitions:

  PUT(cid, bytes):
    precondition: cid = prefix:sha1 ⇒ sha1 = SHA1(bytes)
    store' = store ⊕ {cid ↦ bytes}

  GET(cid):       return store(cid)
  HAS(cid):       return cid ∈ dom(store)
  LIST(prefix):   return [cid | cid ∈ dom(store), cid starts_with prefix]
  PIN(cid):       pinned' = pinned ∪ {cid}
  UNPIN(cid):     pinned' = pinned ∖ {cid}
  GC():           removed = {cid | cid ∈ dom(store), cid ∉ pinned}
                  store' = store ∖ removed
                  return removed
  PUT_TMP(k,b):   temporal' = temporal ⊕ {k ↦ bytes}
  GET_TMP(k):     return temporal(k)
  CAS_TMP(k,o,n): if temporal(k) = Some(o):
                    temporal' = temporal ⊕ {k ↦ n}
                    return true
                  return false
```

### 2.5 The Storage Trait (Rust Binding)

`[IMPL]` The canonical Rust interface for HLPP storage. This is the **trait
boundary** — everything above it is pure domain logic.

```rust
pub trait HlppStorage {
    // CA
    fn put(&self, cid: &str, bytes: &[u8]) -> Result<(), HlppError>;
    fn get(&self, cid: &str) -> Result<Option<Vec<u8>>, HlppError>;
    fn has(&self, cid: &str) -> Result<bool, HlppError>;
    fn list(&self, prefix: &str) -> Result<Vec<String>, HlppError>;
    fn pin(&self, cid: &str) -> Result<(), HlppError>;
    fn unpin(&self, cid: &str) -> Result<(), HlppError>;
    fn gc(&self) -> Result<Vec<String>, HlppError>;
    // Temporal
    fn get_tmp(&self, key: &str) -> Result<Option<Vec<u8>>, HlppError>;
    fn put_tmp(&self, key: &str, bytes: &[u8]) -> Result<(), HlppError>;
    fn cas_tmp(&self, key: &str, old: &[u8], new: &[u8]) -> Result<bool, HlppError>;
}
```

**Implemented backends:** `MemoryStorage` (dev/test), `IpfrsNative` (sled/local)
`[IMPL]`; `RedisStorage` (enterprise) `[IMPL]`.

**The Trait-Boundary Principle:** Isolate infrastructure behind a minimal trait
boundary. Every backend implements the same methods. Everything above the trait
is pure domain logic — none of it knows or cares where bytes live. This is why
`RedisStorage` took 150 lines of Rust to light up the entire framework.

### 2.6 Interface Bindings

`[PART]` **Status:** Rust binding is implemented. Lua CA operations are
implemented; Lua temporal operations are not yet implemented. Forth binding is
parser-only (see §4.4). HTTP binding is specified but not implemented.

#### Lua

```lua
-- CA (implemented)
hllset.store(elem)       -- PUT
hllset.load(key)         -- GET
hllset.exists(key)       -- HAS
hllset.list(prefix)      -- LIST
hllset.pin(key)          -- PIN
hllset.unpin(key)        -- UNPIN
hllset.gc()              -- GC

-- Temporal (to implement)
hllset.get_tmp(key)      -- GET_TMP
hllset.put_tmp(key, val) -- PUT_TMP
hllset.cas_tmp(k, o, n)  -- CAS_TMP
```

#### HTTP (Specified)

```text
GET    /api/v1/hllset/<cid>         → 200 + binary | 404
PUT    /api/v1/hllset/<cid>         → 201 | 409 (mismatch)
HEAD   /api/v1/hllset/<cid>         → 200 | 404
GET    /api/v1/hllset?prefix=h:     → 200 + [cid, ...]
POST   /api/v1/hllset/<cid>/pin     → 200
DELETE /api/v1/hllset/<cid>/pin     → 200
POST   /api/v1/hllset/gc            → 200 + [removed, ...]

GET    /api/v1/temporal/<key>       → 200 + binary | 404
PUT    /api/v1/temporal/<key>       → 200
POST   /api/v1/temporal/<key>/cas   → 200 (true) | 409 (false)
```

### 2.7 IPLD Integration

`[SPEC]` HLPP objects are IPLD nodes. The Commit is a dag-json document. Every
CID field is an IPLD Link — the lattice is a navigable DAG in IPFS.

Commit objects use the dag-json multicodec (`0x0129`). CID references use the
`/` prefix:

```json
{
  "ts": 1719876543210,
  "s":  {"/": "o:a1e7647eb2c601256c..."},
  "h":  {"/": "t:4b38ac2be97210956c..."},
  "d":  {"/": "d:9d8ac7f6d54ba51164..."},
  "r":  {"/": "r:4b38ac2be97210956c..."},
  "n":  {"/": "n:c15d62bb4a11190381..."}
}
```

---

## Part III: Core Concepts

### 3.1 The TF vs Rank Separation Principle

`[IMPL]` **TF is stored. Rank is derived. They are not the same thing.**

This is the single most important distinction in the architecture. Confusing TF
and rank produces reasoning errors. Keeping them separate enables domain-universal
operation.

#### Three Distinct Concepts

| Concept | Level | What it measures | Example |
| --------- | ------- | ----------------- | --------- |
| Token TF | per-token | How often did this word appear? | `tf("hello") = 42` |
| Bit TF | per-position (32,768) | How much activity at this hash bucket? | `TF[1023][17] = sum of tf(t) for all t hashing here` |
| Rank | per-HLLSet (derived) | How important is this HLLSet right now? | `rank(H) = Σ TF[b] for b ∈ H` |

#### The Storage Rule

```text
                 ┌──────────────────────────────┐
                 │      Shared TF Vector        │
                 │      Key: system:tf          │
                 │      32,768 × f64            │
                 │                              │
                 │  Updated ONLY by ingestion   │
                 │  Monotonic (increment only)  │
                 │  CRDT-convergent by IICA     │
                 │  Bit-level, not token-level  │
                 └──────────────┬───────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
    aggregated rank       rank vector          normalized rank
    Σ TF[b] ∀ b∈H        {TF[b] ∀ b∈H}        ΣTF / |H|
    (scalar)              (vector)             (density)

    All computed locally from TF — never stored in protocol
```

| Action | Reads TF? | Writes TF? |
| -------- | :---------: | :----------: |
| `INSCRIBE` (tokenize) | No | **Yes** — increments bit-level TF |
| `UNION` / `INTERSECT` / `DIFF` | No | No — bitmask-only |
| Rank query (any form) | **Yes** — projects onto bit-level TF | No |
| Commit | No — stores DRN CIDs | No |

#### When to Use TF vs Rank

```text
┌─────────────────────────────────────────────────────────────┐
│                     SAME TOKEN BASE                         │
│  Use TF for:                                                │
│    • Distance metrics (KL divergence)                       │
│    • Derivatives (ΔTF, Δ²TF)                                │
│    • Fisher matrix (co-occurrence coupling)                 │
│    • Holographic projection (time lens)                     │
│    • Materialization (LUT disambiguation)                   │
├─────────────────────────────────────────────────────────────┤
│                     DIFFERENT TOKEN BASES                   │
│  Use Rank for:                                              │
│    • Structural similarity (rank correlation)               │
│    • Cross-domain matching (same shape, different content)  │
│    • Hash bridge comparison                                 │
│    • Lattice isomorphism detection                          │
│    • Domain-universal HLLSet classification                 │
│                                                             │
│  TF is INAPPLICABLE across different token bases.           │
│  Rank is MEANINGFUL across different token bases.           │
└─────────────────────────────────────────────────────────────┘
```

**TF requires the same token base.** If two HLLSets were produced by different
hash functions (or the same hash function applied to different token
vocabularies), their bit positions encode different things. Comparing their TF
is meaningless — the numbers are in different spaces.

**Rank is hash-function-agnostic.** It doesn't care what bit position 314 means
in any given domain. It only cares that it's the 3rd most active position.

| | TF | Rank |
| --- | ----- | ------ |
| Type | Continuous scalar | Ordinal position |
| Monotonic? | Yes (CRDT) | Re-sorted on every TF change |
| Domain portability | No (bound to hash function) | Yes (invariant under re-hashing) |
| Information preserved | Magnitude + ordering | Ordering only |
| Used for | Distance, derivatives, Fisher, materialization | Structural similarity, cross-domain matching |

### 3.2 The Five-Level Rank Algebra

`[IMPL]` Rank propagates from raw token frequency to compound lattice operations
through five compositional levels. Each level is a function from its inputs to a
rank value; the levels compose deterministically. All five functions (F, G, H, K,
L, M) are **pluggable design parameters** — the architecture specifies the
framework; the application chooses the functions.

```text
Level 5: compound HLLSet rank    L(max{R}) for union, M(min{R}) for intersection
         ↑
Level 4: HLLSet rank             K(degree in lattice graph)
         ↑
Level 3: register rank           H({bit-R[tz] | tz ∈ 0..31})
         ↑
Level 2: bit rank                G({token-R | all tokens hashing to (reg, tz)})
         ↑
Level 1: token rank              F(TF) — rank derived from token frequency
```

#### Level 1: Token Rank — F(TF)

```text
token-R(t) = F(TF(t))
```

F must be **monotonic** to preserve CRDT convergence. Candidates: identity,
logarithmic, sigmoidal.

#### Level 2: Bit Rank — G({token-R})

```text
bit-R(reg, tz) = G({token-R(t) | hash(t) = (reg, tz)})
```

G aggregates the ranks of all tokens that hash to the same bit position.
Candidates: max (dominant token), sum (all contribute), weighted mean (balanced).

**tz independence:** tz ∈ 0..31 is an address, not a weight. All 32 bit positions
within a register are equal citizens.

#### Level 3: Register Rank — H({bit-R})

```text
reg-R(r) = H({bit-R(r, tz) | tz ∈ 0..31})
```

H aggregates across the 32 bit positions within a register. Candidates: mean,
max-pool, active-only mean, population-weighted.

#### Level 4: HLLSet Rank — K(degree)

```text
hllset-R(H) = K(degree(H), centrality(H), ...)
```

Structural importance in the lattice DAG. Candidates: raw degree,
popcount-weighted degree, PageRank-like iterative propagation.

#### Level 5: Compound Rank — L(max) / M(min)

```text
rank(A ∪ B) = L(max{rank(A), rank(B)})
rank(A ∩ B) = M(min{rank(A), rank(B)})
```

Simplest forms: L = max, M = min.

#### Summary

| Level | Function | Design space | FPGA-native choice | FPGA operations |
| ------- | ---------- | ------------- | -------------------- | ----------------- |
| 1 | F | Identity, log, sigmoid | Identity or integer ⌊log₂(x)⌋ | Load, LZCNT |
| 2 | G | Max, sum, weighted mean | Max or Sum | CMP, ADD |
| 3 | H | Mean, max-pool, active-only | Sum or Max-pool | ADD (32 terms) or CMP tree |
| 4 | K | Degree, weighted degree | Degree (popcount of adjacency row) | POPCOUNT |
| 5 | L, M | Max/min, blended | Max / Min | CMP |

### 3.3 Rank Depletion and the Observable Mask

`[SPEC]` Rank not only builds upward from TF — it also depletes downward as
relative frequencies shift. The five functions (F, G, H, K, L, M) are the same
for depletion as for construction.

**Depletion is driven by baseline shift, not TF decrease.** The TF vector is
monotonic (only increments), so token-R never decreases through TF alone. A
token's relative rank drops because other tokens' TFs rise around it.

**Observable mask:** The set of HLLSets with hllset-R above threshold θ:

```math
O(θ) = {H ∈ H | \text{ hllset-R}(H) > θ}
```

This is a mask over the complete collection. HLLSets enter and exit the
observable set as ranks reshuffle. This is not deletion — every HLLSet remains
content-addressed and retrievable. The mask controls **attention**, not existence.

**Phase boundary at Level 3→4.** Levels 1→2→3 are content-based (internal
composition). Level 4 is structure-based (lattice graph position). These metric
families are correlated but not reducible to each other.

### 3.4 Dimensional Nesting: D_P = N + 2

`[IMPL]` The HLLSet Algebra is a complete algebraic framework for hierarchically
nested world models. Multi-perceptron fusion, robot swarms, and recursive command
hierarchies require **zero new algebraic operations**.

```text
Level 1:  N_s sensors      → D_P = N_s + 2  (a robot)
Level 2:  N_r robots        → D_P = N_r + 2  (a swarm)
Level 3:  N_c command posts → D_P = N_c + 2  (a theater)
...
Level k:  N_k units         → D_P = N_k + 2  (any command level)

Same operations. Same 4KB fingerprints. Same five functions.
```

The five operations used by every level:

| Operation | Bitwise | Meaning |
| ----------- | --------- | --------- |
| ∪ (OR) | OR | Aggregate observations |
| ∩ (AND) | AND | Find shared structure (R-link) |
| \ (AND-NOT) | AND-NOT | Find unique structure |
| popcount | popcount | Measure weight |
| key() | SHA1 | Content-address identity |

**Algebraic guarantees:**

1. **Union closure:** The union of any HLLSets is an HLLSet
2. **Intersection closure:** The intersection of any HLLSets is an HLLSet
3. **Idempotence:** A ∪ A = A and A ∩ A = A — prevents dimensional explosion
4. **Monotonicity:** H(t) ⊆ H(t+1) — the top never shrinks; Noether invariant holds at every level

---

## Part IV: The Architecture

### 4.1 The Evolution Equation

`[IMPL]` The fundamental equation of system evolution:

```text
H(t) = H( S(t), H(t-1), D(t-1), R(t-1), N(t) )

where:
  S(t)     = current scan — new observation as an HLLSet
  H(t-1)   = previous lattice state
  D(t-1)   = Departed   = H_prev - H_curr
  R(t-1)   = Retained   = H_prev ∩ H_curr
  N(t)     = New        = H_curr - H_prev
```

D, R, N are themselves HLLSets. The evolution record IS an HLLSet.

### 4.2 The Temporal Pyramid (L0–L6)

`[SPEC]` **Code status:** No crate-level implementation. The pyramid is fully
specified here. Notebook 06 demonstrates the concept against the CLI binary.

#### The Default Pyramid

```text
Layer 6  YEAR     L6 = ∪ S(t) over 365 days          ← coarsest
Layer 5  MONTH    L5 = ∪ S(t) over 30 days
Layer 4  WEEK     L4 = ∪ S(t) over 7 days
Layer 3  DAY      L3 = ∪ S(t) over 24 hours
Layer 2  HOUR     L2 = ∪ S(t) over 60 minutes
Layer 1  MINUTE   L1 = ∪ S(t) over 60 seconds
Layer 0  SECOND   L0 = ∪ S(t) over current second    ← finest

Total coverage: 7 layers → ~1 year of compressed history
```

#### Automatic Building (Union Aggregation)

The pyramid builds itself mechanically:

```text
Every second boundary:
  L1 = L1 ∪ L0          // previous second absorbed into minute
  L0 = ∅               // reset for next second

Every minute boundary:
  L2 = L2 ∪ L1          // previous minute absorbed into hour
  L1 = ∅

...and so on up to L6 (year)
```

After compression, layers are **mutually exclusive** — no time slice appears
in more than one layer. The complete system state is their union:

```text
H_system(t) = L0 ∪ L1 ∪ L2 ∪ L3 ∪ L4 ∪ L5 ∪ L6
```

The union is **bit-lossless** — every bit from every S(t) survives. What is lost
is **temporal differentiation**: you cannot recover which second within a minute
a bit came from. Every original S(t) HLLSet remains stored in IPFS at its own CID.

#### Compression Ratios

```text
  L0 → L1:  60:1    (seconds → minute)
  L1 → L2:  60:1    (minutes → hour)
  L2 → L3:  24:1    (hours → day)
  L3 → L4:   7:1    (days → week)
  L4 → L5:  ~4:1    (weeks → month)
  L5 → L6:  12:1    (months → year)

  Total pyramid: 60×60×24×7×4×12 ≈ 14.5 million seconds compressed into 1 HLLSet
```

#### Configurable Pyramid

The 7-layer second→year pyramid is **one instance** of a general sliding window.
The pyramid shape is a tunable parameter: number of layers N and their durations
[d₀, d₁, ..., d_{N-1}].

| Application | N | [d₀..d_{N-1}] | Total span | Character |
| ------------ | --- | --------------- | ------------ | ----------- |
| High-frequency trading | 5 | 100ms each | 500ms | Micro-burst detection |
| Real-time control | 4 | 250ms each | 1s | Fast reflex, no deep history |
| Conversational agent | 10 | 6s each | 1min | Sentence-to-sentence coherence |
| Document analysis | 6 | 10min each | 1hr | Section-level context |
| Original (default) | 7 | [1s, 1min, 1hr, 1d, 1w, 1mo, 1yr] | 1yr | Long-term memory |

Nothing else in the architecture changes — Noether, Fisher, rank algebra all
operate identically.

**Design principle.** The pyramid is not a calendar. It is a configurable sliding
window whose depth and granularity are chosen per application.

#### The Noether Invariant

```math
⋃_{i=0}^{6} L_i = \text { constant over time }
```

**Symmetry:** multiple paths through the pyramid (L0→L1→L2 or L0→L2 directly).
**Conserved:** total information in the union of all layers.
**Guarantee:** H_system converges regardless of path.

This IS eventual consistency — not as a protocol we must implement, but as a
property that falls out of the structure. No consensus algorithm. No leader
election. The lattice converges because the union is monotonic and multiple
paths guarantee every bit eventually reaches every layer that needs it.

### 4.3 D/R/N Decomposition

`[IMPL]` The D/R/N split is the attention filter. For every arriving observation
S(t), compute against the previous state H(t-1):

```text
N(t) = S(t) \ H(t-1)      ← new information
D(t) = H(t-1) \ S(t)      ← departed information
R(t) = S(t) ∩ H(t-1)      ← retained (already known)
```

Three popcounts. O(1) per scan. The R-link IS the relationship — storable as
`r:<sha1>`, composable, content-addressed.

**Code location:** `hllset-ranks/src/derivatives.rs:76–87` — ephemeral
computation only; R-links are not yet persisted.

### 4.4 R-Links: Replacing BSS with Topological Intersection

`[IMPL]` in `hllset-ranks`; `[SPEC]` for system-wide adoption.

**Resolution:** Previous documents used BSS (Bell State Similarity) — a scalar
float τ = |A ∩ B| / |B|. Section 12 of the bible replaced BSS with R-links
(topological intersection). The standard adopts this: **R-links are the
architectural direction. BSS remains in code as a convenience function but
is not the primary comparison mechanism.**

```text
BEFORE (scalar BSS):
    τ = |A ∩ B| / |B|     floating-point division, multi-cycle
    Compare τ > threshold   scalar comparison

AFTER (topological R):
    R = A ∩ B               bitwise AND across 1024 registers, single-cycle
    weight = popcount(R)    count set bits, single-cycle
    Select: f(rank, weight) integer arithmetic only
```

| Property | BSS (scalar) | R-link (topological) |
| ---------- | ------------- | --------------------- |
| Output type | float in [0,1] | HLLSet (1024×32 bits) |
| Storable | No (ephemeral) | Yes (content-addressed) |
| Composable | No | Yes — R can be intersected with C |
| FPGA-native | No (division) | Yes (AND + popcount) |
| Information | 1 number | 32,768 bits of relationship structure |

#### R-Links as Feedback Gates

R-links select what to feed back into the main loop:

```text
S(t) → for each layer L_i:
         R = S(t) ∩ L_i                          // compute R-link
         if popcount(R) > θ:                     // is it relevant?
             materialize(L_i)                    // yes, feed it back
```

The R-link weight IS the relevance score. No separate relevance model. No
attention mechanism. Just bitwise AND + popcount.

#### The BSS Convergence Signal

`[PART]` BSS remains useful as a **system-level convergence signal** (not as a
per-comparison operation). It's more suitable for the actuator scope where a special calculators can be implemented:

```text
t=1: BSS(mixed_stream, reference) ≈ 0.2   # crude, few tokens in memory
t=2: BSS(mixed_stream, reference) ≈ 0.5   # sharpening
t=3: BSS(mixed_stream, reference) ≈ 0.8   # converging toward understanding
```

### 4.5 Fire-and-Forget Communication

`[IMPL]` at the conceptual level; `[PART]` in code (mesh is single-process).

Each state sends output to exactly two destinations — no coordination:

```text
Window W(t) = { S0(now), S1, S2, S3, S4(deepest) }
                  |      |    |   |   |
                  v      v    v   v   v
          [ MATERIALIZER ] ← collects ALL outputs
               |
          S0 → S1 → S2 → S3 → S4
          (aggregation chain, lossy OK)
```

**Key properties:**

- Each state Si sends to the materializer (fire-and-forget, always succeeds — content-addressed)
- Each state Si sends to Si+1 for aggregation (lossy tolerated)
- Si+1 may miss Si's output — the materializer still received it directly
- The materializer eventually has the complete picture via union
- Possible because HLLSets are idempotent, content-addressed, and CRDT-mergeable

### 4.6 The Noether Controller

`[SPEC]` No crate-level implementation.

The Noether controller monitors the cross-layer relationship matrix and decides
**which layer** should drive the next action:

```text
if L0 (second) ↔ L1 (minute) τ < 0.5:
    → DIVERGENCE detected
    → Override instinct with L1 context

elif L0 (second) ↔ L3 (day) τ < 0.3:
    → RE-ROUTE: current input diverging from long-term goal
    → Restore direction with L3

else:
    → stable: instinct aligned with context and trajectory
    → Continue with L0
```

#### The Full Controller Surface

```text
Controller = {
    pyramid:    { layers: N, durations: [d₀..d_{N-1}] },
    ranks:      { F, G, H, K, L, M },
    attention:  { θ (threshold) },
    steering:   { bit_threshold, rank_threshold },
}
```

Six knobs. Same operations. Any scale.

### 4.7 Rank-Based Learning

`[IMPL]` HLLSets are immutable. Only ranks change. That IS learning.

```text
HLLSets (fixed, content-addressed)
    │
    ▼
Forth Dictionary ──→ assigns RANKS ──→ THIS is learning
    │
    ▼
Behavior = highest-ranked HLLSet drives action
```

| Component | What it does | Does it learn? |
| ----------- | ------------- | :---: |
| **Tokenizer** | bytes → HLLSet | No (deterministic function) |
| **Materializer** | collects all HLLSet outputs | No (passive observer) |
| **Forth Dictionary** | assigns ranks to HLLSets | **Yes** (only this) |

**The LUT is the dynamic bridge.** A bit position in an HLLSet references a
(reg, tz) bucket in the TokenLUT. The HLLSet stores the bitmask; the LUT stores
the actual token collections and their current TF values. An HLLSet's rank is
computed at query time:

```text
rank(HLLSet) = sum( LUT[reg][tz].TF for all (reg, tz) where bit == 1 )
```

This pushes HLLSets up or down in ranking without touching them. An HLLSet
created years ago can rise in relevance today because the LUT changed around
it — new slang makes its bit positions hot; stale positions quietly fall.

### 4.8 The Forth Dictionary

`[PART]` **Code status:** Parser → AST → `compile_to_lua` only. Colons are
parsed but not lowered. Advertised backends (`lower_rust`, `lower_hw`) do not
exist. The dictionary is not yet the seat of learning — it is a syntax layer.

The Forth hllang serves as the **canonical AST** that unifies all backends:

```text
                    ┌────────────────────────┐
                    │    Forth Source (write)│
                    └──────────┬─────────────┘
                               ▼
                    ┌────────────────────────┐
                    │   Forth AST (canonical)│
                    └──────────┬─────────────┘
                               │
          ┌────────────────────┼──────────────────┐
          ▼                    ▼                  ▼
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │  Lower: Lua  │    │ Lower: Rust  │    │ Lower: HW    │
   │  (software)  │    │ (software)   │    │ (FPGA)       │
   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
          └───────────────────┼───────────────────┘
                              ▼
                    ┌─────────────────────┐
                    │   HLPP (IPFS)       │
                    └─────────────────────┘
```

### 4.9 System Lifecycle

`[SPEC]` Systems are mortal. They develop rank bubbles. Instead of tweaking
a running system, you let it live — and spawn a new one.

```text
BIRTH:  Seed HLLSets + initial lattice (ranks)
   │
   ▼
LIFE:   Tokenizer → HLLSets
        Forth → reshuffles ranks (learns)
        Materializer → collects outputs
        Ranks inevitably develop bubbles
   │
   ▼
DEATH:  Don't fix it. Don't tweak it.
   │
   ▼
REPRODUCE:  Copy HLLSets + lattice → new system
            Fresh start, accumulated knowledge
            No rank bubbles. No pathologies.
```

**Why reproduction works (IICA properties):**

- Immutable: HLLSets never change — safe to copy, identical in any system
- Idempotent: Copy twice = same result — IPFS deduplicates automatically
- Content-Addressed: Every HLLSet has a CID — transfer = `ipfs get <cid>`

### 4.10 Actuation

`[IMPL]` in Lua materializer; `[PART]` in Rust (De Bruijn edge construction
has known defects — see §7).

The materializer produces candidate tokens. But the real world needs ORDER.
The actuator restores order for the target modality:

| Modality | Token structure | Order encoding | Actuation strategy |
| ---------- | ---------------- | ---------------- | ------------------- |
| **Text** | n-grams with boundary pads | Overlap adjacency | DeBruijn (Eulerian path) |
| **Images** | Spatial patches (HxW) | Patch coordinates | 2D layout reconstruction |
| **Audio** | Spectral bins (Mel) | Temporal frames | Overlap-add windowing |
| **DNA** | k-mers (fixed length) | Overlap by k-1 | DeBruijn (standard) |
| **Robotics** | Action primitives | Temporal sequence | Plan execution ordering |

The actuator sits AFTER the materializer. It does NOT feed back into the loop.

### 4.11 Holographic Memory

`[IMPL]` The lattice top H_system = ∪ L_i implicitly contains every HLLSet ever
observed. The ordered sequence of TF vectors forms a temporal stack:

```math
past_state(t) ≈ H_system(now) ⊙ TF_stack[t]
```

Where ⊙ means: project each HLLSet through the TF vector at time t to derive its
approximate rank at that time. The TF vector acts as a **time lens**.

**Practical implication:** You don't need to store every historical lattice
snapshot. Store (1) the current lattice top (one HLLSet, 4KB) and (2) the TF
stack (262KB per time step, compressible). From these, approximate any past state.

### 4.12 The Lattice as an Optimization Surface

`[SPEC]` A HLLSet is a measurement, not a context container. The task is not to
find the best HLLSet — it's to find the best **cover**.

**Minimal Cover with Maximal Rank:**

Find $C ⊆ L$ such that:

- $⋃_{H ∈ C} H ⊇ H_{system}$    (cover)
- $Σ_{H ∈ C} \text{ rank}(H)$ is maximized    (quality)
- $|C|$ is minimized    (parsimony)

At each evolution step, the system takes one gradient step toward a better cover.
It never arrives — the lattice keeps evolving.

**Temperature T** controls exploration vs exploitation:

- T → 0 (cold): Only high-rank HLLSets. Stick to what's proven.
- T → ∞ (hot): Any HLLSet can be selected. Explore the dictionary.
- T(t): Annealing schedule. Start hot, cool down as system stabilizes.

### 4.13 The Lattice as a Neural Network

`[SPEC]` The rank vector of any HLLSet H is the element-wise product of the TF
vector and the HLLSet's bitmask:

```math
\text{rank}(H) = TF ⊙ \text{bitmask}(H)
```

This is structurally identical to a neuron: bitmask = weights, TF = activation
signal, rank = output. But the "weights" are content-addressed and immutable.

| Concept | Classical NN | HLLSet Lattice |
| --------- | ------------- | ---------------- |
| Weights | Float matrix W | Binary bitmask(H) — content-addressed |
| Activation | σ(Wx + b) | TF ⊙ bitmask(H) — CRDT-convergent |
| Forward pass | Matrix multiply + nonlinearity | Bitwise AND + popcount |
| Learning rule | Backpropagation (∂L/∂W) | Bitmask selection in dictionary |
| Gradient | Continuous ∂L/∂w_ij | Binary Jacobian J_ij ∈ {0, 1} |
| Convergence | Local minima of loss surface | CRDT convergence (monotonic union) |

### 4.14 Noether Steering with Fisher Matrix

`[SPEC]` The Noether steering equation is a conservation law on bit count:

```math
|\text{card}(N(t)) - \text{card}(D(t-1))| → 0
```

**Rank-weighted steering** (stronger condition):

```math
|Σ_{b∈N(t)} R_b(t) - Σ_{b∈D(t-1)} R_b(t-1)| → 0
```

**Fisher-like cross-layer matrix:**

```math
F_{bb'} = Σ_{i=0}^{6} B^{(i)}_b · B^{(i)}_{b'}
```

Counts how many layers have both bits b and b' set simultaneously. This
enables the controller to distinguish isolated fluctuations from systemic
phase transitions.

### 4.15 Content Novelty Regimes

`[IMPL]` **Scan frequency doesn't matter. Content change does.**

| Regime | Rate | Content | Result |
| -------- | ------ | --------- | -------- |
| **Deep** | any | stable | All layers form. Environment is consistent. |
| **Adaptive** | slow | changing | Layers form between changes. |
| **Reactive** | fast | changing | L0 + L1 only. Change outruns window. |
| **Reflexive** | flood | chaotic | Perpetual L0. No layer survives. |

**The structural guarantee:** the system is only as unstable as its environment.
In a static world, it converges. In a changing world, it adapts. In a chaotic
world, it fragments. The system doesn't have an internal failure mode — it has
an environmental response curve.

### 4.16 Intelligence as Window Depth

`[SPEC]` Ashby's homeostat (1948) is the ancestral architecture. The core
insight: **intelligence is operational memory depth.**

```text
window_size = 1  →  Ashby's homeostat  →  pure L0
window_size = 2  →  can compare now with recent past
window_size = 3  →  can see patterns across time
window_size = 4  →  can hold long-term direction
window_size = 5  →  can anticipate future
```

**Intelligence is not a special capability. It is operational memory depth.**
Any system with enough layers will exhibit increasingly intelligent behavior.
The architecture doesn't change. The window just needs to be wider.

---

## Part V: The Universal Bridge

### 5.1 Two-Pass Ingestion

`[IMPL]` Every external input enters the HLLSet Algebra twice.

#### **Pass 1: Representation**

```text
domain_input → murmurhash3 → H_src (lives in source domain's bit space)
```

#### **Pass 2: Re-Representation**

```text
H_src's active bit positions → formatted as tokens → murmurhash3 → H_bridge
                               ("reg:314:tz:17", "reg:8912:tz:3", ...)
```

H_bridge lives in the **target domain's bit space**. It is a citizen of the
target lattice. BSS, R-links, union, intersection with all target-domain
HLLSets work directly — no cross-domain translation layer.

| | Pass 1 (H_src) | Pass 2 (H_bridge) |
| --- | --------------- | ------------------- |
| Bit space | Source domain | Target domain |
| Materialize via source LUT | ✓ (recovers original) | ✗ (different token base) |
| Materialize via target LUT | ✗ (different token base) | ✓ (structural interpretation) |
| BSS with target HLLSets | ~0 (different seeds/spaces) | ✓ (same bit space) |
| BSS with source HLLSets | ✓ | ~0 |

### 5.2 3-Gram Structural Fingerprinting

`[IMPL]` A 3-gram HLLSet is built from all consecutive token triples. It encodes
both adjacency patterns AND vocabulary. The 3-gram HLLSet is the structural
invariant — two texts in different languages with similar discourse structure
produce 3-gram HLLSets with correlated rank distributions.

### 5.3 The Bridge Algorithm

`[SPEC]` for the full implementation. Currently demonstrated in notebooks.

```text
algorithm bridge(source_HLLSet, target_lattice):
    # Pass 2: re-represent source into target bit space
    H_bridge = re_represent(source_HLLSet)

    # Extract 3-gram structural fingerprints
    S_3gram = extract_3gram(source_HLLSet)
    B_3gram = extract_3gram(H_bridge)

    # Rank-correlate against all HLLSets in target lattice
    candidates = []
    for H_target in target_lattice:
        T_3gram = extract_3gram(H_target)
        ρ = spearman_rank_correlation(B_3gram, T_3gram)
        if ρ > threshold:
            candidates.append((H_target, ρ))

    # Select minimal cover with maximal rank correlation
    cover = select_cover(candidates, S_3gram)

    return {
        bridge: H_bridge,
        cover: cover,
        top_match: candidates[0]
    }
```

### 5.4 Domain LUTs

`[IMPL]` Each domain owns its own Token-LUT. Materialization always goes through
the LUT of the HLLSet's native domain.

```text
Source HLLSet → source_LUT → source tokens (approximates original)
Bridge HLLSet → target_LUT → target tokens (structural interpretation)
```

Domain LUTs are independent. They don't interfere. They don't need to agree.

### 5.5 The Statistics Constraint

`[SPEC]` **This is a design rule, not a code issue.**

When bridging between nested HLLSet lattices, **frequencies and ranks from the
outer lattice cannot be transferred to the inner lattice.** Each lattice must
learn its own statistics through its own experience.

| Property | Bridge transfers | Bridge does NOT transfer |
| ---------- | :---: | :---: |
| Bit positions (H_src → H_bridge) | ✓ | — |
| Content keys | ✓ | — |
| TF vectors | — | ✗ Must be learned independently |
| Rank orderings | — | ✗ Domain-specific |
| Temporal pyramid state | — | ✗ Experience-accumulated |
| G1/G2/G3 globals | — | ✗ Each universe has its own |

> **The bridge transfers structure (bit positions), not statistics (counts).**

```text
Universe 1 (source)          Universe 2 (CAAL + I Ching)
─────────────────────        ─────────────────────────────
o-HLLSet pool (source)       o-HLLSet pool (CAAL vocab)
Source LUT                    CAAL LUT          ← independent
Source TF vectors             CAAL TF vectors   ← independent
Source ranks                  CAAL ranks        ← independent
Source temporal pyramid       CAAL pyramid      ← independent
```

### 5.6 Disambiguation: Closing the Loop (R-R → R)

`[SPEC]` The gate into the sub-lattice is one-way compression. The reverse path
— disambiguation — expands the sub-lattice's output back into the parent
vocabulary:

```text
R → R-R (bridge: compress into sub-lattice)
  → sub-lattice deliberation
  → R-R → R (disambiguate: tokenize output with source tokenizer)
  → H_src_output (in parent bit space, full vocabulary)
  → FEED BACK: H_src_output ∪ H_src_next
```

Disambiguation is itself an IICA morphism. **Structure transferred; statistics
NOT transferred.**

### 5.7 Bridges Are Not Special

`[IMPL]` The universal bridge, vocabulary compression, nested sub-lattices,
disambiguation — none require new algebra. They are all the **same ingestion
pattern with different token definitions:**

```text
                            ┌─ Source tokens → H_src
                            │
Any input ──→ Tokenizer ────┼─ Bit-position labels → H_bridge
                            │
                            ├─ Sub-vocabulary → H_sub
                            │
                            └─ Output text → H_src_output
                                     │
                                     ▼
                            Same murmurhash3
                            Same HLLSet::from_tokens()
                            Same five operations
```

What changes is the **token definition** — what string you feed to the hash
function. The architecture emerges from how you tokenize, not from new code.

---

## Part VI: Self-Ingestion & Development

### 6.1 Codebase as Lattice Input

`[SPEC]` The system's own source code is a data stream — it evolves through
edits, refactors, and feature additions. The codebase can be ingested into the
HLLSet lattice.

**Trigger: git commit.** Each committed file is tokenized into an HLLSet,
content-addressed, and stored. The commit becomes a lattice event with its own
D/R/N decomposition — which files were added (N), modified (retained with drift),
or deleted (D).

| Query | Mechanism |
| ------- | ----------- |
| Which files changed most this sprint? | Rank velocity Δ²R over commit history |
| Which files tend to be edited together? | Fisher matrix: F(file_a, file_b) across commits |
| Is the codebase in a refactoring or stable phase? | Noether steering: \|N\| vs \|D\| across commits |
| Which modules are "hot" right now? | Observable mask O(θ) over file ranks |

### 6.2 LLM Context Views (llms.txt + l: prefix)

`[SPEC]` Each code directory gets an `llms.txt` file — a human-written (or
auto-generated) annotation that describes what the code does. The `llms.txt`
is ingested with the `l:` prefix.

A **folder view** (v: prefix) aggregates all code files in a directory plus
its `llms.txt`:

```text
crates/hllset-storage/
├── llms.txt               → l:<sha1>   LLM context
├── src/
│   ├── lib.rs             → h:<sha1>   code
│   ├── storage.rs         → h:<sha1>   code
│   └── ...
└── [view]                 → v:<sha1>   union(lib, storage, ..., llms.txt)
```

**Query flow for prompt → code matching:**

```text
1. User prompt: "connect to the database"
2. tokenize(prompt) → HLLSet P
3. Phase 1 — LLM context scan:
     for each l:<sha1> in lattice:
         τ = BSS(P, l:llms)
     if τ > 0.5: that directory is semantically relevant
4. Phase 2 — Folder view refinement:
     for top-K directories from Phase 1:
         τ = BSS(P, v:folder_view)
     return top matches with their code files
5. AI coder receives: matched llms.txt + matched code files + prompt
```

**Auto-generation:** The post-commit hook extracts doc comments (`//!`, `///`)
from changed files and regenerates `llms.txt` automatically. Same commit that
changes code also regenerates the semantic index.

### 6.3 Trait-Boundary Principle

`[IMPL]` Isolate infrastructure behind a minimal trait boundary. Every backend
implements the same interface. Everything above the trait is pure domain logic.
When `RedisStorage::connect("redis://...")` passed its first PING, the entire
HLLSet lattice stack was already running against it — 212 tests, 0 failures
— because the trait boundary had already been paid for.

### 6.4 Unified Development Interface

`[SPEC]` A single Textual terminal TUI with three frames: Prompt, Response,
Files — integrating DeepCode interaction + HLLSet ingestion + git commit in
one workflow. Self-reflection is standard ingest pipeline (no special
`metadata.json`).

---

## Part VII: Implementation Status Matrix

This matrix maps every concept in this standard to its current implementation
state in hllset-next. Based on the `HLLSET_NEXT_REVIEW.md` audit (July 21, 2026).

### 7.1 Core Data Structures

| Concept | Status | Crate(s) | Notes |
| --------- | -------- | ---------- | ------- |
| HLLSet bitmap (1024×32) | `[IMPL]` | hllset-core | Core `HLLSet` struct, serialization, content keys |
| murmurhash3 tokenization | `[IMPL]` | hllset-core | Deterministic hash → (reg, tz) |
| Content addressing (SHA1) | `[IMPL]` | hllset-core | `h:<sha1>` and `c:<sha1>` only; o/r/d/n/t/v/l not yet |
| TFVec wire format | `[PART]` | — | Specified, not yet in hllset-core |
| Commit struct | `[PART]` | — | Specified, not yet in hllset-core |
| TokenLUT / CatalogLUT | `[IMPL]` | hllset-materialize | Multi-seed consensus, monotonic TF tracking |
| BSS (inclusion/exclusion) | `[IMPL]` | hllset-core | Float-based; architectural direction is R-links |
| Jaccard similarity | `[IMPL]` | hllset-core | Float-based |
| HT cardinality estimation | `[IMPL]` | hllset-core | Float-based |

### 7.2 Storage

| Concept | Status | Crate(s) | Notes |
| --------- | -------- | ---------- | ------- |
| `Storage` trait (6 methods) | `[IMPL]` | hllset-storage | The best seam in the codebase |
| `MemoryStorage` | `[IMPL]` | hllset-storage | Dev/test |
| `IpfrsNative` (sled) | `[IMPL]` | hllset-storage | Local persistent |
| `RedisStorage` | `[IMPL]` | hllset-storage-redis | Enterprise, ~150 lines |
| Temporal ops (get_tmp, put_tmp, cas_tmp) | `[PART]` | — | Specified by HLPP, not yet in trait |
| `ipfrs-core` path dependency | `[INACC]` | hllset-storage | Absolute path — blocks standalone builds |
| Lua CA bindings | `[IMPL]` | hllset-dsl | store/load/exists/list/pin/unpin/gc |
| Lua temporal bindings | `[PART]` | — | get_tmp/put_tmp/cas_tmp not yet |

### 7.3 Rank Algebra

| Concept | Status | Crate(s) | Notes |
| --------- | -------- | ---------- | ------- |
| Five-level rank (F, G, H, K, L, M) | `[IMPL]` | hllset-ranks | Integer-only, real tests |
| Rank derivatives (ΔR, Δ²R) | `[IMPL]` | hllset-ranks | D/R/N decomposition for rank flux |
| D/R/N ephemeral computation | `[IMPL]` | hllset-ranks | Not yet persisted as r:/d:/n: |
| Per-register TF ranking | `[PART]` | hllset-ranks | Algorithm specified, partial impl |
| Cross-layer BSS matrix | `[SPEC]` | — | No implementation |

### 7.4 Temporal Pyramid

| Concept | Status | Crate(s) | Notes |
| --------- | -------- | ---------- | ------- |
| L0–L6 layer structure | `[SPEC]` | — | No crate; demonstrated in notebook 06 |
| Carry mechanism | `[SPEC]` | — | Union aggregation at time boundaries |
| Configurable pyramid (N, d_i) | `[SPEC]` | — | Specified, not implemented |
| TF stack (per-layer TF snapshots) | `[SPEC]` | — | system:tf_0 through system:tf_6 |
| Layer clock | `[SPEC]` | — | DRN at every carry boundary |

### 7.5 Architecture

| Concept | Status | Crate(s) | Notes |
| --------- | -------- | ---------- | ------- |
| Evolution equation H(t) | `[IMPL]` | — | Conceptual; used across codebase |
| Fire-and-forget model | `[PART]` | hllset-mesh | Single-process broadcast bus; not distributed |
| Noether controller | `[PART]` | hllset-mesh | Float-based (0.9 decay); differs from ranks (integer) |
| System lifecycle | `[SPEC]` | — | Specified, not implemented |
| Holographic memory | `[IMPL]` | — | Conceptual; not a separate crate |
| Actuation (DeBruijn) | `[PART]` | hllset-dsl | Greedy DFS with 1000-step cap; known edge construction defect |
| Content-addressable PC | `[SPEC]` | — | argmax(BSS(input, word)) — not in code |

### 7.6 Forth / DSL

| Concept | Status | Crate(s) | Notes |
| --------- | -------- | ---------- | ------- |
| Forth parser | `[IMPL]` | hllset-forth | Parser → AST |
| Forth → Lua lowerer | `[IMPL]` | hllset-forth | `compile_to_lua` |
| Colon-definition mechanism | `[PART]` | hllset-forth | Parsed but not lowered |
| Forth → Rust lowerer | `[SPEC]` | — | Advertised, does not exist |
| Forth → Verilog lowerer | `[SPEC]` | — | Advertised, does not exist |
| SNOBOL-inspired Pattern | `[IMPL]` | hllset-dsl | Composable pattern matching |
| Tokenizer pipeline | `[IMPL]` | hllset-dsl | n-gram tokenization |

### 7.7 Mesh / Distribution

| Concept | Status | Crate(s) | Notes |
| --------- | -------- | ---------- | ------- |
| In-process mesh (broadcast) | `[PART]` | hllset-mesh | tokio broadcast; each CLI creates own bus |
| Distributed mesh | `[SPEC]` | — | Not implemented |
| Mesh tests | `[PART]` | — | Zero tests for hllset-mesh and hllset-cli |

### 7.8 Bridge

| Concept | Status | Crate(s) | Notes |
| --------- | -------- | ---------- | ------- |
| Re-representation (Pass 2) | `[IMPL]` | — | Demonstrated in notebook; no dedicated crate |
| 3-gram structural fingerprinting | `[IMPL]` | — | In notebook pipeline |
| Domain LUTs | `[IMPL]` | hllset-materialize | Per-domain token LUT |
| hllset-bridge crate | `[SPEC]` | — | re_represent.rs, ngram.rs, rank.rs, cover.rs, lut.rs |
| Statistics constraint | `[SPEC]` | — | Design rule, not code |

### 7.9 Self-Ingestion

| Concept | Status | Crate(s) | Notes |
| --------- | -------- | ---------- | ------- |
| Git commit → ingest pipeline | `[SPEC]` | — | Specified, not implemented |
| llms.txt + l: prefix | `[SPEC]` | — | Specified, not implemented |
| Folder views (v: prefix) | `[SPEC]` | — | Specified, not implemented |
| Unified dev TUI (Textual) | `[SPEC]` | — | Specified, not implemented |

### 7.10 Known Defects (from HLLSET_NEXT_REVIEW.md)

| Defect | Location | Severity |
| -------- | ---------- | ---------- |
| `to_bytes` swallows serialization errors via `unwrap_or_default` | hllset-core | High — silent corruption risk for content keys |
| De Bruijn edge construction inserts bogus second edge | hllset-dsl | High — reconstruction errors |
| `ChunkMaterializer::open` always returns `Err` | hllset-duckdb | Medium — broken stub |
| `hllset-duckdb` is actually SQLite | hllset-duckdb | Low — misleading name |
| `hllset-storage/src/ipfs.rs` has no IPFS I/O | hllset-storage | Low — misleading name |
| `key_cid_index` computed but never populated | hllset-storage | Low — dead code path |
| Production panics: `ngrams` assert, `with_seeds` assert | hllset-dsl | Low — should be Result |
| `Materializer` ignores `positions` argument | hllset-dsl | Low |
| Hash layout: reg from low 10 bits (non-standard HLL convention) | hllset-core | Documentation — must be noted |

### 7.11 What Is Solid (do not break)

From the review:

- `hllset-core`: bitmap tensor + ops + BSS + content keys; good property tests
- Tokenizer/pattern pipeline in `hllset-dsl` (SNOBOL-inspired, composable)
- `TokenLUT`/`CatalogLUT` (multi-seed consensus)
- `Storage` trait boundary — the best seam in the codebase
- `hllset-ranks`: integer-only five-level algebra with real tests

---

## Part VIII: caal-llm Redesign Guide

### 8.1 Current caal-llm Architecture

The current implementation (July 2026) has 5 crates:

| Crate | LOC (est.) | Purpose |
| ------- | ----------- | --------- |
| `caal-core` | ~400 | Domain-agnostic core: TokenLUT, globals, materialize, retrieve, ngram |
| `caal-zh` | ~150 | Chinese-specific: wraps caal-core with CAAL ZH vocabulary |
| `caal-iching` | ~200 | I Ching: hexagram R-link matrix, consultation, navigation |
| `caal-pipeline` | ~100 | Pipeline orchestration: scene → KB → consultation → feedback |
| `caal-py` | ~500 | Python bindings (PyO3) for all crates |

### 8.2 What caal-llm Currently Depends On

caal-llm depends on `hllset-core` only — no internal hllset-next dependencies.
This was a deliberate choice to avoid the `ipfrs-core` path dependency problem.
**Keep it this way until HLPP Phase 1 is complete** (see §8.12).

### 8.3 Design Mandate

caal-llm is not just "an application that works." It is the **reference
application** for the HLLSet Algebra ecosystem. Every design decision in
caal-llm must answer: "Does this demonstrate the best way to build an
HLLSet Algebra application?"

The mandate has five requirements:

| # | Requirement | Consequence |
| --- | ------------- | ------------- |
| **R1** | **Reference quality.** caal-llm demonstrates best practices in architecture, design, and development. Code that works is not enough — code must be exemplary. | Traits are minimal and well-documented. Crate boundaries are clean. Error handling is explicit. Tests demonstrate patterns for downstream consumers. |
| **R2** | **Upstream-first prototyping.** When hllset-next lacks a module (e.g., temporal pyramid, bridge crate), caal-llm implements a local prototype, then contributes it upstream once proven. | caal-llm contains `[PROTO]`-marked modules that are candidates for extraction into hllset-next crates (§8.9). |
| **R3** | **Modular and extensible.** caal-llm defines the standard for domain extensions. Any future domain (sensor data, DNA, audio) must plug in through the same trait boundaries. | Extension traits are first-class (§8.8). The I Ching domain is one implementation of the `Domain` trait — not a special case. |
| **R4** | **Storage-agnostic with Redis default.** caal-llm works with any storage backend implementing the `Storage` trait. Redis is the practical default for users who want a real database. | Storage selection is configuration, not compilation. The architecture never couples to a specific backend (§8.7). |
| **R5** | **Multi-backend ready.** The architecture anticipates new storage backends (S3, PostgreSQL, IPFS-native) as they become available in hllset-next. Adding a backend is implementing a trait. | The storage abstraction is a passthrough to hllset-next's `Storage` trait — no caal-llm-specific storage code (§8.7). |

### 8.4 Redesign Principles

1. **caal-llm is an application of hllset-next, not a fork.** It should not
   reimplement anything that hllset-next provides. If something is needed,
   it goes in hllset-next first — through the upstream-first path (§8.9).

2. **caal-llm depends only on `[IMPL]` capabilities.** Spec-level concepts
   (temporal pyramid, Noether, lifecycles) remain as future extensions. For
   gaps in hllset-next, caal-llm prototypes under `[PROTO]` markers (§8.9),
   then contributes upstream once stable.

3. **Each universe has its own LUT, TF, and ranks.** The statistics constraint
   §5.5 applies: the CAAL universe and the I Ching sub-universe maintain
   independent statistics. The bridge transfers structure only.

4. **Trait boundaries are the pattern.** Just as hllset-next separates
   `Storage` behind a trait, caal-llm separates every extension point behind
   a trait boundary. No module knows about the implementation details of
   any other module — only the trait it satisfies.

5. **Storage is a configuration choice, not a compile-time decision.**
   The application selects its backend at startup. Code above the storage
   layer never imports a specific backend crate. Adding a new backend
   requires zero changes to application logic.

6. **Extensions are implementations of traits, not modifications of core.**
   A new domain (e.g., English bridge, DNA analysis, sensor fusion) is a
   new crate implementing `Domain` + `Tokenizer` + `Bridge`. It drops in.
   Nothing in `caal-core` changes.

### 8.5 Recommended caal-llm Architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│                        caal-pipeline                             │
│              (orchestration, not logic)                          │
└────────┬─────────────────────────────────────────┬──────────────┘
         │                                         │
         ▼                                         ▼
┌──────────────────┐                    ┌──────────────────────────┐
│   caal-core      │                    │   caal-iching            │
│   (domain-       │                    │   (I Ching sub-universe) │
│    agnostic)     │                    │                          │
│                  │                    │  Own LUT, TF, ranks      │
│  TokenLUT        │                    │  Hexagram R-link matrix  │
│  globals (G1-3)  │                    │  Consultation/navigation │
│  materialize     │                    │  Compression→disambig    │
│  ngram (1/2/3)   │                    │                          │
│  retrieve (BSS)  │                    │  implements Domain trait │
│  Bridge trait    │                    │                          │
│  Rank trait      │                    │                          │
│  Domain trait    │                    │                          │
└──┬───────┬───────┘                    └──────────┬───────────────┘
   │       │                                       │
   │       └───────────────────────┐               │
   ▼                               ▼               │
┌──────────────┐          ┌──────────────────┐     │
│   caal-zh    │          │  caal-*          │     │
│  (CAAL Zh)   │          │  (future         │     │
│              │          │   domains)       │     │
│  80K Chinese │          │                  │     │
│  characters  │          │  e.g. caal-en    │     │
│              │          │       caal-dna   │     │
│  implements  │          │       caal-audio │     │
│  Domain      │          │                  │     │
└──────┬───────┘          └────────┬─────────┘     │
       │                           │               │
       └───────────┬───────────────┴───────────────┘
                   │
                   ▼
          ┌─────────────────────┐
          │  Storage Abstraction│
          │  (passthrough to    │
          │   hllset Storage)   │
          └──────────┬──────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
   ┌──────────────┐      ┌──────────────┐
   │ Redis        │      │ Memory /     │
   │ (production  │      │ Sled / IPFS  │
   │  default)    │      │ (dev/test)   │
   └──────────────┘      └──────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │  hllset-core    │
            │  (external dep) │
            └─────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │  caal-py        │
            │  (Python        │
            │   bindings)     │
            └─────────────────┘
```

### 8.6 Crate Responsibilities (Redesigned)

#### caal-core (keep, refine)

- **Domain-agnostic.** Does not know about Chinese, I Ching, or any specific domain.
- TokenLUT with monotonic CRDT TF tracking
- Global aggregators (G1/G2/G3) — union of all n-grams
- Materialize (LUT-mediated token recovery)
- Retrieve (BSS-based similarity search against stored HLLSets)
- ngram (1/2/3-gram tokenization)
- **New: Bridge trait** — `fn re_represent(&self, src: &HLLSet) -> HLLSet`
- **New: Rank trait** — defines F, G, H aggregation interfaces (delegates to hllset-ranks when available)

#### caal-zh (keep, refine)

- 80K Chinese character vocabulary
- CAAL ZH tokenizer (character-level murmurhash3)
- Own LUT, own TF, own globals
- Implements Bridge trait for CAAL ZH ↔ other domains

#### caal-iching (redesign)

- **Must have its own LUT** — independent of caal-zh
- I Ching sub-vocabulary (~200 characters actually used in the corpus)
- Hexagram R-link matrix (64×64, pre-computed at corpus ingestion)
- Consultation: `argmax BSS(H_bridge, hex_i)` — structural match
- Navigation: `argmax R-link_weight(hex_i, hex_j)` — transition
- Implements the compression→deliberation→disambiguation cycle:
  - Compression: scene → murmurhash3 → filter through I Ching sub-vocabulary
  - Deliberation: consultation + navigation within the I Ching universe
  - Disambiguation: guidance text → source tokenizer → H_src_guidance
- **No shared LUT with caal-zh.** Statistics constraint enforced.

#### caal-pipeline (refine)

- Scene ingestion → KB retrieval → I Ching consultation → feedback merge
- Two-fork model: strategic path (I Ching) + tactical path (KB)
- Feedback: `H_src_next ∪ H_guidance` — CRDT merge
- Fire-and-forget: each fork is independent, no scheduling

#### caal-py (keep, refine)

- Expose caal-core, caal-zh, caal-iching to Python
- Support notebook-driven experimentation

### 8.7 Storage Abstraction Layer

caal-llm does not implement storage. It delegates to hllset-next's `Storage`
trait. The storage abstraction is a **passthrough** — a thin initializer that:

1. Selects a backend at startup (Redis, Memory, Sled/IPFS-native)
2. Wraps it in `Arc<dyn Storage>` for shared access across the application
3. Provides it to all components that need persistence

```rust
// caal-storage (new crate, ~80 lines)
pub fn init_storage(config: &StorageConfig) -> Result<Arc<dyn Storage>, CaalError> {
    match config.backend {
        Backend::Redis { url } => {
            let redis = RedisStorage::connect(&url)?;
            Ok(Arc::new(redis))
        }
        Backend::Memory => {
            Ok(Arc::new(MemoryStorage::new()))
        }
        Backend::IpfrsNative { path } => {
            let sled = IpfrsNative::open(&path)?;
            Ok(Arc::new(sled))
        }
        // Backend::Postgres { ... } ← future, no code change needed
        // Backend::S3 { ... }       ← future, no code change needed
    }
}
```

**Design rules for the storage layer:**

| Rule | Rationale |
| ------ | ----------- |
| Storage is selected by configuration (TOML, env, CLI flag), not by `#[cfg]` | Same binary works with any backend. No recompilation. |
| Application code receives `Arc<dyn Storage>`, never a concrete type | Adding a backend = adding a variant to the enum + one match arm. Zero changes above this layer. |
| The storage crate is the **only** crate that imports backend crates | `hllset-storage-redis` is imported only in `caal-storage`. If it's path-blocked, only this crate needs a workaround. |
| Redis is the default for production use | Documented in quickstart. Memory backend is for testing. Sled is for single-node development. |

**Backend selection guideline:**

| Backend | When to use | Status in hllset-next |
| --------- | ------------ | ---------------------- |
| `MemoryStorage` | Unit tests, CI, quick experimentation | `[IMPL]` |
| `IpfrsNative` (sled) | Single-node development, local persistence | `[IMPL]` (path-blocked) |
| `RedisStorage` | Production, multi-client access, real workloads | `[IMPL]` |
| `PostgresStorage` | Enterprise with SQL query requirements | `[SPEC]` (future) |
| `S3Storage` | Cloud-native, object storage | `[SPEC]` (future) |

### 8.8 Extension Standards

caal-llm defines three traits that every domain extension must implement. These
are the **stable API contract** — they will not change when new domains are
added, because they capture what is universal about any domain, not what is
specific to Chinese or I Ching.

#### The Domain Trait

```rust
/// A domain that can produce HLLSets from its native input format.
/// Every domain (CAAL Zh, English, DNA, sensor data) implements this.
pub trait Domain: Send + Sync {
    /// Unique name for this domain (e.g., "caal-zh", "caal-en")
    fn name(&self) -> &str;

    /// Tokenize domain-native input into an HLLSet (Pass 1)
    fn tokenize(&self, input: &str) -> Result<HLLSet, CaalError>;

    /// The LUT for this domain — owns its own TF statistics
    fn lut(&self) -> &TokenLUT;

    /// Mutable access to LUT (for ingestion)
    fn lut_mut(&mut self) -> &mut TokenLUT;

    /// Global accumulators (G1, G2, G3) for this domain
    fn globals(&self) -> &Globals;
}
```

#### The Tokenizer Trait

```rust
/// How a domain converts raw text/bytes into tokens.
/// Separated from Domain so that one domain can have multiple tokenizers.
pub trait Tokenizer: Send + Sync {
    /// Split input into tokens
    fn tokenize(&self, input: &str) -> Vec<String>;

    /// What n-gram sizes this tokenizer supports
    fn ngram_sizes(&self) -> &[usize];
}
```

#### The Bridge Trait

```rust
/// Re-representation: maps an HLLSet from a source domain into
/// this domain's bit space (Pass 2).
pub trait Bridge: Send + Sync {
    /// Re-represent a source HLLSet into this domain's bit space.
    /// The returned HLLSet is a citizen of this domain's lattice.
    fn re_represent(&self, src: &HLLSet) -> Result<HLLSet, CaalError>;

    /// Extract 3-gram structural fingerprint from any HLLSet
    fn extract_3gram(&self, hllset: &HLLSet) -> Result<HLLSet, CaalError>;
}
```

#### How Extensions Plug In

A new domain (e.g., `caal-en` for English) is exactly:

```text
crates/caal-en/
├── Cargo.toml          ← depends on caal-core (for traits), hllset-core (for HLLSet)
├── src/
│   └── lib.rs           ← ~100 lines:
│       1. Define English tokenizer (space-split + BPE or word-level)
│       2. Implement Domain trait (own LUT, own globals)
│       3. Implement Bridge trait (Pass 2 re-representation)
│       4. Register in caal-pipeline's domain registry
└── tests/
    └── integration.rs   ← BSS against known English HLLSets
```

**Registration is a one-line addition** in `caal-pipeline`:

```rust
// caal-pipeline/src/lib.rs
let domains: Vec<Arc<dyn Domain>> = vec![
    Arc::new(CaalZh::new()),
    Arc::new(CaalEn::new()),       // ← added, nothing else changes
    Arc::new(CaalIching::new()),
];
```

The extension standard guarantees:

- `caal-core` never imports `caal-en` (or any domain crate)
- `caal-pipeline` only knows `Arc<dyn Domain>` — never a concrete type
- New domains require zero changes to existing crates

### 8.9 Upstream-First Prototype Path

When hllset-next specifies a capability but hasn't implemented it (marked
`[SPEC]` in Part VII), caal-llm follows a three-phase path:

```text
Phase A: PROTOTYPE in caal-llm
  │  Marked with #[doc = "[PROTO]"] on the module.
  │  Implements the specified behavior from this standard.
  │  Used by caal-llm internally. Not a public API commitment.
  │
  ▼
Phase B: STABILIZE through usage
  │  Used in real caal-llm workflows (notebooks, tests, pipelines).
  │  API churn is expected. Breaking changes are cheap (single consumer).
  │  Once the interface stabilizes and tests pass consistently →
  │
  ▼
Phase C: CONTRIBUTE upstream to hllset-next
  │  Extract the [PROTO] module into a new hllset-next crate.
  │  caal-llm switches from local [PROTO] to upstream crate.
  │  Mark the [SPEC] as [IMPL] in Part VII of this standard.
```

**Current [PROTO] candidates for caal-llm:**

| Module | Maps to hllset-next crate | Priority | Rationale |
| -------- | -------------------------- | ---------- | ----------- |
| `caal-core/src/bridge.rs` | `hllset-bridge` | P0 | Two-pass re-representation is fundamental; caal-llm needs it now |
| `caal-core/src/rank.rs` | `hllset-ranks` (local subset) | P1 | Five-level rank algebra; currently path-blocked, so local prototype |
| `caal-core/src/ngram.rs` (3-gram) | `hllset-dsl` extension | P2 | 3-gram structural fingerprinting for cross-domain matching |
| `caal-pipeline/src/temporal.rs` | `hllset-temporal` | P3 | Per-session temporal tracking; prototype before full pyramid crate |

**Rules for [PROTO] modules:**

1. Must replicate the standard's specification faithfully — no shortcuts
2. Must have standalone tests (not dependent on caal-llm integration)
3. Must be documented with the target hllset-next crate name
4. Must not be part of caal-llm's public API — `pub(crate)` visibility
5. Extraction to hllset-next must not break caal-llm's tests

### 8.10 What Must Change (Priority Order)

| Priority | Change | Rationale |
| ---------- | -------- | ----------- |
| **P0** | Separate `caal-iching` LUT from `caal-zh` LUT | Enforce statistics constraint §5.5. Current POC shares a LUT. |
| **P0** | Implement bridge via trait in caal-core | Two-pass re-representation as a first-class concept |
| **P1** | Rename `caal-zh` to clarify role | It's the CAAL domain adapter, not "the Chinese crate" |
| **P1** | Implement disambiguation cycle in caal-pipeline | R → R-R → (deliberation) → R-R → R feedback |
| **P2** | Add 3-gram fingerprinting to caal-core ngram | Enables cross-domain structural matching |
| **P2** | Implement Spearman rank correlation in caal-core | For bridge candidate ranking |
| **P3** | Move to full namespace prefixes (o:/h:/r:/d:/n:) | When hllset-core adds them |
| **P3** | Implement fold views (v: prefix) for llms.txt | For self-describing codebase |

### 8.11 What caal-llm Should NOT Do

- **Do not implement a temporal pyramid.** This is hllset-next's responsibility.
  Local temporal tracking (per-session) is acceptable as application logic.
- **Do not reimplement ranks or derivatives.** Use hllset-ranks when the
  ipfrs-core path dependency is resolved. In the meantime, a local [PROTO]
  implementation is acceptable (§8.9).
- **Do not design around floating-point limitations.** The architectural
  direction is integer-only. caal-llm can use float BSS as a convenience
  but should not build core logic on float operations.
- **Do not create a Forth dictionary in caal-llm.** The Forth dictionary is
  hllset-next's learning component. caal-llm's "learning" is the LUT TF
  accumulation and the bridge's structural matching — not rank reshuffling.
- **Do not couple to a specific storage backend.** All storage access goes
  through `Arc<dyn Storage>`. No module imports `RedisStorage` directly
  except `caal-storage`.

### 8.12 Dependency Policy

| Dependency | Status | Policy |
| ------------ | -------- | -------- |
| `hllset-core` | `[IMPL]` usable | **Primary dependency.** All crates depend on this. |
| `hllset-storage` | `[INACC]` path-blocked | Depend on the **trait**, not the crate. caal-llm defines its own `Storage` re-export that wraps hllset-next's trait once available. |
| `hllset-storage-redis` | `[IMPL]` | **Production default.** Used via `caal-storage`. The only crate that imports this. |
| `hllset-ranks` | `[IMPL]` but path-blocked | [PROTO] local implementation in `caal-core/src/rank.rs`. Switch to upstream when path is fixed. |
| `hllset-mesh` | `[PART]` single-process | Not needed for caal-llm (single-process application). |
| `hllset-forth` | `[PART]` parser only | Not needed for caal-llm. |
| `hllset-dsl` | `[IMPL]` path-blocked | Do not depend on. |
| `hllset-bridge` | `[SPEC]` not yet exists | [PROTO] local implementation in `caal-core/src/bridge.rs`. Contribute upstream when stable (§8.9). |
| `caal-storage` | **New crate** | Thin passthrough to hllset-next `Storage` backend. Only crate that imports concrete backend types. |

### 8.13 Future Integration Points

When hllset-next progresses, caal-llm upgrades:

| When hllset-next... | caal-llm should... |
| --------------------- | ------------------- |
| Resolves ipfrs-core path dependency | Switch from local [PROTO] to `hllset-ranks` and `hllset-storage` |
| Implements full prefix taxonomy (o/r/d/n) | Adopt content-addressable D/R/N storage |
| Ships temporal pyramid crate | Add temporal depth to consultation history |
| Ships `hllset-bridge` crate | Replace `caal-core/src/bridge.rs` [PROTO] with upstream |
| Implements `l:` prefix + folder views | Ingest caal-llm's own source code for self-description |
| Adds new storage backends (Postgres, S3) | Add variant to `Backend` enum in `caal-storage` — no other changes |

---

## Appendix A: Resolved Contradictions

This appendix records contradictions found across the source documents and
how this standard resolves them.

| Contradiction | Documents | Resolution |
| -------------- | ----------- | ------------ |
| BSS (float scalar) vs R-links (topological HLLSet) | Bible §2, §12 replaced BSS; code still uses BSS | Standard: R-links are the architectural direction (§4.4). BSS is a convenience function. |
| Float vs integer arithmetic | Docs reject float for FPGA; code uses float for BSS/cardinality | Standard: Integer-only is the target (§3.2). Current code has mixed arithmetic — must converge. |
| Forth dictionary as "seat of learning" vs parser-only implementation | Bible §5; Review §1.1 | Standard: Recognized as [PART] — the architecture is correct, the code isn't there yet. |
| Statistics sharing in notebook 11 vs statistics constraint | Notebook cells 10-12; IICA_STATISTICS_CONSTRAINT.md | Standard: Notebook is a POC shortcut. The correct architecture enforces separate LUTs (§5.5). |
| D_P = N_s + 2 vs N_s + 1 | DIMENSIONAL_NESTING.md §1.1 vs §1.2 | Standard: Dynamic D_P = N + 2 (including temporal scanning). Static D_P = N + 1 (presentation only). |
| "system:tf" as 32,768 × f64 vs integer-only FPGA constraint | HLPP §3.2; Bible §17.1 | Standard: TF is stored as f64 (current wire format). Rank is integer-derived from TF. This is an acceptable bridge between storage format and computation regime. |

---

## Appendix B: The CAAL-LLM Proof (July 2026)

The notebook `11_caal_llm_demo.ipynb` demonstrated that a content-addressed LLM
built on HLLSet Algebra works:

```text
Training data: 10 Chinese sentences (~100 characters, driving rules)
Tokenization:  character-level
Questions:     5 driving scenario questions in Chinese
Correct:       4/5 (80%)

Q: "what to do at an intersection?"    → "slow down at intersections" ✓
Q: "what to watch for on the highway?" → "keep safe distance on highway" ✓
Q: "what to watch for in rain?"        → "reduce speed on wet roads" ✓
Q: "what to do on red light?"          → "signal before turning" (close)
Q: "what to do seeing a pedestrian?"   → "yield to emergency vehicles" (close)
```

No gradient descent. No weight matrices. No GPU. No transformer. No BPE
tokenizer. Just murmurhash3 + bitwise AND + popcount.

This validates two architectural principles:

1. **CAAL (Chinese as Assembly Language):** Characters ARE tokens. Fixed set
   (~80K), deterministic by construction.
2. **Context (HLLSet) based LLM:** Learning = accumulating HLLSets. Inference
   = structural similarity via BSS.

---

## Appendix C: Document Map

For developers moving from the old per-topic docs to this standard:

| Old document | See STANDARD.md section(s) |
| ------------- | --------------------------- |
| `HLPP.md` | Part II (full) |
| `TF_VS_RANK.md` | §3.1 |
| `DIMENSIONAL_NESTING.md` | §3.4 |
| `UNIVERSAL_BRIDGE.md` | Part V (full) |
| `IICA_PRINCIPLES.md` | Part I (full) |
| `IICA_STATISTICS_CONSTRAINT.md` | §5.5 |
| `SELF_REPROGRAMMING_ARCHITECTURE.md` | Parts III, IV, VI, Appendix B |
| `HLLSET_NEXT_REVIEW.md` | Part VII |

---

## Appendix D: LUT Initialization Constraint (Discovered July 28, 2026)

### The Problem

Loading a LUT with a large external vocabulary (e.g., a 128K BPE tokenizer vocabulary)
where all tokens have equal TF = 0 or TF = 1 causes **random materialization**.
Each HLLSet bit position maps to multiple tokens in the LUT, and with equal TFs,
the TF tie-break is arbitrary. Jaccard drops to ~0.03.

### The Rule

> **A LUT may only contain tokens whose accumulated TF reflects actual experience.**
> Three valid initialization states:

| State | Vocabulary source | TF values | When to use |
| ------- | ------------------ | ----------- | ------------- |
| **Cold start** | Empty | N/A | New system, no prior knowledge |
| **Lattice-covered** | Vocabulary extracted from HLLSets already in the current lattice | From materialization TF | System with existing HLLSet corpus |
| **Donor transfer** | Vocabulary from a donor system's LUT | Copied from donor TF | Deep knowledge transfer between systems |

| State | Vocabulary source | TF values | Result |
| ------- | ----------------- | ----------- | -------- |
| **INVALID** | External vocabulary (e.g., tokenizer vocab) | Equal TF (= 0 or = 1) | Random materialization, Jaccard ~0.03 |

### Refinement of §5.5

The original Statistics Constraint states: *"Statistics are not transferable between
independent algebras."* This remains true for **arbitrary** transfer. However:

- TF can be transferred **from a donor LUT** that has earned its statistics through
  real experience. The donor LUT's TF distribution reflects actual token frequencies
  in a real corpus — it is not arbitrary.
- The constraint is not "TF cannot be transferred" but "TF must reflect experience."
  Equal-TF initialization from an external vocabulary violates this because it
  pretends all tokens are equally frequent, which is never true in any real corpus.
- A donor LUT from a system that has processed real documents carries valid TF
  distributions that meaningfully distinguish tokens at shared bit positions.

### Practical implications for DeepSeek-OCR

1. **Cold start:** Begin with an empty LUT. Process a training corpus of documents.
   After ~50-100 pages, the LUT accumulates enough TF for accurate materialization.

2. **Lattice-covered:** If the OCR system has previously processed documents
   (stored as HLLSets), extract the vocabulary from those HLLSets by materializing
   against the cold-start LUT. Use those tokens to seed the next session.

3. **Donor transfer:** If another OCR system (or caal-llm) has a mature LUT with
   real TF values, import that LUT directly. The TF distribution, earned through
   real experience, provides valid disambiguation.

4. **Never:** Load `tokenizer.json` vocabulary with TF = 1 for all tokens.
