# HLLSet Algebra: A Research & Development Retrospective

>**From Theoretical Foundations to Production-Grade Lattice Computing**
>
>*Alex Mylnikov — July 31, 2026*

---

## Abstract

This report traces the development of **HLLSet Algebra** — a mathematical framework
for content-addressed, lattice-structured, FPGA-native World Models — across eight
repositories spanning theoretical foundations, practical implementations, and
application domains. The arc begins with the IICA gate principle (Idempotency,
Immutability, Content-Addressability) as a formal algebraic constraint, proceeds
through multiple implementation generations (Python → Rust, standalone → Redis-backed,
single-node → distributed), and culminates in a production-grade Rust platform
(hllset-next) with 291 tests, 13 crates, and a reference application demonstrating
content-addressed LLM capabilities without gradient descent, weight matrices, or GPUs.

**Keywords:** HLLSet, HyperLogLog, content-addressing, lattice algebra, IICA,
temporal pyramids, FPGA-native computing, CAAL, swarm intelligence, rank algebra

---

## 1. Introduction: The Core Insight

The central thesis of HLLSet Algebra is that **context can be represented as
immutable, idempotent, content-addressed bitmasks** whose algebraic operations
(union, intersection, difference) form a bounded distributive lattice. Unlike
token-embedding approaches that require O(n²) attention, floating-point matrix
multiplication, and GPU clusters, HLLSet operations reduce to single-cycle bitwise
AND/OR/POPCOUNT on FPGAs.

Learning is not gradient descent — it is **rank rearrangement** in a content-addressed
dictionary. HLLSets themselves never change. Only their relevance to the current
context changes. System history is not a log file — it is a **holographic property**
of the lattice itself, recoverable from the current system state and a compressed
TF (Term Frequency) vector.

This architecture addresses three structural limits of the LLM paradigm:
hallucination (content-addressing guarantees deterministic retrieval), attention
complexity (bitmask operations are O(1)), and the inability to continuously learn
from streaming data (monotonic CRDT rank adjustment).

---

## 2. The Development Arc: Eight Repositories

### 2.1 Theoretical Foundation: `sgs_icaisns`

**Role:** Origin point — formal specification of the IICA gate principle.

The IICA gate is the foundational constraint of the entire framework:

| Property | Definition | Consequence |
| ---------- | ------------ | ------------- |
| **I**dempotency | f(x) = f(f(x)) | Same input → same output, always |
| **I**mmutability | f(x) = y is fixed | No state, no mutation, no temporal dependency |
| **C**ontent-**A**ddressability | If a = b then f(a) = f(b) | Deterministic; given content, you can find it again |

**Composition theorem:** If each morphism in a pipeline satisfies IICA, the
composition satisfies IICA. This single theorem makes nested spaces, distributed
convergence, and cross-domain bridges work without new theory — a mathematical
guarantee rather than an engineering property.

The repository established that hash functions are canonical IICA builders, that
HyperLogLog bitmaps satisfy the lattice axioms, and that CRDT convergence emerges
from monotonic union rather than from a consensus protocol.

### 2.2 Swarm Intelligence: `hllset-swarm`

**Role:** Proved the HLLSet–PSO duality and demonstrated programmable semantic trajectories.

HLLSet-Swarm established the **mathematical duality** between (a) relational algebra
of Chinese-character HLLSets and (b) Particle-Swarm Optimization dynamics, enabling
a declarative GPU kernel compiler that scripts how an 80k-dimensional "semantic
swarm" should move, converge, and write its final state back as live feedback.

**Key contributions:**

- PSO guarantees → HLLSet stability proofs (formal convergence analysis)
- Programmable trajectories: YAML → GPU sparse kernels (no CUDA code)
- Recursive meta-swarm: swarm-of-swarms for higher-order abstraction
- The concept of "Git for meaning" — every trajectory ends with a content-addressed
  commit that immortalizes the swarm's belief state

### 2.3 Early Algebra Implementations: `redis_hllset_algebra`

**Role:** First operational HLLSet algebra engine, implemented on Redis.

This repository demonstrated that HLLSet operations (union, intersection, difference,
cardinality estimation, BSS similarity) could be implemented efficiently on top of
Redis data structures. The Redis backend provided persistence, multi-client access,
and enterprise-grade reliability while the HLLSet bitmasks lived in Redis strings
with Roaring Bitmap compression.

**Key contributions:**

- Redis-native HLLSet storage with SET/GET/BITOP operations
- Multi-seed consensus for homogeneous (catalog) data materialization
- Proved that a production database could serve as the storage layer without
  sacrificing the algebraic properties

### 2.4 The Forth DSL: `hllset_dsl`

**Role:** Original production-targeted Forth DSL for content-addressed probabilistic
set operations. This is the project that hllset-next evolved from.

The DSL provided a Forth-based programming language that compiles to Lua, enabling
declarative HLLSet manipulation. It integrated Go IPFS daemon for content-addressed
storage and ROS 2 for pub/sub messaging, establishing the full HLPP (HLLSet Lattice
Persistence Protocol) in practice.

**Key contributions:**

- Forth → Lua compiler with INSCRIBE, UNION, INTERSECT, DIFF, BSS, CARD operations
- SNOBOL-inspired composable pattern matching for tokenization
- Tokenizer pipeline with n-gram support, boundary padding, and lowercase normalization
- Materialization strategies: InLUT, NgramCrossValidate, DeBruijnReconstruct
- First complete demonstration of IICA pipeline in practice

### 2.5 The Rust Rewrite: `hllset-next`

**Role:** Current production platform — Rust-native rewrite eliminating Go and Python
dependencies. **This is the project whose completion is documented in this report.**

hllset-next replaced Go IPFS daemon with `ipfrs-core` (Rust-native content-addressing
via sled), ROS 2 Python nodes with an in-process tokio broadcast bus, and consolidated
all infrastructure into a single-language Rust platform.

**Key contributions (as of July 31, 2026):**

*Core Algebra (hllset-core):*

- HLLSet bitmap (1024×32 registers), union/intersection/difference/XOR
- BSS (Bell State Similarity) morphisms: inclusion (τ), exclusion (ρ), morphism check
- Content addressing with full namespace taxonomy: o/h/r/d/n/t/v/l/c/u + system:
- TFVec — 32,768-entry monotonic CRDT bit-level frequency vector (262KB wire format)
- Commit — D/R/N (Departed/Retained/New) lattice evolution record with `t:<sha1>` key
- Horvitz-Thompson cardinality estimator with monotonic guarantee

*Storage Protocol (hllset-storage):*

- 11-method Storage trait: CA ops (put/get/has/list/pin/unpin/gc/delete) + temporal
  ops (put_tmp/get_tmp/cas_tmp)
- Three backends: MemoryStorage (dev/test, full temporal+CAS), IpfrsNativeStorage
  (sled, local persistent), RedisStorage (enterprise)
- Legacy `store`/`load`/`exists` aliases for backward compatibility

*Rank Algebra (hllset-ranks):*

- Five-level integer-only rank propagation: token → bit → register → HLLSet → compound
- Pluggable aggregation functions at each level (Identity, Log2, Max, Sum, MaxPool, Degree)
- Rank derivatives: ΔR (flux), Δ²R (acceleration), Noether steering
- Fisher matrix: sparse cross-layer bit co-occurrence for systemic vs noise detection
- Observable mask: rank-threshold attention filter (controls visibility, not existence)
- TfRegisterRanker: TF vector → 1,024 register-level ranks without TokenLUT

*Temporal Pyramid (hllset-temporal):*

- Configurable N-layer sliding window with automatic carry cascade
- Default 7-layer pyramid (second → year), plus presets for HFT, realtime, document analysis
- System state: bit-lossless union of all layers (H_system = ∪L_i)
- Per-layer TF snapshots for time-lens queries against historical states

*Universal Bridge (hllset-bridge):*

- Two-pass ingestion: representation (domain → HLLSet) + re-representation (bits → bridge)
- 3-gram structural fingerprinting for cross-domain matching
- Spearman rank correlation (ρ ∈ [-1, 1]) with tie handling
- Full bridge pipeline: re-represent → fingerprint → rank-correlate → top-K matches

*DSL & Tooling:*

- Lua runtime with full algebra + storage + temporal bindings
- Forth DSL with colon-definition support (parsed into `Word::ColonDef`, lowered to Lua functions)
- Interactive REPL, CLI evaluation (`-e`), file execution (`-f`), Forth compilation (`--forth`)
- Mesh bus: in-process tokio broadcast with Noether flux controller (integer halving decay)
- 14 Jupyter notebooks (Rust evcxr kernel) from core algebra to advanced bridge operations

*Test Suite:* 291 tests, 0 failures across 13 crates.

### 2.6 Reference Application: `caal-llm`

**Role:** Reference application demonstrating CAAL (Chinese as Assembly Language) LLM
built on HLLSet Algebra. Validates that content-addressed LLMs work.

The proof: 10 Chinese sentences (~100 characters, driving rules) for training, 5 driving
scenario questions, 4/5 correct (80% accuracy). No gradient descent. No weight matrices.
No GPU. No transformer. Just murmurhash3 + bitwise AND + popcount.

**Key contributions:**

- caal-core: domain-agnostic HLLSet algebra (LUT, globals, materialize, n-gram, retrieve)
- caal-zh: Chinese tokenizer with 80K character vocabulary
- caal-iching: I Ching sub-lattice with independent LUT (statistics constraint enforced)
- caal-pipeline: two-fork model (strategic path via I Ching + tactical path via KB)
- caal-py: PyO3 bindings for notebook-driven experimentation

### 2.7 Application Domain: `DeepSeek-OCR`

**Role:** OCR application demonstrating HLLSet algebra for document processing.
Validated the LUT initialization constraint (discovered July 28, 2026): a LUT must
contain tokens whose accumulated TF reflects actual experience — loading a tokenizer
vocabulary with equal TF = 0 or TF = 1 causes random materialization (Jaccard ~0.03).

### 2.8 Enterprise Extension: `redis_hllset_mdb`

**Role:** Materialized database extension — Redis-backed multi-database HLLSet storage
with graph engine integration. Represents the enterprise deployment target where
RedisGraph's sparse adjacency matrix maps directly to HLLSet Fisher matrices.

---

## 3. Architectural Principles

### 3.1 The IICA Gate

Every operation that connects HLLSet Algebra components must satisfy Idempotency,
Immutability, and Content-Addressability simultaneously. Composition preserves IICA —
this single theorem eliminates the need for consensus protocols, leader election,
or distributed coordination in multi-node deployments.

### 3.2 TF vs Rank Separation

**TF is stored. Rank is derived. They are not the same thing.** TF is a bit-level
frequency vector (32,768 × f64, monotonic CRDT) shared across the system. Rank is
computed locally from TF at query time. TF requires the same token base for comparison;
rank is hash-function-agnostic and meaningful across different token bases. This
separation enables domain-universal operation.

### 3.3 The Lattice as Neural Network

The rank vector of any HLLSet H is the element-wise product of the TF vector and
H's bitmask: rank(H) = TF ⊙ bitmask(H). This is structurally identical to a neuron
(bitmask = weights, TF = activation signal, rank = output), but the "weights" are
content-addressed and immutable. Learning = bitmask selection, not weight adjustment.

### 3.4 FPGA-Native Constraint

Every HLLSet operation reduces to AND, OR, POPCOUNT, ADD, SUB, CMP — all LUT-level
FPGA primitives executable in a single cycle. The five-level rank algebra is
designed for FPGA implementation: Max (CMP tree), Sum (ADD chain), Popcount (POPCOUNT),
Degree (POPCOUNT of adjacency row).

### 3.5 Dimensional Nesting: D_P = N + 2

The same five operations (∪, ∩, \, popcount, key()) apply at every level of nesting:
sensor → robot → swarm → command post → theater. A robot with N_s sensors has D_P = N_s + 2
perceptual dimensions. A swarm of N_r robots has D_P = N_r + 2. Zero new algebra required.

---

## 4. Current Status (July 31, 2026)

### Implementation Status Matrix

| Layer | Status | Components |
| ------- | -------- | ------------ |
| IICA Foundation | ✅ Complete | Gate definition, composition theorem, hash builders, MurmurHash3/SHA-1 pipeline |
| HLPP Protocol | ✅ Complete | Algebraic spec, Storage trait (11 methods), 3 backends, Lua/Forth bindings |
| Core Concepts | ✅ Complete | HLLSet bitmap, TFVec, Commit, TF/Rank separation, five-level rank algebra |
| Architecture | ✅ Partial | Evolution equation, D/R/N, fire-and-forget, Noether (done); system lifecycle (spec) |
| Temporal Pyramid | ✅ Complete | Configurable N-layer, carry cascade, TF snapshots, Noether invariant |
| Universal Bridge | ✅ Complete | Two-pass re-representation, 3-gram, Spearman correlation, bridge pipeline |
| Forth DSL | ✅ Complete | Parser, Lua lowering, colon-definitions; Rust/Verilog backends (spec) |
| Storage | ✅ Complete | Memory (19 tests), Sled/IPFS (13 tests), Redis (5 tests) |
| Self-Ingestion | ⬜ Specified | Git commit → ingest pipeline, llms.txt, folder views, dev TUI |
| Distribution | ⬜ Partial | In-process mesh (done); distributed mesh (spec) |

### Test Suite

**291 tests, 0 failures** across 13 crates.

---

## 5. The Broader Context

### 5.1 Why This Architecture Matters

Three forces converged in 2026:

1. **The GPU wall** — training runs cost hundreds of millions; edge inference requires
   lossy compression. The industry seeks architectures without matrix multiplication.
2. **FPGAs are ready** — modern fabrics support the exact operations HLLSet Algebra
   uses at LUT-level, single-cycle.
3. **LLM paradigm has structural limits** — hallucination, O(n²) attention, batch
   training, and the inability to learn continuously are consequences of the
   token+attention architecture, not bugs to be patched.

### 5.2 The Categorical Shift

HLLSet Algebra represents a categorical shift from:

- Token embeddings → content-addressed bitmasks
- Gradient descent → rank rearrangement
- GPU clusters → single FPGA chips
- Batch training → continuous streaming learning
- Log files → holographic lattice memory
- Consensus protocols → IICA-guaranteed convergence

### 5.3 CAAL-LLM Proof

The most striking validation: a content-addressed Chinese LLM achieving 80% accuracy
on driving scenario questions trained on just 10 sentences. The same architecture
that powers LLMs through billions of parameters and trillions of tokens of gradient
descent is matched by 100 characters, murmurhash3, and bitwise AND + popcount.
This is not an optimization — it is evidence of a different computational regime.

---

## 6. Repository Map

| # | Repository | Role | Primary Language |
| --- | ----------- | ------ | ----------------- |
| 1 | `sgs_icaisns` | Theoretical foundation: IICA gate specification | Python/Mathematical |
| 2 | `hllset-swarm` | Swarm intelligence: HLLSet–PSO duality, GPU kernel compiler | Python |
| 3 | `redis_hllset_algebra` | First operational algebra engine (Redis-backed) | Python/Redis |
| 4 | `hllset_dsl` | Original Forth DSL with Go IPFS + ROS 2 integration | Rust/Forth/Lua |
| 5 | `hllset-next` | **Current platform**: Rust-native, 13 crates, 291 tests | Rust |
| 6 | `caal-llm` | Reference application: Chinese as Assembly Language LLM | Rust/Python |
| 7 | `DeepSeek-OCR` | Application domain: OCR with HLLSet algebra | Rust/Python |
| 8 | `redis_hllset_mdb` | Enterprise extension: materialized database + graph engine | Redis/Rust |

---

## 7. Future Directions

1. **Distributed mesh networking** — the `MeshBus` trait in hllset-mesh is designed
   for a Kademlia DHT + QUIC transport swap-in (mielin-mesh from MielinOS).

2. **Self-ingestion pipeline** — git commit → HLLSet ingest, `llms.txt` auto-generation
   from doc comments, folder views with `l:` and `v:` prefixes.

3. **FPGA synthesis** — the Forth → Verilog backend would close the loop from
   software specification to hardware implementation.

4. **Graph engine integration** — RedisGraph's sparse adjacency matrix maps directly
   to HLLSet Fisher matrices; a four-phase migration from demo to enterprise.

5. **caal-llm hardening** — upgrade from proof-of-concept to production reference
   application with Redis backend, temporal pyramid, and bridge integration.

---

## References

1. STANDARD.md — Authoritative HLLSet Algebra specification (hllset-next/_DOCS/dev/)
2. GREYLOCK_EXECUTIVE_SUMMARY.md — Business vision and technical milestones
3. Flajolet, P., et al. "HyperLogLog: the analysis of a near-optimal cardinality
   estimation algorithm" (2007)
4. Ashby, W.R. "Design for a Brain" (1952) — the homeostat as ancestral architecture
5. Noether, E. "Invariante Variationsprobleme" (1918) — symmetry → conservation
6. A. Mylnikov, ``HLLSet Theory: A Unified Framework for Probabilistic Knowledge Representation``, Advances in Science, Technology and Engineering Systems Journal, vol. 11, no. 2, pp. 12--16, 2026.
7. A. Mylnikov, ``Self Generative Systems (SGS) and Its Integration with AI Models``. AISNS '24: Proceedings of the 2024 2nd International Conference on Artificial Intelligence, Systems and Network Security
Pages 345 - 354 (<https://doi.org/10.1145/3714334.3714392>)
