# ONNX vs HLLSet Lattice: A Comparative Architecture Analysis

>**Two Graph-Based Computational Paradigms — Divergent Foundations, Convergent Patterns** (Prepared with assistance from DeepCode)
>
>*July 31, 2026*

---

## Abstract

ONNX (Open Neural Network Exchange) and hllset-next represent two graph-based
computational architectures that, despite serving radically different purposes
(model exchange format vs live computational lattice), exhibit surprising structural
parallels. Both organize computation as directed graphs, both support composable
sub-graphs, and both separate storage from execution. However, their foundations
diverge categorically: ONNX operates on mutable float tensors with named parameters
and gradient-based learning, while hllset-next operates on immutable content-addressed
bitmasks with lattice algebraic operations and rank-based adaptation. This report
analyzes the architectures across nine dimensions, identifies where they converge
and where they fundamentally differ, and explores the implications of each design
choice for the future of computational intelligence.

---

## 1. Introduction

ONNX emerged in 2017 as an industry consortium effort (Microsoft, Facebook, AWS)
to create an open standard for representing machine learning models, enabling
interoperability between frameworks (PyTorch, TensorFlow, scikit-learn) and
runtimes (ONNX Runtime, TensorRT, OpenVINO). It is the de facto standard for
model exchange in production ML pipelines.

HLLSet Algebra emerged from a different lineage — formal algebraic constraints
(IICA), probabilistic data structures (HyperLogLog inspired HLLSets), and the conviction that
content-addressing and immutability could replace the brittle infrastructure of
token embeddings, attention mechanisms, and gradient descent. The paradigm shift was categorical: **from explicit tokens and implicit contexts to explicit contexts and implicit tokens**.

Yet both architectures are, at their core, **graph-based computational systems**
where nodes represent operations and edges represent data flow. This structural
convergence makes comparison illuminating — it reveals what each architecture
gains and loses from its foundational choices.

---

## 2. Architectural Overview

### 2.1 ONNX: The Model Exchange Standard

```text
┌──────────────────────────────────────────────────┐
│                  ONNX Model                      │
│  ┌─────────┐    ┌─────────┐     ┌─────────┐      │
│  │ Input   │──▶│ Conv2D  │───▶│  ReLU   │      │
│  │ (float) │    │ (float) │     │ (float) │      │
│  └─────────┘    └────┬────┘     └────┬────┘      │
│                      │ Weight        │           │
│                      │ (named)       ▼           │
│                      │           ┌─────────┐     │
│                      │           │  MatMul │     │
│                      │           │ (float) │     │
│                      │           └────┬────┘     │
│                      │                │          │
│                      ▼                ▼          │
│               ┌──────────┐        ┌──────────┐   │
│               │   Loss   │◀──────│  Output  │   │
│               │ (scalar) │        │ (tensor) │   │
│               └──────────┘        └──────────┘   │
│                                                  │
│  DAG of named float tensors + standard ops       │
│  Serialized as Protobuf                          │
│  Gradients flow backward for training            │
└──────────────────────────────────────────────────┘
```

**Core characteristics:**

- **Graph type:** Directed Acyclic Graph (DAG) with typed edges
- **Node type:** Standardized operators (Conv, MatMul, ReLU, BatchNorm, ~180 ops)
- **Edge type:** Tensors with shape and dtype (float16/32/64, int8/32/64)
- **Parameters:** Named tensors ("layer1.weight", "layer2.bias") stored as initializers
- **Serialization:** Protobuf binary format (.onnx)
- **Execution:** Graph is a static computation plan; runtime interprets it
- **Learning:** External — backpropagation through the DAG computes gradients
- **Versioning:** Operator versioning via opset imports

### 2.2 HLLSet Lattice: The Computational World Model

```text
┌──────────────────────────────────────────────────┐
│              HLLSet Lattice                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │ Source  │──▶│ HLLSet  │──▶│ Bridge  │──┐    │
│  │ (text)  │    │ h:<sha1>│    │ (bits)  │  │    │
│  └─────────┘    └────┬────┘    └─────────┘  │    │
│                      │ R-link               │    │
│                      │ (r:<sha1>)           ▼    │
│                      │               ┌─────────┐ │
│                      │               │  Rank   │ │
│                      │               │ (int)   │ │
│                      │               └────┬────┘ │
│                      │                    │      │
│                      ▼                    ▼      │
│               ┌──────────┐         ┌──────────┐  │
│               │  TFVec   │◀───────│ System   │  │
│               │ (CRDT)   │         │ State    │  │
│               └──────────┘         └──────────┘  │
│                                                  │
│  Lattice of immutable content-addressed bitmasks │
│  All operations: bitwise AND/OR/POPCOUNT         │
│  Learning = rank rearrangement, not gradient     │
└──────────────────────────────────────────────────┘
```

**Core characteristics:**

- **Graph type:** Bounded distributive lattice (partially ordered set with join/meet)
- **Node type:** HLLSets (1024×32 bitmaps) with content keys
- **Edge type:** Structural relations (R-links, temporal layers, bridge projections)
- **Parameters:** Not named — content-addressed (SHA-1 of bitmap IS the identity)
- **Serialization:** Roaring Bitmap binary + canonical JSON for commits
- **Execution:** Lattice is live — operations happen at query time, results are cached
- **Learning:** Internal — rank rearrangement based on TF accumulation (streaming)
- **Versioning:** Immutable history via commit chain (every state is recoverable)

---

## 3. Nine-Dimensional Comparison

### 3.1 Computational Model

| Dimension | ONNX | HLLSet Lattice |
| ----------- | ------ | ---------------- |
| **Primitive** | Float tensor (n-dimensional array) | Bitmask (32,768 bits: 1024×32 registers) |
| **Operations** | ~180 standardized ops (Conv, MatMul, etc.) | 5 core ops: ∪ (OR), ∩ (AND), \ (AND-NOT), popcount, key() |
| **Operation semantics** | Continuous (real-valued outputs) | Discrete (bit positions set/unset) |
| **Computation cost** | O(n²) for attention, O(n³) for matrix ops | O(1) bitwise — single cycle on FPGA |
| **Precision** | Float32/16, Int8 (quantized) | Binary (bit set = 1, unset = 0) |
| **Numerical stability** | Requires careful initialization, normalization | Idempotent — no numerical drift possible |

**Analysis:** ONNX's continuous tensor model enables rich function approximation
but at the cost of O(n²) attention and floating-point brittleness. HLLSet's binary
model sacrifices continuous representation for O(1) operations and absolute
stability. The trade is expressiveness vs efficiency: ONNX can represent any
differentiable function; HLLSet can't represent smooth gradients **but needs none**.

### 3.2 Graph Structure

| Dimension | ONNX | HLLSet Lattice |
| ----------- | ------ | ---------------- |
| **Graph type** | Directed Acyclic Graph (DAG) | Bounded distributive lattice |
| **Cycles** | Forbidden (must be acyclic) | Allowed (R-links form cycles; convergence via idempotence) |
| **Node identity** | Name-based ("layer1.conv.weight") | Content-based (`h:<sha1>` — hash of the data) |
| **Edge meaning** | Data flow (tensor output → tensor input) | Structural relation (intersection, bridge, temporal carry) |
| **Topology** | Fixed by model architecture | Dynamic — evolves with observations |
| **Optimization** | Graph transformations (fusion, constant folding) | Lattice properties exploited (idempotence, monotonicity) |

**Analysis:** ONNX's acyclic constraint enables clean execution ordering but
prevents recurrent computation. HLLSet's lattice structure allows cycles because
idempotence guarantees convergence — the same operation applied twice is a no-op,
so cyclic connections can't diverge. Both architectures use graph optimization:
ONNX fuses operators; HLLSet exploits algebraic laws (A ∪ A = A, A ∪ B = B ∪ A).

### 3.3 Identity and Addressing

| Dimension | ONNX | HLLSet Lattice |
| ----------- | ------ | ---------------- |
| **Identity scheme** | Named parameters | Content-addressed (SHA-1) |
| **Name collision** | Possible (namespace conflicts) | Impossible (different content → different hash) |
| **Deduplication** | Manual (must detect duplicate weights) | Automatic (same content → same hash → same storage) |
| **Versioning** | External (model version number) | Internal (commit chain — every state is a `t:<sha1>`) |
| **Provenance** | Metadata (training run ID, timestamp) | Inherent (R-link history traces every bit's origin) |

**Analysis:** This is the most fundamental divergence. ONNX names are arbitrary
labels assigned by the model author; HLLSet keys are cryptographic hashes of the
content. This means HLLSet enjoys automatic deduplication (IPFS/IPLD-native),
tamper-proof provenance (you can't rename a key without changing its hash), and
distributed convergence without coordination (two nodes computing the same HLLSet
produce the same key). ONNX requires external infrastructure for all three.

### 3.4 Learning Mechanism

| Dimension | ONNX | HLLSet Lattice |
| ----------- | ------ | ---------------- |
| **Learning paradigm** | Gradient descent (batch) | Rank rearrangement (streaming, continuous) |
| **Loss function** | Scalar loss (cross-entropy, MSE) | Rank threshold θ (attention filter) |
| **Update rule** | ∂L/∂W → W -= η·∇W | Ingest observation → TF[bits] += 1 → ranks reshuffle |
| **Convergence** | Local minima of loss surface | CRDT convergence (monotonic union) |
| **Catastrophic forgetting** | Yes (requires replay/rehearsal) | No (TF is monotonic — old knowledge accumulates) |
| **Training data** | IID batches, shuffled | Streaming observations, temporal order preserved |
| **Hardware for learning** | GPU clusters ($100M+) | FPGA (single chip, $100s) |

**Analysis:** This is where the architectures diverge most dramatically. ONNX's
gradient-based learning requires batched IID data, careful hyperparameter tuning,
and massive compute. HLLSet's rank rearrangement is continuous, monotonic, and
requires no batches — it learns from every observation as it arrives. The price:
rank rearrangement can't learn functions that require backpropagation (it has no
notion of gradient). The gain: it never forgets, requires no replay buffer, and
converges deterministically.

### 3.5 State Management

| Dimension | ONNX | HLLSet Lattice |
| ----------- | ------ | ---------------- |
| **Mutability** | Tensors are mutable during training | HLLSets are immutable (IICA) |
| **State after training** | Frozen weights (inference-only) | Live lattice (TF continues to accumulate) |
| **Checkpoint** | Snapshot of all weights (large) | Commit chain (each commit is a 4KB HLLSet union) |
| **Rollback** | Load previous checkpoint | Navigate commit chain (every past state is addressable) |
| **State size** | GB–TB (dense float matrices) | 4KB per HLLSet + 262KB per TF snapshot |

**Analysis:** ONNX models are snapshot-based: training produces a frozen state;
inference uses that state. HLLSet lattices are live: every observation updates
the TF vector, and ranks reflect current relevance. Checkpoint is expensive for
ONNX (must save all weights); for HLLSet it's a 4KB commit record. The trade:
HLLSet can't represent as much per-unit-state as ONNX; ONNX can't continuously
adapt without retraining.

### 3.6 Composability

| Dimension | ONNX | HLLSet Lattice |
| ----------- | ------ | ---------------- |
| **Sub-graph reuse** | Functions (reusable computation sub-graphs) | Colon-definitions (`: NAME ... ;`) |
| **Composition guarantee** | Type compatibility (shape/dtype must match) | IICA: composition of IICA morphisms is IICA |
| **Domain transfer** | Requires retraining/re-calibration | Bridge: structural projection (no retraining) |
| **Nested models** | Sub-models (nested ONNX graphs) | Nested lattices (D_P = N + 2 scaling) |
| **Operator extension** | Custom ops (requires runtime support) | Same five ops at every level of nesting |

**Analysis:** Both support composability, but through different mechanisms. ONNX
functions are computational sub-graphs reused as operators; HLLSet colon-definitions
are Forth words compiled to Lua functions. ONNX requires type compatibility checks;
HLLSet's IICA composition is guaranteed by the algebraic properties of hash
functions. The bridge capability is unique to HLLSet: no retraining needed to
project between domains — it's a structural re-representation.

### 3.7 Hardware Target

| Dimension | ONNX | HLLSet Lattice |
| ----------- | ------ | ---------------- |
| **Primary target** | GPU (CUDA), NPU, CPU (SIMD) | FPGA (LUT-level primitives) |
| **Key operations** | Float matrix multiply (Tensor Cores) | Bitwise AND/OR, POPCOUNT, integer ADD/CMP |
| **Memory pattern** | Dense, coalesced, high bandwidth (HBM) | Sparse, random access, low latency (BRAM) |
| **Parallelism** | SIMT (warp-level), tensor cores | Pipeline-level (single-cycle per operation) |
| **Power budget** | 300–700W (datacenter GPU) | 5–25W (embedded FPGA) |
| **Quantization** | Required for edge deployment | Native — already binary at the core |

**Analysis:** ONNX targets the GPU ecosystem optimized for dense float matrix
operations at massive scale. HLLSet targets FPGA fabrics where AND/OR/POPCOUNT
are single-cycle LUT operations. This is a categorical hardware divergence: the
same computation that requires a datacenter GPU for ONNX runs on an embedded FPGA
for HLLSet. The price: HLLSet can't run existing ONNX models; ONNX can't exploit
FPGA's bitwise parallelism natively.

### 3.8 Temporal Structure

| Dimension | ONNX | HLLSet Lattice |
| ----------- | ------ | ---------------- |
| **Time model** | Static (model is a snapshot) | Dynamic (L0–L6 temporal pyramid) |
| **Sequence handling** | RNN/LSTM/Transformer (explicit) | Temporal pyramid (implicit, configurable) |
| **History compression** | None (or manual checkpoint) | Automatic: L1 = ∪L0, L2 = ∪L1, ... cascade |
| **Temporal query** | Not supported | Time-lens: past_state(t) ≈ H_system ⊙ TF_stack[t] |
| **Window granularity** | Fixed by architecture | Configurable N layers, durations [d₀..d_{N-1}] |

**Analysis:** ONNX models exist outside of time — they are static snapshots.
Temporal modeling requires explicit architectures (RNNs, Transformers) built into
the graph. HLLSet's temporal pyramid embeds time as a first-class structural
property: layers cascade automatically at boundaries, system state is the
bit-lossless union of all layers, and the TF stack enables time-lens queries
against any past moment. This is not a feature — it's a consequence of the
lattice being live rather than frozen.

### 3.9 Convergence and Consistency

| Dimension | ONNX | HLLSet Lattice |
| ----------- | ------ | ---------------- |
| **Convergence guarantee** | Loss minimization (non-convex, local minima) | CRDT monotonic union (provable, global) |
| **Distributed consistency** | Parameter server / all-reduce (complex) | Content-addressing (trivial — same bits → same key) |
| **Consensus protocol** | Required (Paxos/Raft for parameter servers) | Not required (IICA + monotonic union ≡ eventual consistency) |
| **Fault tolerance** | Checkpoint + restore | Content-addressing (any node can reconstruct from shared storage) |
| **Split-brain handling** | Application-level logic | Automatic (union is commutative, order-independent) |

**Analysis:** Distributed ONNX training requires consensus protocols for parameter
synchronization. HLLSet achieves distributed consistency without them — content-addressing
ensures nodes produce identical keys for identical computations, and monotonic union
is order-independent. This is the Noether invariant: multiple paths through the
lattice converge to the same state because union is commutative and idempotent.

---

## 4. Structural Convergences

Despite their divergent foundations, the architectures exhibit unexpected parallels:

### 4.1 The Neural Network Mapping

The standard neural network equation (`output = σ(Wx + b)`) has a direct lattice
analog:

| Neural Network | HLLSet Lattice |
| --------------- | ---------------- |
| Weights W | bitmask(H) — a 32,768-bit binary pattern |
| Input x | TF vector — 32,768 accumulated frequencies |
| Matrix multiply Wx | TF ⊙ bitmask(H) — element-wise product |
| Bias b | Not needed (TF baseline is 0) |
| Activation σ | Observable mask O(θ) — threshold at rank θ |
| Forward pass | rank(H) = Σ TF[b] for b ∈ bitmask(H) |
| Learning rate η | Ingestion rate (how fast TF accumulates) |
| Gradient ∂L/∂W | Bitmask selection (which HLLSet to attend to) |

This is not an analogy — it is a structural isomorphism. Both systems compute a
weighted sum of input signals and apply a threshold. The difference is that HLLSet
"weights" are content-addressed, immutable, and binary, while ONNX weights are
named, mutable, and continuous.

### 4.2 The Fisher Matrix / Hessian Parallel

ONNX training computes a Hessian (or approximation) to guide optimization. HLLSet
computes a Fisher-like matrix F_{bb'} that counts how many temporal layers have
both bits b and b' set. Both matrices capture co-occurrence structure:

| ONNX Hessian | HLLSet Fisher Matrix |
| ------------- | --------------------- |
| ∂²L/∂w_i∂w_j | $Σ_i B^(i)_b · B^(i)_{b'}$ |
| 2nd-order optimization | Systemic vs noise detection |
| Requires backpropagation | Pure popcount |
| Dense, float | Sparse, integer |

### 4.3 Composable Sub-Graphs

Both architectures support reusable computational units:

| ONNX Functions | HLLSet Colon-Definitions |
| --------------- | -------------------------- |
| `def relu(x): return max(0,x)` | `: RELU 0 MAX ;` |
| Type-checked inputs/outputs | Stack-based (no type system) |
| Versioned (opset) | Versioned (immutable commit chain) |
| Compiled to runtime ops | Compiled to Lua functions |

### 4.4 Graph Optimization

Both apply graph-level optimizations, though through different mechanisms:

| ONNX Optimization | HLLSet Lattice Equivalent |
| ------------------- | -------------------------- |
| Constant folding | Idempotence (A ∪ A = A — skip computation) |
| Operator fusion | Composition (h_n ∘ ... ∘ h_1 = single hash chain) |
| Dead code elimination | Observable mask (O(θ) hides low-rank HLLSets) |
| Quantization (FP32→INT8) | Not needed (already binary) |

---

## 5. Divergent Design Philosophies

### 5.1 Probability vs Determinism

ONNX operates in the continuous domain: outputs are probabilities, weights are
real numbers, convergence is approximate. HLLSet operates in the discrete domain:
bits are set or unset, keys are exact hashes, convergence is provable. This is
not a question of which is "better" — they serve different purposes. Probability
enables rich function approximation; determinism enables content-addressing and
distributed convergence without coordination.

### 5.2 External vs Internal Learning

ONNX separates model architecture (the graph) from learning (the training loop).
The model is a static structure; learning happens externally through gradient
descent on batches. HLLSet embeds learning within the architecture: ingestion
updates TF, which reshuffles ranks, which changes the observable mask. There is
no training loop — the lattice is always learning.

### 5.3 Named vs Content-Addressed Identity

This is the deepest philosophical divergence. ONNX names are human-assigned,
hierarchical, and semantically meaningful ("encoder.layer3.attention.qkv.weight").
HLLSet keys are machine-assigned, flat, and semantically opaque ("h:a3f82c1d...").
The trade: ONNX names are debuggable but fragile (rename a layer = break the
model); HLLSet keys are inscrutable but robust (same content = same key, always).

### 5.4 Snapshot vs Lattice World Model

ONNX models are snapshots — they represent what the model learned from a training
corpus at a specific point in time. HLLSet lattices are world models — they
continuously accumulate observations, compress history through temporal layers,
and enable queries against any past state. A snapshot is frozen; a lattice is alive.

---

## 6. Practical Implications

### 6.1 When ONNX Excels

- **Function approximation:** Any differentiable function can be learned
- **Perceptual tasks:** Vision, speech, language — domains where continuous
  representations capture subtle patterns
- **Batch processing:** Large static datasets where training cost is amortized
- **GPU ecosystem:** Mature tooling, optimized runtimes, cloud deployment
- **Standardized deployment:** ONNX Runtime provides a single execution target
  across hardware backends

### 6.2 When HLLSet Lattice Excels

- **Streaming contexts:** Continuous observation streams where batch retraining
  is impossible
- **Edge deployment:** FPGA targets where power budget precludes GPUs
- **Distributed systems:** Multi-node deployments where consensus protocols are
  a bottleneck
- **Tamper-proof audit trails:** Content-addressing provides cryptographic provenance
- **Cross-domain reasoning:** Bridge mechanism transfers structure between domains
  without retraining
- **Chinese/character-based languages:** Fixed token set enables deterministic
  tokenization without BPE/subword complexity

### 6.3 Potential Integration Points

The architectures are not mutually exclusive. Several integration patterns emerge:

1. **ONNX as inference engine, HLLSet as memory:** ONNX handles perception
   (vision, speech); HLLSet serves as long-term content-addressed memory with
   temporal depth.

2. **HLLSet as pre-filter for ONNX:** HLLSet's BSS similarity quickly identifies
   relevant context windows; ONNX performs deep inference on the narrowed scope.

3. **ONNX weights stored as HLLSets:** Model weights are content-addressed and
   versioned through the lattice; model deployment becomes an IICA operation.

4. **Hybrid FPGA pipeline:** HLLSet handles bitwise operations on FPGA fabric;
   ONNX-compatible float units handle remaining continuous computation on the
   same chip.

---

## 7. Conclusion

ONNX and HLLSet Lattice represent two points in a larger design space for
computational intelligence:

```text
    Continuous ◄────────────────► Discrete
         │                          │
    Float tensors              Binary bitmasks
    Gradient descent           Rank rearrangement  
    Named identity             Content-addressed
    Static snapshot            Live world model
    GPU datacenter             FPGA edge
        │                          │
       ONNX                    HLLSet Lattice
```

Neither is universally superior. ONNX dominates the current AI landscape because
continuous function approximation with gradient descent has proven remarkably
effective for perceptual tasks. HLLSet addresses a complementary set of problems:
streaming contexts, distributed consistency, edge deployment, and cross-domain
reasoning — areas where the GPU/gradient paradigm shows its structural limits.

The most promising direction may not be choosing one over the other, but
understanding when each architecture's foundational assumptions match the
problem's requirements. The convergence patterns identified in §4 suggest that
hybrid architectures are not only possible but natural: a lattice-structured
content-addressed memory feeding a continuous inference engine, or a GPU-trained
perception model whose outputs are ingested into a temporal pyramid for
streaming context tracking.

The categorical shift HLLSet Algebra represents is not a replacement for neural
networks — it is a recognition that some computational problems (identity,
convergence, temporal depth, cross-domain transfer) are better solved through
algebraic structure than through gradient descent. The architectures that combine
both will define the next generation of intelligent systems.

---

## References

1. ONNX Specification: <https://github.com/onnx/onnx/blob/main/docs/IR.md>
2. ONNX Operators: <https://github.com/onnx/onnx/blob/main/docs/Operators.md>
3. STANDARD.md — HLLSet Algebra authoritative specification (hllset-next/_DOCS/dev/)
4. Flajolet, P., et al. "HyperLogLog: the analysis of a near-optimal cardinality
   estimation algorithm" (2007)
5. SELF_REPROGRAMMING_ARCHITECTURE.md — Lattice-as-neural-network mapping (§4.13)
6. GREYLOCK_EXECUTIVE_SUMMARY.md — HLLSet Algebra business and technical context
