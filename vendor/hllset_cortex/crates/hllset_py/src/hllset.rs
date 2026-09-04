//! Python bindings for `hllset_core::HLLSet`.
//!
//! Wraps the core HLLSet struct with lattice operations (union, intersection, BSS)
//! and content addressing.

use hllset_core::HLLSet;
use pyo3::prelude::*;

/// A content-addressed HLLSet fingerprint with set-algebra operations.
///
/// HLLSets are 32,768-bit bitmap tensors (1024 registers × 32 bits).
/// They form a bounded distributive lattice under union (OR) and
/// intersection (AND) — associative, commutative, and idempotent.
///
/// IICA properties:
/// - **Idempotent**: same tokens → same HLLSet, every time
/// - **Immutable**: once created, never changes
/// - **Content-Addressed**: key = SHA1 of serialized bytes
#[pyclass(name = "HLLSet")]
#[derive(Clone)]
pub struct PyHLLSet {
    pub(crate) inner: HLLSet,
}

#[pymethods]
impl PyHLLSet {
    /// Create a new empty HLLSet.
    #[new]
    fn new() -> Self {
        PyHLLSet {
            inner: HLLSet::new(),
        }
    }

    /// Create an HLLSet from a list of token strings.
    ///
    /// Each token is MurmurHash3-hashed and the resulting bit position
    /// is set in the 32,768-bit bitmap.
    #[staticmethod]
    fn from_tokens(tokens: Vec<String>) -> Self {
        PyHLLSet {
            inner: HLLSet::from_tokens(&tokens),
        }
    }

    /// Create an HLLSet from raw byte tokens (from Tokenizer.tokenize()).
    ///
    /// Each byte token is MurmurHash3-hashed directly — no UTF-8 decoding.
    /// This is the preferred path when using the standard hllset-dsl Tokenizer.
    #[staticmethod]
    fn from_token_bytes(tokens: Vec<Vec<u8>>) -> Self {
        PyHLLSet {
            inner: HLLSet::from_tokens(&tokens),
        }
    }

    /// SHA-1 content key: "h:<40 hex chars>".
    ///
    /// Deterministic — same bit pattern always produces the same key.
    fn content_key(&self) -> String {
        self.inner.content_key()
    }

    /// Number of set bits in the bitmap (population count).
    fn popcount(&self) -> u64 {
        self.inner.popcount()
    }

    /// Estimated cardinality (Horvitz-Thompson estimator).
    fn cardinality(&self) -> f64 {
        self.inner.cardinality()
    }

    /// Is this HLLSet empty (no bits set)?
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Active bit positions: list of (register, trailing_zeros) tuples.
    ///
    /// Used for LUT lookups during materialization.
    fn active_positions(&self) -> Vec<(u32, u32)> {
        self.inner.active_positions()
    }

    /// Number of registers with at least one bit set.
    fn non_zero_registers(&self) -> u32 {
        self.inner.non_zero_registers()
    }

    // ── Lattice operations ──

    /// Union (join): A ∪ B — bitwise OR.
    fn union(&self, other: &PyHLLSet) -> PyHLLSet {
        PyHLLSet {
            inner: self.inner.union(&other.inner),
        }
    }

    /// Intersection (meet): A ∩ B — bitwise AND.
    fn intersection(&self, other: &PyHLLSet) -> PyHLLSet {
        PyHLLSet {
            inner: self.inner.intersection(&other.inner),
        }
    }

    /// Difference: A \ B — bits in A but not in B.
    fn difference(&self, other: &PyHLLSet) -> PyHLLSet {
        PyHLLSet {
            inner: self.inner.difference(&other.inner),
        }
    }

    // ── Similarity measures ──

    /// BSSτ inclusion: |self ∩ other| / |other|.
    ///
    /// Answers: "How much of other's content is also in self?"
    /// Returns 1.0 if other is empty.
    fn bss_inclusion(&self, other: &PyHLLSet) -> f64 {
        self.inner.bss_inclusion(&other.inner)
    }

    /// Jaccard similarity: |A ∩ B| / |A ∪ B|.
    ///
    /// Returns 1.0 if both sets are empty.
    fn jaccard(&self, other: &PyHLLSet) -> f64 {
        self.inner.jaccard_similarity(&other.inner)
    }

    // ── Python dunder methods ──

    fn __repr__(&self) -> String {
        format!(
            "HLLSet(key={}..., popcount={}, card={:.1})",
            &self.inner.content_key()[..24],
            self.inner.popcount(),
            self.inner.cardinality()
        )
    }

    fn __len__(&self) -> usize {
        self.inner.popcount() as usize
    }
}
