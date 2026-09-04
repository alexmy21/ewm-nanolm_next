//! Core HLLSet engine.
//!
//! Modules:
//! - `hllset` — the HLLSet data structure (Roaring bitmap)
//! - `hashing` — MurmurHash3 hashing, content key generation
//! - `cardinality` — Horvitz-Thompson cardinality estimator
//! - `operations` — set algebra (∪, ∩, \, ⊕, merge)
//! - `bss` — Bell State Similarity morphisms
//! - `content_addr` — content-addressable key generation
//! - `serialization` — binary serialization/deserialization

pub mod bss;
pub mod cardinality;
pub mod content_addr;
pub mod hashing;
pub mod hllset;
pub mod operations;
pub mod serialization;

// Re-export commonly used items
pub use hllset::{HLLSet, BITS_PER_REG, M, P};
pub use hashing::{murmur3_hash, murmur3_hash_seeded};
