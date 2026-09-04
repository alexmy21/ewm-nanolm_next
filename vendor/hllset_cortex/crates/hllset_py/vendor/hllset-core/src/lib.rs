//! Core HLLSet algebra engine.
//!
//! This crate provides the fundamental HLLSet data structure along with:
//!
//! - **Set operations**: union (∪), intersection (∩), difference (\), XOR (⊕)
//! - **Cardinality estimation**: Horvitz-Thompson estimator for bitmap registers
//! - **Hashing**: MurmurHash3 (seeded/unseeded) for token inscription
//! - **Content addressing**: deterministic SHA-1 keys (`h:`, `c:`) for idempotent storage
//! - **BSS morphisms**: Bell State Similarity — inclusion, exclusion, and morphism checks
//! - **Serialization**: Roaring bitmap compression with serde support

pub mod core;

pub use core::bss;
pub use core::cardinality;
pub use core::content_addr;
pub use core::hashing;
pub use core::hllset::{HLLSet, BITS_PER_REG, M, P};
pub use core::operations;
pub use core::serialization;
