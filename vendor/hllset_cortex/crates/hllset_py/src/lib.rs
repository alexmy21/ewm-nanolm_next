//! # hllset_py — Python bindings for hllset-next
//!
//! Thin PyO3 wrapper around hllset-core providing:
//!
//! - [`PyHLLSet`] — content-addressed HLLSet fingerprints with lattice operations
//! - [`PyTokenLut`] — reverse index with monotonic TF tracking
//! - Materialization functions
//!
//! This crate is the Python interface for the HLLSet Algebra platform.
//! It wraps hllset-next directly — no caal-llm dependency.

use pyo3::prelude::*;

mod hllset;
mod lut;
mod tokenizer;

// Vendored hllset-dsl tokenizer (standard HLLSet Algebra tokenizer)
#[path = "../vendor/hllset-dsl/pattern.rs"]
mod pattern;
#[path = "../vendor/hllset-dsl/tokenizer.rs"]
mod dsl_tokenizer;

use hllset::PyHLLSet;
use lut::{materialize, materialize_debruijn, materialize_top_n, murmur3_hash_py, token_to_position_py, PyTokenLut};
use tokenizer::PyTokenizer;

/// The `hllset_py` Python module.
#[pymodule]
fn hllset_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ── HLLSet ──
    m.add_class::<PyHLLSet>()?;

    // ── TokenLut ──
    m.add_class::<PyTokenLut>()?;

    // ── Tokenizer (standard hllset-dsl pipeline) ──
    m.add_class::<PyTokenizer>()?;

    // ── Materialization ──
    m.add_function(wrap_pyfunction!(materialize, m)?)?;
    m.add_function(wrap_pyfunction!(materialize_top_n, m)?)?;
    m.add_function(wrap_pyfunction!(materialize_debruijn, m)?)?;

    // ── Hashing utilities ──
    m.add_function(wrap_pyfunction!(murmur3_hash_py, m)?)?;
    m.add_function(wrap_pyfunction!(token_to_position_py, m)?)?;

    Ok(())
}
