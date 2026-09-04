//! Python bindings for the standard HLLSet Algebra tokenizer.
//!
//! Wraps the vendored hllset-dsl Tokenizer — the canonical tokenization
//! pipeline for all HLLSet Algebra applications.

use crate::dsl_tokenizer::{Normalizer, Tokenizer};
use crate::pattern::Pattern;
use pyo3::prelude::*;

/// Standard HLLSet Algebra tokenizer — composable pipeline.
///
/// Pipeline: Bytes → [Pattern Match] → Tokens → [Normalize] → [N-grams] → [Boundary Pad]
///
/// This is the canonical tokenizer from hllset-dsl, exposed to Python.
/// All HLLSet Algebra applications should use this standardized tokenizer.
///
/// Usage:
///     tok = Tokenizer()                    # default: ASCII word pattern
///     tok = Tokenizer.word_pattern()       # explicit word pattern
///     tok = Tokenizer.lowercase()          # add lowercase normalizer
///     tok = Tokenizer.ngrams(1, 3)         # 1-gram, 2-gram, 3-gram
///     tok = Tokenizer.pattern(b"0123456789")  # custom: digits only
///
///     tokens = tok.tokenize(b"hello world")   # → ["hello", "world", "hello\0world"]
#[pyclass(name = "Tokenizer")]
#[derive(Clone)]
pub struct PyTokenizer {
    inner: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    /// Create a new tokenizer with word pattern (ASCII letters + digits).
    #[new]
    fn new() -> Self {
        PyTokenizer {
            inner: Tokenizer::new(),
        }
    }

    /// Use the default word pattern (ASCII letters + digits, no n-grams, no normalization).
    #[staticmethod]
    fn word_pattern() -> Self {
        PyTokenizer {
            inner: Tokenizer::new().word_pattern(),
        }
    }

    /// Set a custom pattern from allowed bytes.
    ///
    /// Example: `Tokenizer.new().pattern(b"0123456789")` matches digits only.
    fn pattern(&self, allowed: Vec<u8>) -> Self {
        PyTokenizer {
            inner: self.inner.clone().pattern(Pattern::span(&allowed)),
        }
    }

    /// Add lowercase normalizer (ASCII only).
    fn lowercase(&self) -> Self {
        PyTokenizer {
            inner: self.inner.clone().lowercase(),
        }
    }

    /// Add trim normalizer (removes leading/trailing ASCII whitespace).
    fn trim(&self) -> Self {
        PyTokenizer {
            inner: self.inner.clone().trim(),
        }
    }

    /// Keep only bytes in the allowed set.
    fn keep_only(&self, allowed: Vec<u8>) -> Self {
        PyTokenizer {
            inner: self.inner.clone().keep_only(&allowed),
        }
    }

    /// Generate n-grams in range [min, max] (inclusive).
    ///
    /// ngrams(1, 1) = unigrams only
    /// ngrams(1, 3) = unigrams + bigrams + trigrams
    /// ngrams(2, 2) = bigrams only
    fn ngrams(&self, min: usize, max: usize) -> Self {
        PyTokenizer {
            inner: self.inner.clone().ngrams(min, max),
        }
    }

    /// Add boundary padding tokens.
    ///
    /// START token is prepended, END token appended before n-gram generation.
    /// This enables n-grams that include start/end boundaries.
    fn pad(&self, start: Vec<u8>, end: Vec<u8>) -> Self {
        PyTokenizer {
            inner: self.inner.clone().pad(&start, &end),
        }
    }

    /// Extract raw tokens from input bytes using the configured pattern.
    ///
    /// Returns list of byte tokens (as Python bytes objects).
    /// No normalization or n-gram generation — just pattern matching.
    fn extract(&self, input: Vec<u8>) -> Vec<Vec<u8>> {
        self.inner.extract(&input)
    }

    /// Tokenize input through the full pipeline: extract → normalize → n-grams.
    ///
    /// Returns list of byte tokens ready for HLLSet.from_tokens().
    /// N-grams use NUL (0x00) as separator — the standard HLLSet convention.
    fn tokenize(&self, input: Vec<u8>) -> Vec<Vec<u8>> {
        self.inner.tokenize(&input)
    }

    /// Tokenize a string (convenience — encodes to UTF-8 first).
    fn tokenize_str(&self, text: &str) -> Vec<Vec<u8>> {
        self.inner.tokenize(text.as_bytes())
    }

    fn __repr__(&self) -> String {
        format!(
            "Tokenizer(ngrams={}..{}, pattern=span)",
            self.inner.ngram_min(),
            self.inner.ngram_max(),
        )
    }
}

// Add accessor methods to the vendored Tokenizer
impl Tokenizer {
    pub fn ngram_min(&self) -> usize {
        self.ngram_min
    }
    pub fn ngram_max(&self) -> usize {
        self.ngram_max
    }
}
