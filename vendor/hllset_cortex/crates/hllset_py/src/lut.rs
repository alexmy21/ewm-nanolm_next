//! Python bindings for TokenLut with monotonic TF tracking.
//!
//! The TokenLut is the dynamic layer of the HLLSet lattice — it maps
//! bit positions back to candidate tokens and tracks term frequency.
//! Per STANDARD.md §3.1: TF is stored, rank is derived.
//!
//! Key invariant (STANDARD.md Appendix D):
//!   A LUT may only contain tokens whose accumulated TF reflects
//!   actual experience. Never seed with equal-TF external vocabulary.

use crate::hllset::PyHLLSet;
use hllset_core::core::hashing::{murmur3_hash, token_to_position};
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

// ── TokenLut ─────────────────────────────────────────────────────────────

/// Token Lookup Table with monotonic TF tracking.
///
/// Maps (register, trailing_zeros) bit positions back to candidate tokens,
/// with per-token term frequency that is monotonically non-decreasing.
///
/// The LUT is the lattice's "understanding" of its domain. TF accumulates
/// as documents are ingested — never reset, never decreased.
///
/// **LUT Initialization Constraint (STANDARD.md Appendix D):**
/// Never populate with equal-TF vocabulary (causes random materialization).
/// Valid states: cold start (empty), lattice-covered (from HLLSet corpus),
/// or donor transfer (from experienced LUT).
#[pyclass(name = "TokenLut")]
pub struct PyTokenLut {
    /// (reg, zeros) → set of candidate tokens
    index: HashMap<(u32, u32), Vec<String>>,
    /// token → (reg, zeros) — forward index
    forward: HashMap<String, (u32, u32)>,
    /// token → term frequency — monotonic CRDT, never decreases
    tf: HashMap<String, u64>,
}

#[pymethods]
impl PyTokenLut {
    /// Create an empty LUT (cold start).
    ///
    /// This is the recommended initial state per STANDARD.md Appendix D.
    /// The LUT accumulates vocabulary and TF through document ingestion.
    #[new]
    fn new() -> Self {
        PyTokenLut {
            index: HashMap::new(),
            forward: HashMap::new(),
            tf: HashMap::new(),
        }
    }

    /// Record one occurrence of a token — increments its TF.
    ///
    /// If the token is new, it is added to the LUT with TF=1.
    /// If already present, TF is incremented.
    /// Tokens are hashed with MurmurHash3 to determine their bit position.
    fn record(&mut self, token: &str) {
        let pos = token_to_position(token.as_bytes());
        let t = token.to_string();

        // Update reverse index
        self.index
            .entry(pos)
            .or_default()
            .push(t.clone());

        // Update forward index
        self.forward.entry(t.clone()).or_insert(pos);

        // Increment TF (monotonic)
        *self.tf.entry(t).or_insert(0) += 1;
    }

    /// Record multiple tokens.
    fn record_all(&mut self, tokens: Vec<String>) {
        for t in &tokens {
            self.record(t);
        }
    }

    /// Record raw byte tokens (from Tokenizer.tokenize()).
    fn record_all_bytes(&mut self, tokens: Vec<Vec<u8>>) {
        for t in &tokens {
            let s = String::from_utf8_lossy(t).to_string();
            self.record(&s);
        }
    }

    /// Get the current TF for a token.
    ///
    /// Returns 0 for unknown tokens.
    fn tf(&self, token: &str) -> u64 {
        self.tf.get(token).copied().unwrap_or(0)
    }

    /// Look up candidate tokens at a given bit position (register, trailing_zeros).
    ///
    /// Returns list of token strings that hash to this position.
    /// Multiple tokens may map to the same position due to hash collisions.
    fn lookup_position(&self, reg: u32, tz: u32) -> Vec<String> {
        self.index
            .get(&(reg, tz))
            .cloned()
            .unwrap_or_default()
    }

    /// Get the bit position for a known token.
    fn position_of(&self, token: &str) -> Option<(u32, u32)> {
        self.forward.get(token).copied()
    }

    /// Number of unique tokens in the LUT.
    fn len(&self) -> usize {
        self.forward.len()
    }

    /// Whether the LUT has no tokens.
    fn is_empty(&self) -> bool {
        self.forward.is_empty()
    }

    /// Number of unique bit positions occupied.
    fn position_count(&self) -> usize {
        self.index.len()
    }

    /// Get all tokens sorted by descending TF.
    fn ranked_tokens(&self) -> Vec<(String, u64)> {
        let mut entries: Vec<(String, u64)> = self
            .tf
            .iter()
            .map(|(t, &f)| (t.clone(), f))
            .collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries
    }

    fn __repr__(&self) -> String {
        format!("TokenLut({} tokens, {} positions)", self.forward.len(), self.index.len())
    }
}

// ── Materialization ─────────────────────────────────────────────────────

/// Materialize an HLLSet back to tokens via the LUT, TF-ranked.
///
/// For each active bit position in the HLLSet, resolves to the token
/// with the highest TF among all candidates at that position.
///
/// This is the primary disambiguation mechanism: when multiple tokens
/// collide at the same hash position, the one with the most experience
/// (highest TF) wins.
#[pyfunction]
pub fn materialize(hllset: &PyHLLSet, lut: &PyTokenLut) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut result = Vec::new();

    for (reg, tz) in hllset.inner.active_positions() {
        let candidates = lut.lookup_position(reg, tz);
        if candidates.is_empty() {
            continue;
        }

        // Select highest-TF token at this position
        let best = candidates
            .iter()
            .max_by_key(|t| lut.tf(t))
            .cloned();

        if let Some(token) = best {
            if seen.insert(token.clone()) {
                result.push(token);
            }
        }
    }

    result
}

/// Materialize only the top-N tokens by TF across ALL active positions.
#[pyfunction]
pub fn materialize_top_n(hllset: &PyHLLSet, lut: &PyTokenLut, n: usize) -> Vec<String> {
    let mut candidates: Vec<(String, u64)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for (reg, tz) in hllset.inner.active_positions() {
        for token in lut.lookup_position(reg, tz) {
            if seen.insert(token.clone()) {
                let tf = lut.tf(&token);
                candidates.push((token, tf));
            }
        }
    }

    candidates.sort_by(|a, b| b.1.cmp(&a.1));
    candidates.truncate(n);
    candidates.into_iter().map(|(t, _)| t).collect()
}

/// De Bruijn graph reconstruction from bigram tokens.
///
/// Given boundary-padded bigrams (from a tokenizer with `.pad(start, end).ngrams(2,2)`),
/// builds a De Bruijn graph where each bigram "a\0b" becomes an edge a→b, then
/// finds an Eulerian path from start_marker to end_marker to reconstruct token order.
///
/// # Arguments
/// * `hllset` — the HLLSet fingerprint containing the bigram bits
/// * `lut` — the TokenLut mapping bit positions back to token strings
/// * `start_marker` — boundary start token (e.g., "<S>")
/// * `end_marker` — boundary end token (e.g., "</S>")
///
/// # Returns
/// A Vec of token strings in reconstructed order, or empty if no path found.
#[pyfunction]
pub fn materialize_debruijn(
    hllset: &PyHLLSet,
    lut: &PyTokenLut,
    start_marker: &str,
    end_marker: &str,
) -> Vec<String> {
    let positions = hllset.inner.active_positions();

    // Step 1: Collect bigrams from LUT at active HLLSet positions
    // Each bigram "prefix\0suffix" becomes an edge prefix→suffix
    let mut edges: Vec<(String, String)> = Vec::new(); // (prefix, suffix)
    let mut seen_edges: HashSet<(String, String)> = HashSet::new();

    for (reg, tz) in &positions {
        for token in lut.lookup_position(*reg, *tz) {
            // Only process bigrams (contain NUL separator)
            if let Some(nul_pos) = token.find('\0') {
                let prefix = token[..nul_pos].to_string();
                let suffix = token[nul_pos + 1..].to_string();
                let edge = (prefix, suffix);
                if seen_edges.insert(edge.clone()) {
                    edges.push(edge);
                }
            }
        }
    }

    if edges.is_empty() {
        return Vec::new();
    }

    // Step 2: Build adjacency list (prefix → list of suffixes)
    let mut adj: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();
    for (prefix, suffix) in &edges {
        adj.entry(prefix.clone())
            .or_default()
            .push(suffix.clone());
    }

    // Step 3: Greedy DFS path from start_marker to end_marker
    let start = start_marker.to_string();
    let end = end_marker.to_string();
    let mut path: Vec<String> = Vec::new();
    let mut current = start.clone();
    let mut visited: HashSet<(String, String)> = HashSet::new();

    path.push(current.clone());

    for _ in 0..10000 {
        // safety limit
        if current == end {
            return path;
        }

        if let Some(nexts) = adj.get(&current) {
            let mut found = false;
            for next in nexts {
                let edge_key = (current.clone(), next.clone());
                if !visited.contains(&edge_key) {
                    visited.insert(edge_key);
                    path.push(next.clone());
                    current = next.clone();
                    found = true;
                    break;
                }
            }
            if !found {
                return path; // dead end — return what we have
            }
        } else {
            return path; // no outgoing edges
        }
    }

    path
}

// ── Hashing utilities ───────────────────────────────────────────────────

/// MurmurHash3 64-bit hash of a token string (seed 0).
///
/// This is the same hash function used internally by HLLSet::from_tokens.
/// Useful for pre-computing hashes or debugging.
#[pyfunction]
pub fn murmur3_hash_py(token: &str) -> u64 {
    murmur3_hash(token.as_bytes())
}

/// Decompose a token into its (register, trailing_zeros) bit position.
///
/// Returns (reg, tz) tuple. `reg` is in 0..1023, `tz` in 0..31.
/// Same algorithm used internally by the TokenLut and HLLSet.
#[pyfunction]
pub fn token_to_position_py(token: &str) -> (u32, u32) {
    token_to_position(token.as_bytes())
}
