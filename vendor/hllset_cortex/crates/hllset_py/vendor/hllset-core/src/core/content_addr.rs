//! Content-addressable key generation for HLLSet.
//!
//! Every HLLSet has a deterministic content key derived from its data.
//! This module provides key generation for both:
//!
//! - **Heterogeneous data** (`h:` keys): n-gram tokenized byte sequences
//! - **Homogeneous data** (`c:` keys): catalog/enumerable values with
//!   ontological identifiers
//!
//! ## Key formats
//!
//! | Prefix | Type          | Input                        | Example           |
//! |--------|---------------|------------------------------|-------------------|
//! | `h:`   | Heterogeneous | Serialized HLLSet bitmap     | `h:a3f82c1d...`  |
//! | `c:`   | Homogeneous   | Ontological SHA1 of catalog  | `c:b7e91d4f...`  |

use crate::core::hashing::sha1_hex;

/// Generate a heterogeneous content key from token byte sequences.
///
/// Tokens are sorted and deduplicated, joined with null bytes,
/// then SHA-1 hashed. This ensures key stability regardless of
/// insertion order.
///
/// Returns: `h:<sha1>`
pub fn content_key_from_tokens<'a, I, B>(tokens: I) -> String
where
    I: IntoIterator<Item = &'a B>,
    B: AsRef<[u8]> + 'a,
{
    // Collect to owned, sort, dedup
    let mut sorted: Vec<Vec<u8>> = tokens
        .into_iter()
        .map(|t| t.as_ref().to_vec())
        .collect();
    sorted.sort();
    sorted.dedup();

    // Build canonical representation: sorted tokens joined by null
    let mut canonical = Vec::new();
    for (i, token) in sorted.iter().enumerate() {
        if i > 0 {
            canonical.push(0u8);
        }
        canonical.extend_from_slice(token);
    }

    format!("h:{}", sha1_hex(&canonical))
}

/// Generate a homogeneous (catalog) content key.
///
/// For catalog/enumerable data, the key identifies the ontological
/// source (database, table, column) rather than the token content.
///
/// Returns: `c:<sha1>`
pub fn content_key_from_catalog<I, B>(catalog_values: I) -> String
where
    I: IntoIterator<Item = B>,
    B: AsRef<[u8]>,
{
    // Collect to owned, sort, dedup
    let mut sorted: Vec<Vec<u8>> = catalog_values
        .into_iter()
        .map(|v| v.as_ref().to_vec())
        .collect();
    sorted.sort();
    sorted.dedup();

    let mut canonical = Vec::new();
    for (i, val) in sorted.iter().enumerate() {
        if i > 0 {
            canonical.push(0u8);
        }
        canonical.extend_from_slice(val);
    }

    format!("c:{}", sha1_hex(&canonical))
}

/// Generate an ontological catalog key from structural position.
///
/// An ontological SHA1 is derived from the structural position
/// (parent + sequence number), never from names. This provides
/// a stable identifier that doesn't depend on naming conventions.
///
/// Returns a raw SHA1 hex string (no prefix — the consumer adds `c:`).
pub fn ontological_key(parent_sha1: &str, seq: u64) -> String {
    let mut data = parent_sha1.as_bytes().to_vec();
    data.extend_from_slice(&seq.to_le_bytes());
    sha1_hex(&data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_key_from_tokens_deterministic() {
        let k1 = content_key_from_tokens(&[b"hello", b"world"]);
        let k2 = content_key_from_tokens(&[b"world", b"hello"]); // different order
        assert_eq!(k1, k2, "key must be order-independent");
    }

    #[test]
    fn test_content_key_from_tokens_prefix() {
        let key = content_key_from_tokens(&[b"test"]);
        assert!(key.starts_with("h:"));
        assert_eq!(key.len(), 42); // "h:" + 40 hex
    }

    #[test]
    fn test_content_key_from_tokens_dedup() {
        let k1 = content_key_from_tokens(&[b"a", b"a", b"b"]);
        let k2 = content_key_from_tokens(&[b"a", b"b"]);
        assert_eq!(k1, k2, "duplicates should not affect key");
    }

    #[test]
    fn test_content_key_different_for_different_data() {
        let k1 = content_key_from_tokens(&[b"x"]);
        let k2 = content_key_from_tokens(&[b"y"]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_catalog_key_prefix() {
        let key = content_key_from_catalog(&[b"alice@example.com"]);
        assert!(key.starts_with("c:"));
        assert_eq!(key.len(), 42);
    }

    #[test]
    fn test_ontological_key_deterministic() {
        let k1 = ontological_key("a3f82c1d", 42);
        let k2 = ontological_key("a3f82c1d", 42);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_ontological_key_differs_by_seq() {
        let k1 = ontological_key("a3f82c1d", 1);
        let k2 = ontological_key("a3f82c1d", 2);
        assert_ne!(k1, k2);
    }
}
