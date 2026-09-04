# hllset_cortex/filter.py
"""
HLLSetFilter — semantic compressor for ds-ocr encoding streams.

Receives encoding IDs from ds-ocr's vision encoder, processes them
through the HLLSet Algebra pipeline, and returns restored encoding
IDs for the decoder.

Architecture (per STANDARD.md):
    ds-ocr encoding IDs → hllset_py.Tokenizer (standard pipeline)
      → MurmurHash3 → HLLSet (32,768-bit bitmap)
        → ∩ gate_TF HLLSet (decoder vocabulary filter)
          → TokenLut (monotonic TF accumulation)
            → materialize (TF-ranked disambiguation)
              → restored encoding IDs → ds-ocr Decoder

The gate_TF HLLSet is a content-addressed system-global built from the
decoder's BPE vocabulary. It filters invalid bit positions at the lattice
level via intersection — no Python-level ID matching.

The LUT is a persistent singleton — created once, grows monotonically
as encoding streams are ingested. Per STANDARD.md Appendix D: TF is earned
through experience, never seeded with equal-weight external vocabulary.
"""

from typing import List, Optional, Dict
from dataclasses import dataclass, field
import statistics

import hllset_py

from hllset_cortex.domain import default_tokenizer


@dataclass
class FilterStats:
    """Statistics from one filter pass."""
    input_tokens: int = 0
    hllset_popcount: int = 0
    gate_popcount: int = 0
    output_tokens: int = 0
    compression_ratio: float = 0.0
    roundtrip_match: int = 0
    roundtrip_total: int = 0


@dataclass
class FilterResult:
    """Output from one HLLSet filter pass."""
    tokens: List[bytes]
    hllset: Optional[hllset_py.HLLSet] = None
    filtered_hllset: Optional[hllset_py.HLLSet] = None
    stats: FilterStats = field(default_factory=FilterStats)
    lut_size: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and len(self.tokens) > 0

    @property
    def token_strings(self) -> List[str]:
        """Tokens as strings (for display)."""
        return [t if isinstance(t, str) else t.decode("utf-8", errors="replace") for t in self.tokens]


@dataclass
class HLLSetFilter:
    """HLLSet-based semantic filter for ds-ocr encoding streams.

    Maintains a single TokenLut that accumulates encoding ID → hash
    mappings with monotonic TF across all ingested documents.

    Optional gate_TF HLLSet filters invalid encoding IDs at the
    lattice level via intersection. Built once from the decoder's
    vocabulary and never changes (IICA: immutable gate).

    Usage:
        filt = HLLSetFilter()
        # Process encoding IDs from ds-ocr
        result = filt.process(encoding_ids_bytes)
        # With gate:
        filt.gate_hllset = gate_hllset
        result = filt.process(encoding_ids_bytes)
        # ... more streams ...
        print(filt.summary())
    """

    _lut: Optional[hllset_py.TokenLut] = None
    _history: List[FilterStats] = field(default_factory=list)
    _gate_hllset: Optional[hllset_py.HLLSet] = None
    _tokenizer: Optional[hllset_py.Tokenizer] = None

    @property
    def lut(self) -> hllset_py.TokenLut:
        """The persistent TokenLut singleton.

        Created on first access. Never rebuilt — TF accumulates
        monotonically across all encoding streams.
        """
        if self._lut is None:
            self._lut = hllset_py.TokenLut()
        return self._lut

    @property
    def tokenizer(self) -> hllset_py.Tokenizer:
        """The standard tokenizer (lazy-init from domain config)."""
        if self._tokenizer is None:
            self._tokenizer = default_tokenizer()
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tok: hllset_py.Tokenizer):
        self._tokenizer = tok

    @property
    def history(self) -> List[FilterStats]:
        return self._history

    @property
    def gate_hllset(self) -> Optional[hllset_py.HLLSet]:
        """The content-addressed gate_TF HLLSet.

        Built once from the decoder's vocabulary. Intersection
        with this HLLSet filters invalid bit positions at the lattice
        level — encoding IDs the decoder can't process have their bits
        removed. None means no gate (all IDs pass through).
        """
        return self._gate_hllset

    @gate_hllset.setter
    def gate_hllset(self, hllset: Optional[hllset_py.HLLSet]):
        self._gate_hllset = hllset

    def process(self, data: bytes) -> FilterResult:
        """Run one filter pass: tokenize → HLLSet → gate ∩ → LUT → materialize.

        1. hllset_py.Tokenizer processes raw encoding IDs (standard pipeline)
        2. HLLSet.from_tokens() hashes via MurmurHash3 into 32,768-bit bitmap
        3. Gate intersection: if gate_hllset is set, filter invalid bit positions
        4. TokenLut.record_all() increments TF for ALL tokens (monotonic CRDT)
        5. materialize() returns highest-TF token at each active bit position

        Args:
            data: Raw encoding ID bytes from ds-ocr vision encoder
        """
        try:
            tokens = self.tokenizer.tokenize(data)
        except Exception as e:
            return FilterResult(tokens=[], error=str(e))

        if not tokens:
            return FilterResult(
                tokens=[],
                stats=FilterStats(input_tokens=0),
                lut_size=self.lut.len(),
            )

        # HLLSet fingerprint from ALL tokenized encoding IDs
        hllset = hllset_py.HLLSet.from_token_bytes(tokens)

        # Accumulate TF for ALL tokens (monotonic CRDT, pre-gate)
        self.lut.record_all_bytes(tokens)

        # Gate intersection: filter invalid bit positions
        if self._gate_hllset is not None:
            filtered = hllset.intersection(self._gate_hllset)
        else:
            filtered = hllset

        # Materialize from filtered HLLSet: TF-ranked disambiguation
        materialized = hllset_py.materialize(filtered, self.lut)

        # Extract unigrams only for input count (no n-gram tokens)
        unigrams = self.tokenizer.tokenize(data) if self.tokenizer else tokens
        # Use first n tokens as baseline (unigrams come first in n-gram output)
        n_unigrams = len([t for t in tokens if b"\0" not in t])

        matched = [t for t in materialized if t.encode("utf-8", errors="replace") in tokens]

        stats = FilterStats(
            input_tokens=n_unigrams,
            hllset_popcount=hllset.popcount(),
            gate_popcount=filtered.popcount(),
            output_tokens=len(materialized),
            compression_ratio=len(materialized) / max(n_unigrams, 1),
            roundtrip_match=len(matched),
            roundtrip_total=len(materialized),
        )
        self._history.append(stats)

        return FilterResult(
            tokens=materialized,
            hllset=hllset,
            filtered_hllset=filtered,
            stats=stats,
            lut_size=self.lut.len(),
        )

    def process_text(self, text: str) -> FilterResult:
        """Convenience: process text through the standard tokenizer.

        For use with natural language text (not raw encoding IDs).
        """
        data = text.encode("utf-8")
        return self.process(data)

    def process_ordered(
        self,
        data: bytes,
        start_marker: str = "<S>",
        end_marker: str = "</S>",
    ) -> FilterResult:
        """Run one filter pass with De Bruijn ordered reconstruction.

        Uses a boundary-padded bigram tokenizer so that
        materialize_debruijn() can reconstruct token sequence order
        via Eulerian path traversal.

        1. Tokenize with debruijn_tokenizer (pad + bigrams only)
        2. HLLSet fingerprint
        3. Gate intersection
        4. LUT TF accumulation
        5. materialize_debruijn() — ordered reconstruction

        Args:
            data: Raw encoding ID bytes
            start_marker: Boundary start token (default "<S>")
            end_marker: Boundary end token (default "</S>")
        """
        from hllset_cortex.domain import debruijn_tokenizer

        db_tok = debruijn_tokenizer(start_marker, end_marker)
        try:
            tokens = db_tok.tokenize(data)
        except Exception as e:
            return FilterResult(tokens=[], error=str(e))

        if not tokens:
            return FilterResult(
                tokens=[],
                stats=FilterStats(input_tokens=0),
                lut_size=self.lut.len(),
            )

        hllset = hllset_py.HLLSet.from_token_bytes(tokens)
        self.lut.record_all_bytes(tokens)

        if self._gate_hllset is not None:
            filtered = hllset.intersection(self._gate_hllset)
        else:
            filtered = hllset

        # De Bruijn ordered reconstruction
        ordered = hllset_py.materialize_debruijn(
            filtered, self.lut, start_marker, end_marker
        )

        n_unigrams = len([t for t in tokens if b"\0" not in t])

        stats = FilterStats(
            input_tokens=n_unigrams,
            hllset_popcount=hllset.popcount(),
            gate_popcount=filtered.popcount(),
            output_tokens=len(ordered),
            compression_ratio=len(ordered) / max(n_unigrams, 1),
            roundtrip_match=0,  # debruijn path, match not applicable
            roundtrip_total=len(ordered),
        )
        self._history.append(stats)

        return FilterResult(
            tokens=ordered,
            hllset=hllset,
            filtered_hllset=filtered,
            stats=stats,
            lut_size=self.lut.len(),
        )

    def process_text_ordered(
        self,
        text: str,
        start_marker: str = "<S>",
        end_marker: str = "</S>",
    ) -> FilterResult:
        """Convenience: process text through the De Bruijn tokenizer.

        Returns tokens in reconstructed sequence order.
        """
        data = text.encode("utf-8")
        return self.process_ordered(data, start_marker, end_marker)

    def process_batch(self, items: List[bytes]) -> List[FilterResult]:
        """Process multiple encoding streams sequentially.

        Each stream feeds the shared LUT — TF accumulates across
        the batch, improving disambiguation for later streams.
        """
        return [self.process(data) for data in items]

    def summary(self) -> Dict:
        """Convergence statistics across all processed streams."""
        if not self._history:
            return {
                "streams": 0,
                "lut_size": self.lut.len(),
                "lut_positions": self.lut.position_count(),
            }
        return {
            "streams": len(self._history),
            "lut_size": self.lut.len(),
            "lut_positions": self.lut.position_count(),
            "gate_popcount": self._gate_hllset.popcount() if self._gate_hllset else 0,
            "avg_input_tokens": statistics.mean(
                s.input_tokens for s in self._history
            ),
            "avg_hllset_popcount": statistics.mean(
                s.hllset_popcount for s in self._history
            ),
            "avg_gate_popcount": statistics.mean(
                s.gate_popcount for s in self._history
            ),
            "avg_output_tokens": statistics.mean(
                s.output_tokens for s in self._history
            ),
            "avg_roundtrip_match": statistics.mean(
                s.roundtrip_match / max(s.roundtrip_total, 1)
                for s in self._history
            ),
        }
