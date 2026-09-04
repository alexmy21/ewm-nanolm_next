# hllset_cortex/pipeline.py
"""
OCRPipeline — black-box HLLSet Cortex integration with DeepSeek-OCR.

The only module external systems interact with. It receives encoded
tokens from DeepSeek-OCR's vision encoder and returns disambiguated
BPE token IDs for the decoder.

Pipeline:
    OCR text → tokenizer.json → gate_TF HLLSet (system-global)
      → DocumentTokenizer → HLLSet → ∩ gate_TF HLLSet
        → LUT (TF accumulation) → materialize
          → BPE encode → token IDs → Decoder

Architecture per STANDARD.md:
    - gate_TF HLLSet = HLLSet.from_tokens(BPE vocabulary) [§2.2 system global]
    - Intersection with gate_TF = indirect bit-level filter [§4.3]
    - The gate is content-addressed, immutable, and idempotent (IICA)
    - The LUT is a persistent singleton with monotonic TF [Appendix D]
    - Materialization uses TF-ranked disambiguation from hllset_py
    - No caal-llm dependency — direct hllset-next bindings
"""

import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import hllset_py

from hllset_cortex.filter import HLLSetFilter, FilterResult


@dataclass
class PipelineResult:
    """Output from one OCR pipeline pass.

    Attributes:
        original: Raw OCR text from vision encoder
        compressed_tokens: Materialized tokens after HLLSet disambiguation
        token_ids: BPE token IDs ready for the language decoder
        reconstructed: BPE-decoded text (for verification)
        filter_result: Raw filter output including HLLSet and stats
    """
    original: str
    compressed_tokens: List[str]
    token_ids: List[int]
    reconstructed: str
    filter_result: FilterResult

    @property
    def ok(self) -> bool:
        return self.filter_result.ok


@dataclass
class OCRPipeline:
    """End-to-end OCR pipeline with HLLSet semantic compression.

    This is the black-box interface for DeepSeek-OCR:
        In:  OCR text (from vision encoder)
        Out: BPE token IDs (for language decoder)

    The tokenizer vocabulary becomes a **gate_TF HLLSet** — a content-
    addressed system-global that filters invalid bit positions at the
    lattice level via intersection. Words the decoder can't encode have
    their bits removed by the gate.

    Usage:
        pipeline = OCRPipeline()
        pipeline.load_tokenizer("path/to/tokenizer.json")
        pipeline.set_gate()                    # build gate_TF HLLSet
        result = pipeline.process(ocr_text)    # returns token IDs for decoder
    """

    _filter: HLLSetFilter = field(default_factory=HLLSetFilter)
    _bpe_vocab: Dict[str, int] = field(default_factory=dict)
    _id_to_token: Dict[int, str] = field(default_factory=dict)

    # ── Tokenizer vocabulary → gate_TF HLLSet ─────────────────────────

    def load_tokenizer(self, tokenizer_json_path: str) -> int:
        """Load tokenizer.json and extract the BPE vocabulary.

        Returns number of tokens loaded. The vocabulary is used for:
        1. Building gate_TF HLLSet (via set_gate)
        2. BPE encoding of materialized words
        """
        with open(tokenizer_json_path) as f:
            data = json.load(f)
        vocab = data.get("model", {}).get("vocab", {})
        self._bpe_vocab = vocab
        self._id_to_token = {v: k for k, v in vocab.items()}
        return len(vocab)

    def set_gate(self) -> "GateInfo":
        """Build gate_TF HLLSet from the loaded BPE vocabulary.

        Extracts English-like words (a-z, 3-20 chars) from the BPE
        vocabulary and ingests them as a content-addressed gate_TF
        HLLSet. This is a system-global — built once, never changes.

        The gate_TF HLLSet is applied via intersection: each document
        HLLSet ∩ gate_TF HLLSet → only valid-vocabulary bit positions
        survive. This is an indirect filter — tokens whose hash bits
        are absent from the gate are filtered out.

        Returns GateInfo with popcount and vocabulary stats.
        """
        valid_words = self._extract_valid_words()

        # Build the gate_TF HLLSet — content-addressed, immutable
        gate_hllset = hllset_py.HLLSet.from_tokens(sorted(valid_words))
        self._filter.gate_hllset = gate_hllset

        return GateInfo(
            vocab_size=len(self._bpe_vocab),
            valid_words=len(valid_words),
            gate_popcount=gate_hllset.popcount(),
            gate_key=gate_hllset.content_key(),
        )

    def _extract_valid_words(self) -> set:
        """Extract English-like words (a-z, 3-20 chars) from BPE vocab.

        Handles LLaMA space-prefix (Ġ = \u0120) and bare tokens.
        """
        valid = set()
        for token in self._bpe_vocab:
            if token.startswith("\u0120"):  # LLaMA space prefix
                word = token[1:]
                if re.match(r"^[a-z]{3,20}$", word):
                    valid.add(word)
            elif re.match(r"^[a-z]{3,20}$", token):
                valid.add(token)
        return valid

    # ── Pipeline ──────────────────────────────────────────────────────

    def process(self, ocr_text: str) -> PipelineResult:
        """Run the full pipeline: HLLSet filter → gate ∩ → BPE encode → decode.

        1. DocumentTokenizer extracts words + bigrams + sentence hashes
        2. HLLSet.from_tokens() → 32,768-bit fingerprint
        3. Gate intersection: HLLSet ∩ gate_TF HLLSet (bit-level filter)
        4. TokenLut accumulates TF for ALL tokens (monotonic CRDT)
        5. Materialize from filtered HLLSet (TF-ranked disambiguation)
        6. BPE encode: words → token IDs (ready for decoder)
        7. BPE decode: token IDs → text (for verification)
        """
        filter_result = self._filter.process(ocr_text)

        if not filter_result.ok:
            return PipelineResult(
                original=ocr_text,
                compressed_tokens=[],
                token_ids=[],
                reconstructed="",
                filter_result=filter_result,
            )

        token_ids = self._encode_words(filter_result.tokens)
        reconstructed = self._decode_ids(token_ids)
        return PipelineResult(
            original=ocr_text,
            compressed_tokens=filter_result.tokens,
            token_ids=token_ids,
            reconstructed=reconstructed,
            filter_result=filter_result,
        )

    def _encode_words(self, words: List[str]) -> List[int]:
        """Words → BPE token IDs via vocabulary lookup."""
        if not self._bpe_vocab:
            return list(range(len(words)))
        ids = []
        for word in words:
            if word in self._bpe_vocab:
                ids.append(self._bpe_vocab[word])
            elif f"\u0120{word}" in self._bpe_vocab:
                ids.append(self._bpe_vocab[f"\u0120{word}"])
            elif word.lower() in self._bpe_vocab:
                ids.append(self._bpe_vocab[word.lower()])
            else:
                ids.append(-1)
        return ids

    def _decode_ids(self, ids: List[int]) -> str:
        """Token IDs → text."""
        if not self._id_to_token:
            return " ".join(str(i) for i in ids)
        tokens = []
        for tid in ids:
            if tid in self._id_to_token:
                token = self._id_to_token[tid]
                if token.startswith("\u0120"):
                    token = " " + token[1:]
                tokens.append(token)
            else:
                tokens.append(f"<{tid}>")
        return "".join(tokens).strip()

    def compare(self, original: str, reconstructed: str) -> Dict:
        """Compare original to reconstructed text."""
        orig_words = set(re.findall(r"[a-z]{3,20}", original.lower()))
        recon_words = set(re.findall(r"[a-z]{3,20}", reconstructed.lower()))
        common = orig_words & recon_words
        return {
            "original_words": len(orig_words),
            "reconstructed_words": len(recon_words),
            "common_words": len(common),
            "jaccard": len(common) / max(len(orig_words | recon_words), 1),
            "word_retention": len(common) / max(len(orig_words), 1),
        }

    def summary(self) -> Dict:
        """Pipeline-level convergence statistics."""
        return self._filter.summary()


@dataclass
class GateInfo:
    """Information about the gate_TF HLLSet."""
    vocab_size: int
    valid_words: int
    gate_popcount: int
    gate_key: str

    def __repr__(self) -> str:
        return (
            f"GateInfo(vocab={self.vocab_size}, valid={self.valid_words}, "
            f"popcount={self.gate_popcount}, key={self.gate_key[:24]}...)"
        )
