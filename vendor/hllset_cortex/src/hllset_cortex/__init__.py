# hllset_cortex
"""
HLLSet Cortex — semantic encoding restoration for DeepSeek-OCR.

A reference implementation for HLLSet Algebra applications per the
hllset-next STANDARD.md. Receives encoding IDs from ds-ocr's vision
encoder, processes them through the HLLSet Algebra pipeline, and
returns restored encoding IDs for the decoder.

Architecture:
    ds-ocr encoding IDs → hllset_py.Tokenizer (standard pipeline)
      → MurmurHash3 → HLLSet (32,768-bit bitmap)
        → ∩ gate_TF HLLSet (decoder vocabulary filter)
          → TokenLut (monotonic TF accumulation)
            → materialization (n-gram disambiguation, TF tie-break) → restored IDs → Decoder

Key properties (IICA):
    - Idempotent: same IDs → same HLLSet, every time
    - Immutable: HLLSets never change once created
    - Content-Addressed: HLLSet key = SHA1 of serialized bytes

Scenario: PDF book scanning
    Each page → HLLSet
    Chapter = ∪ page HLLSets
    Book = ∪ chapter HLLSets
    Commit to temporal pyramid → holographic memory

Dependencies:
    hllset-py (Rust PyO3 bindings: hllset-core + hllset-dsl tokenizer)
    Python 3.10+

Reference docs:
    - docs/STANDARD.md (governing development standard)
    - docs/IICA_PRINCIPLES.md (foundational IICA gate definition)
    - DESIGN.md (this module's design)
"""

from hllset_cortex.domain import (
    default_tokenizer,
    encoding_tokenizer,
    debruijn_tokenizer,
    tid,
    hllset_from_ids,
)
from hllset_cortex.filter import HLLSetFilter, FilterResult, FilterStats
from hllset_cortex.pipeline import OCRPipeline, PipelineResult, GateInfo
from hllset_cortex.grounding import (
    exact_known,
    hallucinated_positions,
    has_hallucination,
    GroundingConfig,
    GroundingReport,
    grounding_report,
)
from hllset_cortex.temporal import (
    DRN,
    TemporalPyramid,
    drn,
    emergent_vocab,
)
from hllset_cortex.search import (
    Document,
    PageAtom,
    SearchConfig,
    SearchHit,
    bss_rho,
    search,
)
from hllset_cortex.lattice import HistoryEntry, Lattice, Precedent
from hllset_cortex.conversation import (
    START,
    END,
    BitAnchors,
    ContextTop,
    ConversationContext,
    Exchange,
    RestoredConversation,
    RoundTripReport,
    SharedBitOrigin,
    SparseAdjacencyMatrix,
    TokenIndex,
    UnfoldedCell,
    bit_position,
    build_prompt,
    build_prompt_from_restored,
    restore_context,
    roundtrip_context,
    roundtrip_exchange,
    roundtrip_exchange_with_extra_tokens,
)

__all__ = [
    # Tokenizer config + token definition (§10.3)
    "default_tokenizer",
    "encoding_tokenizer",
    "debruijn_tokenizer",
    "tid",
    "hllset_from_ids",
    # Filter
    "HLLSetFilter",
    "FilterResult",
    "FilterStats",
    # Pipeline
    "OCRPipeline",
    "PipelineResult",
    "GateInfo",
    # Grounding (Part X §10.7)
    "exact_known",
    "hallucinated_positions",
    "has_hallucination",
    "GroundingConfig",
    "GroundingReport",
    "grounding_report",
    # Temporal + DRN (§4.2–4.3, §10.8)
    "DRN",
    "TemporalPyramid",
    "drn",
    "emergent_vocab",
    # Page-granular search (§4.4)
    "Document",
    "PageAtom",
    "SearchConfig",
    "SearchHit",
    "bss_rho",
    "search",
    # Lattice (submit observations + search + precedents)
    "HistoryEntry",
    "Lattice",
    "Precedent",
    # Conversation context (phase 8)
    "START",
    "END",
    "BitAnchors",
    "ContextTop",
    "ConversationContext",
    "Exchange",
    "RestoredConversation",
    "RoundTripReport",
    "SharedBitOrigin",
    "SparseAdjacencyMatrix",
    "TokenIndex",
    "UnfoldedCell",
    "bit_position",
    "build_prompt",
    "build_prompt_from_restored",
    "restore_context",
    "roundtrip_context",
    "roundtrip_exchange",
    "roundtrip_exchange_with_extra_tokens",
]
