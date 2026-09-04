# hllset_cortex/grounding.py
"""
Grounding — the EWM<->LLM validation step (Part X §10.7, §3.6).

The response is validated against the measured state in two one-sided
diagnoses of hallucination, per STANDARD.md Part X:

    1. Token hallucination (diagnosed by the LUT).  An encoding that never
       arrived through ingestion is unknown — the LUT flags it.  The LUT never
       hallucinates; the HLLSet the LLM produces does.
    2. Structural hallucination (diagnosed by BSS ρ).  Even when every token
       is known, the response may depart structurally from the context; the
       BSS ρ gate (novelty) flags that.

The two diagnoses live in **different spaces**: token hallucination is
token-space (exact-LUT membership over encodings), structural hallucination is
HLLSet-space (BSS ρ over sketches). Tokens and HLLSets never cross directly —
ingest (tokens → HLLSet, via hash + bootstrap) and materialize (HLLSet →
tokens) are the only morphisms between them.

This module ports the EWM-nanoLM findings (``nanolm-context/src/gate.rs``,
``grounding.rs``) onto the ds-ocr substrate, using only the existing
``hllset_py`` lattice operations.

The load-bearing guarantee (STANDARD.md Appendix A):

    A single-HLLSet ``gate ∩`` saturates and leaks ~97% of out-of-vocabulary
    ids by collision (and 2-of-3 consensus is *worse*, ~99.99%).  The correct
    gate is the **exact LUT forward map** — membership lives in the reverse
    index (``TokenLut``), not the sketch.  It has 0 leak and 0 false negatives.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import hllset_py

from hllset_cortex.search import bss_rho


# ── Exact-LUT membership (the corrected gate) ───────────────────────────

def exact_known(lut: hllset_py.TokenLut, token: str) -> bool:
    """Was ``token`` ever measured into the LUT?  Exact forward-map lookup.

    This is the Appendix A gate: membership in the reverse index, not the
    sketch.  Zero OOV leak, zero false negatives.  ``tf`` returns 0 for an
    unknown token, so this is a single monotonic-CRDT read.
    """
    return lut.tf(token) > 0


def hallucinated_positions(
    response_hllset: hllset_py.HLLSet, lut: hllset_py.TokenLut
) -> List[Tuple[int, int]]:
    """The active ``<reg, zeros>`` positions of the response not in the LUT.

    These are the positions the materializer would have to *add* to the LUT —
    content the response emitted that was never measured.  One-sided: a
    measured position is never flagged.
    """
    return [
        (reg, tz)
        for (reg, tz) in response_hllset.active_positions()
        if not lut.lookup_position(reg, tz)
    ]


def has_hallucination(response_hllset: hllset_py.HLLSet, lut: hllset_py.TokenLut) -> bool:
    """Does the response contain any never-measured position?"""
    return bool(hallucinated_positions(response_hllset, lut))


# ── Grounding report (τ/ρ + R-link) ─────────────────────────────────────

@dataclass
class GroundingConfig:
    """Grounding thresholds.

    - ``tau_min`` / ``rho_max`` gate the **token** diagnosis (LUT membership).
    - ``structural_rho_max`` gates the **structural** diagnosis (BSS ρ novelty).
    """
    tau_min: float = 0.8
    rho_max: float = 0.2
    structural_rho_max: float = 0.2


@dataclass
class GroundingReport:
    """Grounding verdict for a response against the measured context.

    Carries **two diagnoses of hallucination in two different spaces**
    (STANDARD.md §10.7):

    1. Token hallucination — **token space**, diagnosed by the LUT: ``flagged``
       is the encodings that never arrived through ingestion (unknown to the
       LUT); ``tau``/``rho`` are the known/unknown fractions. The LUT never
       hallucinates — it only reports; the HLLSet the LLM produces does.

    2. Structural hallucination — **HLLSet space**, diagnosed by BSS ρ:
       ``structural_rho = |response \\ context| / |context|`` is how far the
       response departs from the measured state even when every token is known.
       ``r_link_popcount`` is the same intersection as the FPGA-native integer.

    BSS ρ is only meaningful when ``response`` and ``context`` are at the same
    cardinality scale (see ``search.bss_rho``) — ``|·|`` is the HLLSet
    cardinality, a function of the '1'-bit distribution in the fixed vector.
    """
    tau: float = 1.0
    rho: float = 0.0
    structural_rho: float = 0.0
    grounded: bool = True
    flagged: List[str] = field(default_factory=list)
    r_link_popcount: int = 0
    r_link_key: str = ""

    def __repr__(self) -> str:
        return (
            f"GroundingReport(tau={self.tau:.3f}, rho={self.rho:.3f}, "
            f"srho={self.structural_rho:.3f}, grounded={self.grounded}, "
            f"flagged={len(self.flagged)}, r_link={self.r_link_popcount})"
        )


def grounding_report(
    context_hllset: hllset_py.HLLSet,
    response_ids: List[str],
    lut: hllset_py.TokenLut,
    config: Optional[GroundingConfig] = None,
) -> GroundingReport:
    """Validate ``response_ids`` against the measured state (``context_hllset`` + ``lut``).

    Two one-sided diagnoses, per §10.7:
        1. token hallucination — the LUT flags encodings that never arrived
           through ingestion (``flagged``),
        2. structural hallucination — BSS ρ measures how far the response
           departs from the context (``structural_rho``).

    Read-only: it never mutates the LUT or the lattice.
    """
    cfg = config or GroundingConfig()

    # 1. Token hallucination — the LUT flags encodings never ingested.
    n = len(response_ids)
    in_lut = sum(1 for t in response_ids if exact_known(lut, t))
    tau = in_lut / n if n else 1.0
    rho = (n - in_lut) / n if n else 0.0
    flagged = [t for t in response_ids if not exact_known(lut, t)]

    # 2. Structural hallucination — BSS ρ (novelty) + R-link (FPGA-native).
    response_hllset = hllset_py.HLLSet.from_tokens(response_ids)
    r_link = context_hllset.intersection(response_hllset)
    structural_rho = bss_rho(response_hllset, context_hllset)

    grounded = (
        tau >= cfg.tau_min
        and rho <= cfg.rho_max
        and structural_rho <= cfg.structural_rho_max
    )

    return GroundingReport(
        tau=tau,
        rho=rho,
        structural_rho=structural_rho,
        grounded=grounded,
        flagged=flagged,
        r_link_popcount=r_link.popcount(),
        r_link_key=r_link.content_key(),
    )
