# hllset_cortex/temporal.py
"""
Temporal pyramid + DRN (STANDARD.md §4.2–4.3, Part X §10.8).

Two pieces of the larger loop, built on the real hllset_py lattice (union /
intersection / difference) — not a numpy simulation:

    - ``drn`` — the D/R/N decomposition of an arriving observation S(t)
      against the previous state H(t-1).  The Retained component R is the
      R-link, content-addressed as ``r:<sha1>``.
    - ``TemporalPyramid`` — the L0..L6 sliding-window pyramid.  Each layer is
      an aggregate HLLSet; carries are union aggregations at layer boundaries.

``emergent_vocab`` is the Phase 7.1 feeder: the LUT's persistent tokens
(TF >= threshold) are the versioned tokenizer vocabulary.
"""

from dataclasses import dataclass, field
from typing import List

import hllset_py


# ── DRN decomposition (§4.3) ────────────────────────────────────────────

@dataclass
class DRN:
    """The D/R/N split of an arriving observation against the previous state.

        R = S(t) ∩ H(t-1)   retained (already known) — the R-link
        D = H(t-1) \\ S(t)   departed (left the state)
        N = S(t) \\ H(t-1)   new (never seen before)

    R, D, N are themselves HLLSets; the evolution record *is* an HLLSet.
    """
    r: hllset_py.HLLSet
    d: hllset_py.HLLSet
    n: hllset_py.HLLSet

    @property
    def r_link_key(self) -> str:
        """The R-link, content-addressed (``r:<sha1>``)."""
        return self.r.content_key()

    def __repr__(self) -> str:
        return (
            f"DRN(r={self.r.popcount()}, d={self.d.popcount()}, "
            f"n={self.n.popcount()}, r_key={self.r_link_key[:24]}...)"
        )


def drn(s_t: hllset_py.HLLSet, h_prev: hllset_py.HLLSet) -> DRN:
    """Decompose S(t) against H(t-1) via real lattice ops (one-shot)."""
    return DRN(
        r=s_t.intersection(h_prev),
        d=h_prev.difference(s_t),
        n=s_t.difference(h_prev),
    )


# ── Temporal pyramid (§4.2) ─────────────────────────────────────────────

@dataclass
class TemporalPyramid:
    """The L0..L6 sliding-window pyramid over real HLLSets.

    ``durations[i]`` is how many units of layer ``i`` form one unit of layer
    ``i+1`` (the standard second→minute→...→year pyramid is one instance;
    this is the configurable shape of §4.2).  On each ``step``:

        L0 = L0 ∪ S(t)          (absorb the observation)
        carry L_i → L_{i+1}     (union, then reset L_i) when L_i is full

    The top ``∪ L_i`` is monotonic: every carry preserves bits by union, so
    the system state never shrinks (Noether).  All layers are aggregate
    HLLSets — no numpy, no synthetic token sets.
    """

    durations: List[int] = field(default_factory=lambda: [3, 2, 2, 2, 2, 2])
    layers: List[hllset_py.HLLSet] = field(init=False)
    counters: List[int] = field(init=False)
    _steps: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        n_layers = len(self.durations) + 1
        self.layers = [hllset_py.HLLSet() for _ in range(n_layers)]
        self.counters = [0] * len(self.durations)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def top(self) -> hllset_py.HLLSet:
        """H_system(t) = L0 ∪ L1 ∪ ... ∪ L6 — the system state."""
        top = hllset_py.HLLSet()
        for layer in self.layers:
            top = top.union(layer)
        return top

    @property
    def steps(self) -> int:
        return self._steps

    def step(self, s: hllset_py.HLLSet) -> None:
        """Absorb one observation S(t) and carry full layers upward."""
        self._steps += 1
        self.layers[0] = self.layers[0].union(s)
        self.counters[0] += 1

        for i in range(len(self.durations)):
            if self.counters[i] < self.durations[i]:
                break  # lower layer not full — no carry
            self.layers[i + 1] = self.layers[i + 1].union(self.layers[i])
            self.layers[i] = hllset_py.HLLSet()  # reset after union
            self.counters[i] = 0
            self.counters[i + 1] += 1

    def layer_popcounts(self) -> List[int]:
        return [layer.popcount() for layer in self.layers]

    def __repr__(self) -> str:
        return f"TemporalPyramid({self.layer_popcounts()}, top={self.top.popcount()})"


# ── Feeder (§7.1) ───────────────────────────────────────────────────────

def emergent_vocab(lut: hllset_py.TokenLut, min_tf: int = 1) -> List[str]:
    """The emergent vocabulary: LUT tokens that persist under repeated
    measurement (TF >= ``min_tf``), highest-TF first.

    This is the Phase 7.1 feeder — the content-addressed, versioned tokenizer
    vocabulary that *persists* feeds the next generation.  The LUT is never
    reset, so this set only grows (monotonic CRDT).
    """
    return [t for t, f in lut.ranked_tokens() if f >= min_tf]
