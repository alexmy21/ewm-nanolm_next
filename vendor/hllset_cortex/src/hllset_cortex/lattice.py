# hllset_cortex/lattice.py
"""
The EWM lattice — the measured memory a search runs against.

Two spaces, one boundary:

- **token space** — encoding IDs (``tid{n}``): what the LLM exchanges. The
  TokenLUT (reverse index + TF) and materialization live here.
- **HLLSet space** — 32,768-bit sketches: where every structural operation
  (union, intersection, BSS, R-link, DRN, the temporal pyramid) lives.

The only morphisms between the two spaces are **ingest** (tokens → HLLSet,
hash + bootstrap) and **materialize** (HLLSet → tokens).

Every observation (a scanned page, a search query) is submitted the same way
(STANDARD.md §4.1, §4.3):

    H(t) = H(S(t), H(t-1), D(t-1), R(t-1), N(t))

- its encoding IDs are ingested: hashed into an HLLSet, and TF accumulates in
  the TokenLUT (never gated — everything measured is stored),
- the HLLSet is decomposed against the current state (DRN),
- the HLLSet is committed to the temporal pyramid.

Page atoms are registered as the searchable `o:` originals; the whole document
is the `v:` union view. A search query is itself an observation: it is
converted to an HLLSet and **submitted to the lattice** before the pages are
ranked, so the query is measured — not just compared transiently.

Beyond the shallow grounding (response vs current context), ``precedents``
dives into the history — every submitted page and query — and returns the
prior observations that resemble a query (R-link feedback gate, §4.4), as
reference for decision-making.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import hllset_py

from hllset_cortex.domain import hllset_from_ids, tid
from hllset_cortex.search import Document, PageAtom, SearchConfig, SearchHit, search
from hllset_cortex.temporal import DRN, TemporalPyramid, drn


@dataclass
class HistoryEntry:
    """One recorded observation in the lattice's history (a page or a query)."""
    label: str
    hllset: hllset_py.HLLSet
    key: str


@dataclass
class Precedent:
    """A prior observation retrieved from the history.

    ``tau`` is the BSS coverage ``|entry ∩ query| / |query|`` — the ranking
    key (the measurement, same as ``search``). ``weight`` is the R-link
    popcount ``|query ∩ entry|`` — the FPGA-native form, reported for compat.
    """
    label: str
    key: str
    weight: int
    tau: float

    def __repr__(self) -> str:
        return f"Precedent({self.label}, weight={self.weight}, tau={self.tau:.3f})"


@dataclass
class Lattice:
    """The measured memory.

    ``doc`` / ``context`` / ``pyramid`` are HLLSet-space (structural); ``lut``
    is token-space (the reverse index materialization reads). ``history`` keeps
    every submitted observation (pages and queries), so ``precedents`` can dive
    back into the temporal-pyramid history for decision-making reference.
    """

    doc: Document = field(default_factory=Document)
    lut: hllset_py.TokenLut = field(default_factory=hllset_py.TokenLut)
    pyramid: TemporalPyramid = field(default_factory=TemporalPyramid)
    context: hllset_py.HLLSet = field(default_factory=hllset_py.HLLSet)
    history: List[HistoryEntry] = field(default_factory=list)
    _n_queries: int = field(default=0, init=False, repr=False)

    @classmethod
    def from_pages(cls, pages) -> "Lattice":
        """Build a lattice from ``[(page_id, [encoding_id, ...]), ...]``."""
        lattice = cls()
        for page_id, ids in pages:
            lattice.add_page(page_id, ids)
        return lattice

    def add_page(self, page_id: str, ids) -> "Lattice":
        """Register a page as an `o:` atom and submit it to the lattice."""
        hllset = hllset_from_ids(ids)
        self.doc.pages.append(PageAtom(page_id, hllset))
        self.ingest(hllset, ids, label=page_id)
        return self

    def ingest(self, observation: hllset_py.HLLSet, ids, label: Optional[str] = None) -> DRN:
        """Submit one observation.

        Token space: the ids accumulate TF in the LUT. HLLSet space: the
        observation is decomposed (DRN) against the context and committed to
        the temporal pyramid. If ``label`` is given, the observation is also
        recorded in the history (for ``precedents``).
        """
        self.lut.record_all([tid(i) for i in ids])
        d = drn(observation, self.context)
        self.context = self.context.union(observation)
        self.pyramid.step(observation)
        if label is not None:
            self.history.append(HistoryEntry(label, observation, observation.content_key()))
        return d

    def submit_query(self, query_ids, label: Optional[str] = None) -> Tuple[hllset_py.HLLSet, DRN]:
        """Convert a query to an HLLSet and submit it to the lattice.

        Returns the query HLLSet and its DRN against the measured state
        (``d.n`` non-empty means the query introduced never-measured ids).
        The query is recorded in the history (auto-labelled if ``label`` is
        None), so it can later be found as a precedent.
        """
        query = hllset_from_ids(query_ids)
        if label is None:
            label = f"query_{self._n_queries}"
        self._n_queries += 1
        return query, self.ingest(query, query_ids, label=label)

    def precedents(
        self,
        query: hllset_py.HLLSet,
        top_k: Optional[int] = None,
        min_weight: int = 1,
    ) -> List[Precedent]:
        """Retrieve prior observations that resemble ``query``, ranked by BSS τ
        (coverage) — the same measurement ``search`` uses, aimed at the history
        instead of the document (§4.4 feedback gate).

        This is the deep step for decision-making: instead of comparing only
        against the current context, dive into the history (every submitted
        page and query) and surface the most similar precedents as reference.
        The R-link popcount is reported as ``weight`` (the FPGA-native form).
        """
        hits = []
        for entry in self.history:
            weight = query.intersection(entry.hllset).popcount()
            if weight < min_weight:
                continue
            tau = entry.hllset.bss_inclusion(query)
            hits.append(Precedent(entry.label, entry.key, weight, tau))
        hits.sort(key=lambda h: h.tau, reverse=True)
        if top_k is not None:
            hits = hits[:top_k]
        return hits

    def search(
        self, query_ids, config: Optional[SearchConfig] = None
    ) -> List[SearchHit]:
        """Submit the query to the lattice, then rank the pages against it."""
        query, _ = self.submit_query(query_ids)
        return search(query, self.doc, config)
