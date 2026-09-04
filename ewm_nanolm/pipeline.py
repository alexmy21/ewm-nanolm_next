# ewm_nanolm/pipeline.py
"""NanolmPipeline — the EWM lattice around an LLM token-ID stream.

The anchor demo: token IDs are already discrete, so the boundary is exactly
ds-hllset-cortex minus the OCR encoder.

    token IDs -> "tid671 tid18308 ..." -> HLLSet lattice (gate ∩, LUT,
    De Bruijn order restoration) -> restored "tid..." stream -> LLM

The gate_TF HLLSet is built from a *subset* of the model vocabulary — the
emergent vocabulary the system has actually measured.  That is the world
view ("what the model can express in this conversation"); expanding the gate
re-admits latent tids at their earned TF.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

import hllset_py
from hllset_cortex import HLLSetFilter
from hllset_cortex.domain import encoding_tokenizer, hllset_from_ids, tid

PROJECT = Path(__file__).resolve().parents[1]
CAPTURE = PROJECT / "captures" / "llm_capture.npz"


def load_capture(path: Optional[Path] = None) -> dict:
    """Load the LLM capture; fall back to synthetic token-id streams."""
    path = Path(path) if path is not None else CAPTURE
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return {
            "prompts": [str(p) for p in data["prompts"]],
            "prompt_ids": [np.asarray(a, dtype=np.int64) for a in data["prompt_ids"]],
            "generated_ids": [np.asarray(a, dtype=np.int64) for a in data["generated_ids"]],
            "generated_texts": [str(t) for t in data["generated_texts"]],
            "synthetic": False,
        }
    return synthetic_capture()


def synthetic_capture(seed: int = 0) -> dict:
    """Three synthetic token-id streams, used until the real capture exists."""
    rng = np.random.RandomState(seed)
    prompts = [
        "What is the capital of France?",
        "Explain gravity in one sentence.",
        "Write a haiku about rain.",
    ]
    prompt_ids = [rng.randint(0, 4000, size=18) for _ in prompts]
    generated_ids = [rng.randint(0, 4000, size=24) for _ in prompts]
    generated_texts = [f"synthetic reply {i}" for i in range(3)]
    return {
        "prompts": prompts,
        "prompt_ids": prompt_ids,
        "generated_ids": generated_ids,
        "generated_texts": generated_texts,
        "synthetic": True,
    }


def ids_to_text(ids: Iterable[int]) -> str:
    return " ".join(tid(int(i)) for i in ids)


def _eulerian_path(edges: List[str]) -> List[str]:
    """Hierholzer walk over the bigram edge multiset (De Bruijn order)."""
    from collections import defaultdict

    adj = defaultdict(list)
    for e in edges:
        a, b = e.split("\x00", 1)
        adj[a].append(b)

    stack = ["<S>"]
    path: List[str] = []
    while stack:
        u = stack[-1]
        if adj.get(u):
            stack.append(adj[u].pop())
        else:
            path.append(stack.pop())
    path.reverse()
    if path and path[0] == "<S>":
        path = path[1:]
    if path and path[-1] == "</S>":
        path = path[:-1]
    return path


@dataclass
class RoundTripStats:
    input_ids: int = 0
    unique_input_ids: int = 0
    gated_ids: int = 0
    restored_ids: int = 0
    set_retention: float = 0.0
    ordered_retention: float = 0.0
    exact_set_match: bool = False
    exact_order_match: bool = False
    hllset_popcount: int = 0
    gate_popcount: int = 0
    lut_size: int = 0


@dataclass
class RoundTripResult:
    tids_in: List[str] = field(default_factory=list)
    tids_out: List[str] = field(default_factory=list)
    stats: RoundTripStats = None


@dataclass
class NanolmPipeline:
    """LLM token-ID stream + HLLSet lattice, wired as one pipeline."""

    seed: int = 0

    filter: HLLSetFilter = field(init=False, repr=False)
    gate: object = field(init=False, repr=False)
    _db_tokenizer: object = field(init=False, repr=False)

    def __post_init__(self):
        self.filter = HLLSetFilter()
        self.filter.tokenizer = encoding_tokenizer()
        self.gate = None
        self.filter.gate_hllset = None

        allowed = list((string.ascii_lowercase + string.digits + "_-").encode())
        self._db_tokenizer = (
            hllset_py.Tokenizer()
            .pattern(allowed)
            .lowercase()
            .pad(b"<S>", b"</S>")
            .ngrams(2, 2)
        )

    def set_gate(self, ids: Iterable[int]) -> None:
        """Build the gate_TF HLLSet from a vocabulary subset (world view)."""
        unique = sorted({int(i) for i in ids})
        self.gate = hllset_from_ids(unique)
        self.filter.gate_hllset = self.gate

    def clear_gate(self) -> None:
        self.gate = None
        self.filter.gate_hllset = None

    def cortex_pass(self, text: str) -> dict:
        """Gated set pass + ungated De Bruijn order pass."""
        fres = self.filter.process_text(text)
        gated = [t for t in fres.token_strings if "\x00" not in t]

        ordered: List[str] = []
        bigrams = self._db_tokenizer.tokenize(text.encode())
        if bigrams:
            h_db = hllset_py.HLLSet.from_token_bytes(bigrams)
            self.filter.lut.record_all_bytes(bigrams)
            bigram_strs = [b.decode("utf-8", errors="replace") for b in bigrams]
            ordered = _eulerian_path(bigram_strs)
            if len(ordered) < len(bigrams) - 1:
                out = hllset_py.materialize_debruijn(h_db, self.filter.lut, "<S>", "</S>")
                payload = [
                    t if isinstance(t, str) else t.decode("utf-8", errors="replace")
                    for t in out
                    if t not in ("<S>", "</S>")
                ]
                seq: List[str] = []
                for t in payload:
                    if "\x00" in t:
                        a, b = t.split("\x00", 1)
                        for x in (a, b):
                            if not seq or seq[-1] != x:
                                seq.append(x)
                    elif not seq or seq[-1] != t:
                        seq.append(t)
                ordered = seq

        return {
            "gated_tokens": gated,
            "ordered_tokens": ordered,
            "hllset": fres.hllset,
            "filtered_hllset": fres.filtered_hllset,
            "lut_size": fres.lut_size,
        }

    def roundtrip(self, ids: Iterable[int]) -> RoundTripResult:
        ids = [int(i) for i in ids]
        text = ids_to_text(ids)

        cres = self.cortex_pass(text)
        restored = cres["ordered_tokens"] or cres["gated_tokens"]
        restored_ints = [int(t[3:]) for t in restored if t.startswith("tid")]

        unique = sorted(set(ids))
        gated_set = sorted({int(t[3:]) for t in cres["gated_tokens"] if t.startswith("tid")})

        stats = RoundTripStats(
            input_ids=len(ids),
            unique_input_ids=len(unique),
            gated_ids=len(gated_set),
            restored_ids=len(restored_ints),
            set_retention=len(gated_set) / max(len(unique), 1),
            ordered_retention=len(restored_ints) / max(len(ids), 1),
            exact_set_match=gated_set == unique,
            exact_order_match=restored_ints == ids,
            hllset_popcount=cres["hllset"].popcount() if cres["hllset"] else 0,
            gate_popcount=cres["filtered_hllset"].popcount() if cres["filtered_hllset"] else 0,
            lut_size=cres["lut_size"],
        )

        return RoundTripResult(
            tids_in=text.split(),
            tids_out=restored,
            stats=stats,
        )
