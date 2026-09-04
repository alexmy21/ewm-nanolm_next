# hllset_cortex/conversation.py
"""
Conversation context over HLLSet — Python prototype of ``hllset-context``.

Mirrors the Rust crate ``hllset-next/crates/hllset-context`` on top of the
``hllset_py`` binding.  A user<->LLM conversation is encoded as a sequence
of exchanges, each of which is three HLLSet layers:

    exchange.hll = 1-HLLSet ∪ 2-HLLSet ∪ 3-HLLSet

- 1-HLLSet — vocabulary bits (unpadded ``tid{n}`` tokens).
- 2-HLLSet — padded follow-edges (``_START_\\0tid0``, ``tid0\\0tid1``, ...).
- 3-HLLSet — padded De Bruijn edges (order reconstruction).

The conversation context is the top union of those layers plus a sparse
adjacency matrix over the **1-HLLSet union** (the vocabulary space); the
2-/3-HLLSets are structural equivalents that point into the 1-HLLSet.

Two problems solved (same as the Rust crate):
  1. Context building   — top unions + adjacency matrix + collision groups.
  2. LLM utilization     — 3-gram De Bruijn restore + materialize-to-prompt.

Round-trip composition law (empirically proven in the Rust crate):

    hllA -> materialize -> {tokens} -> ingest -> hllB
      1-hllA = 1-hllB                     (exact)
      2-hllA = 2-hllB, 3-hllA = 3-hllB    (high probability)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import hllset_py

from hllset_cortex.domain import tid

# ── N-gram conventions (bit-identical to hllset-bridge / hllset-context) ──

START = "_START_"
END = "_END_"
SEP = "\0"


def bit_position(token: str) -> int:
    """Flattened bit index ``reg * 32 + tz`` of a token under MurmurHash3."""
    reg, tz = hllset_py.token_to_position_py(token)
    return reg * 32 + tz


def is_boundary(token: str) -> bool:
    return token == START or token == END


def pad_tokens(tokens: List[str]) -> List[str]:
    return [START] + list(tokens) + [END]


def join_ngram(tokens: List[str]) -> str:
    if len(tokens) == 1:
        return tokens[0]
    return SEP.join(tokens)


def split_ngram(token: str) -> List[str]:
    return token.split(SEP)


def generate_ngrams(tokens: List[str], n: int, pad: bool) -> List[str]:
    """n-grams of size ``n``; padded with START/END when ``pad`` is true."""
    if n == 0 or not tokens:
        return []
    seq = pad_tokens(tokens) if pad else list(tokens)
    if len(seq) < n:
        return []
    return [join_ngram(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def _toggle_role(role: str) -> str:
    return "assistant" if role == "user" else "user"


# ── Exchange encoding ──────────────────────────────────────────────────────

@dataclass
class Exchange:
    """One user/assistant turn encoded as three HLLSet layers."""

    role: str
    tokens: List[str] = field(default_factory=list)
    text: str = ""

    # Layer HLLSets (built in __post_init__).
    hll_1: hllset_py.HLLSet = field(init=False, repr=False)
    hll_2: hllset_py.HLLSet = field(init=False, repr=False)
    hll_3: hllset_py.HLLSet = field(init=False, repr=False)
    hll: hllset_py.HLLSet = field(init=False, repr=False)

    def __post_init__(self) -> None:
        g1 = generate_ngrams(self.tokens, 1, False)
        g2 = generate_ngrams(self.tokens, 2, True)
        g3 = generate_ngrams(self.tokens, 3, True)
        self.hll_1 = hllset_py.HLLSet.from_tokens(g1)
        self.hll_2 = hllset_py.HLLSet.from_tokens(g2)
        self.hll_3 = hllset_py.HLLSet.from_tokens(g3)
        self.hll = self.hll_1.union(self.hll_2).union(self.hll_3)

    @classmethod
    def from_ids(cls, role: str, ids) -> "Exchange":
        """Build an exchange from encoding IDs (``tid{i}`` tokens)."""
        return cls(role=role, tokens=[tid(i) for i in ids])

    @classmethod
    def from_tokens(cls, role: str, tokens) -> "Exchange":
        """Build an exchange from already-formatted token strings."""
        return cls(role=role, tokens=list(tokens))

    def ngrams(self, n: int) -> List[str]:
        """Layer ``n`` n-grams (1 = unpadded, 2/3 = boundary-padded)."""
        return generate_ngrams(self.tokens, n, n > 1)

    def len(self) -> int:
        return len(self.tokens)

    def is_empty(self) -> bool:
        return not self.tokens

    def bit_anchors(self, bit: int) -> Optional["BitAnchors"]:
        """Structural 2-gram context of a vocabulary bit in this exchange."""
        tokens = [t for t in self.ngrams(1) if bit_position(t) == bit]
        if not tokens:
            return None
        left: List[str] = []
        right: List[str] = []
        for tg in self.ngrams(2):
            parts = split_ngram(tg)
            if len(parts) != 2:
                continue
            if bit_position(parts[1]) == bit and not is_boundary(parts[0]):
                left.append(parts[0])
            if bit_position(parts[0]) == bit and not is_boundary(parts[1]):
                right.append(parts[1])
        return BitAnchors(bit=bit, tokens=tokens, left=left, right=right)

    def roundtrip(self) -> "RoundTripReport":
        return roundtrip_exchange(self)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Exchange({self.role}, {len(self.tokens)} tokens, "
            f"hll={self.hll.popcount()} bits)"
        )


@dataclass
class BitAnchors:
    """Structural anchors of one vocabulary bit inside one exchange."""

    bit: int
    tokens: List[str]
    left: List[str]
    right: List[str]


# ── Sparse adjacency matrix over the 1-HLLSet union ────────────────────────

@dataclass
class SparseAdjacencyMatrix:
    """Follow-frequency matrix.

    ``counts[(row_bit, col_bit)]`` accumulates how many times a padded
    2-gram ``p\\0s`` was observed.  ``token_counts`` keeps the token-level
    view for prompt materialization and collision unfolding.
    """

    counts: Dict[Tuple[int, int], int] = field(default_factory=dict)
    token_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)

    def increment_edge(self, prefix: str, suffix: str) -> None:
        row, col = bit_position(prefix), bit_position(suffix)
        self.counts[(row, col)] = self.counts.get((row, col), 0) + 1
        self.token_counts[(prefix, suffix)] = (
            self.token_counts.get((prefix, suffix), 0) + 1
        )

    def increment_2grams(self, two_grams: List[str]) -> None:
        for tg in two_grams:
            parts = split_ngram(tg)
            if len(parts) == 2:
                self.increment_edge(parts[0], parts[1])

    def get(self, row: int, col: int) -> int:
        return self.counts.get((row, col), 0)

    def get_token(self, prefix: str, suffix: str) -> int:
        return self.token_counts.get((prefix, suffix), 0)

    def successors(self, row: int) -> List[Tuple[int, int]]:
        items = [((col, n)) for (r, col), n in self.counts.items() if r == row]
        return sorted(items, key=lambda cn: (-cn[1], cn[0]))

    def continuations(self, row: int, k: int) -> List[Tuple[int, int]]:
        return self.successors(row)[:k]

    def greedy_walk(self, start_bit: int, max_len: int) -> List[int]:
        """Cell-value path-switching rule: always take the highest-count
        unused outgoing edge; when exhausted, switch to the next-best cell."""
        if max_len == 0:
            return []
        path = [start_bit]
        used = set()
        current = start_bit
        while len(path) < max_len:
            nxt = None
            for col, _count in self.successors(current):
                if (current, col) not in used:
                    used.add((current, col))
                    nxt = col
                    break
            if nxt is None:
                break
            path.append(nxt)
            current = nxt
        return path

    def nonzero_cells(self) -> int:
        return len(self.counts)

    def shape(self) -> Tuple[int, int, int]:
        rows = {r for r, _ in self.counts}
        cols = {c for _, c in self.counts}
        return (len(rows), len(cols), len(self.counts))

    def total_follows(self) -> int:
        return sum(self.counts.values())

    def top_links(self, k: int) -> List[Tuple[str, str, int]]:
        items = sorted(
            self.token_counts.items(), key=lambda kv: (-kv[1], kv[0])
        )
        return [(p, s, n) for (p, s), n in items[:k]]

    def index_bits(self) -> set:
        bits = set()
        for row, col in self.counts:
            bits.add(row)
            bits.add(col)
        return bits

    def index_is_subset_of(self, vocab_bits) -> bool:
        return all(
            row in vocab_bits and col in vocab_bits for row, col in self.counts
        )

    def unfolded_cells(self, index: "TokenIndex") -> List["UnfoldedCell"]:
        out: List[UnfoldedCell] = []
        for (row, col), count in sorted(self.counts.items()):
            rows = index.get(row) or [f"bit:{row}"]
            cols = index.get(col) or [f"bit:{col}"]
            for rt in rows:
                for ct in cols:
                    out.append(UnfoldedCell(row, col, rt, ct, count))
        return out

    def is_empty(self) -> bool:
        return not self.counts


@dataclass
class TokenIndex:
    """Reverse index: bit position -> candidate tokens (collision group)."""

    groups: Dict[int, List[str]] = field(default_factory=dict)

    @classmethod
    def from_vocabulary(cls, tokens) -> "TokenIndex":
        idx = cls()
        for token in tokens:
            idx.insert(token)
        return idx

    def insert(self, token: str) -> None:
        bit = bit_position(token)
        group = self.groups.setdefault(bit, [])
        if token not in group:
            group.append(token)

    def get(self, bit: int) -> List[str]:
        return self.groups.get(bit, [])

    def len(self) -> int:
        return len(self.groups)

    def is_empty(self) -> bool:
        return not self.groups

    def group_sizes(self) -> List[Tuple[int, int]]:
        return sorted((bit, len(g)) for bit, g in self.groups.items())

    def max_group_size(self) -> int:
        return max((len(g) for g in self.groups.values()), default=0)


@dataclass
class UnfoldedCell:
    """One token-resolved cell: a bit cell duplicated per candidate token."""

    row_bit: int
    col_bit: int
    row_token: str
    col_token: str
    count: int


# ── Conversation context ───────────────────────────────────────────────────

@dataclass
class ContextTop:
    """Per-layer unions of a conversation context."""

    hll_1: hllset_py.HLLSet = field(default_factory=hllset_py.HLLSet)
    hll_2: hllset_py.HLLSet = field(default_factory=hllset_py.HLLSet)
    hll_3: hllset_py.HLLSet = field(default_factory=hllset_py.HLLSet)
    hll: hllset_py.HLLSet = field(default_factory=hllset_py.HLLSet)

    def popcount(self) -> int:
        return self.hll.popcount()

    def cardinality(self) -> float:
        return self.hll.cardinality()

    def is_empty(self) -> bool:
        return self.hll.is_empty()

    def tz_histogram(self) -> List[int]:
        hist = [0] * 32
        for _reg, tz in self.hll.active_positions():
            hist[tz] += 1
        return hist

    def active_tz_columns(self) -> int:
        return sum(1 for c in self.tz_histogram() if c)


@dataclass
class SharedBitOrigin:
    """One shared 1-HLLSet bit between two exchanges and its origins."""

    bit: int
    from_a: List[str]
    from_b: List[str]
    same_token: bool


def _tokens_at_bit(exchange: Exchange, bit: int) -> List[str]:
    tokens = sorted(
        {t for t in exchange.ngrams(1) if bit_position(t) == bit}
    )
    return tokens


@dataclass
class ConversationContext:
    """A conversation: exchange list + top unions + adjacency matrix."""

    exchanges: List[Exchange] = field(default_factory=list)
    top: ContextTop = field(default_factory=ContextTop)
    matrix: SparseAdjacencyMatrix = field(default_factory=SparseAdjacencyMatrix)

    @classmethod
    def build(cls, exchanges: List[Exchange]) -> "ConversationContext":
        ctx = cls()
        for exchange in exchanges:
            ctx.add_exchange(exchange)
        return ctx

    def add_exchange(self, exchange: Exchange) -> None:
        self.matrix.increment_2grams(exchange.ngrams(2))
        self.top.hll_1 = self.top.hll_1.union(exchange.hll_1)
        self.top.hll_2 = self.top.hll_2.union(exchange.hll_2)
        self.top.hll_3 = self.top.hll_3.union(exchange.hll_3)
        self.top.hll = self.top.hll.union(exchange.hll)
        self.exchanges.append(exchange)

    def len(self) -> int:
        return len(self.exchanges)

    def is_empty(self) -> bool:
        return not self.exchanges

    def bss_inclusion(self, exchange_index: int) -> Optional[float]:
        if exchange_index >= len(self.exchanges):
            return None
        return self.top.hll.bss_inclusion(self.exchanges[exchange_index].hll)

    def exchange_overlap(self, i: int, j: int) -> Optional[float]:
        if i >= len(self.exchanges) or j >= len(self.exchanges):
            return None
        return self.exchanges[i].hll.jaccard(self.exchanges[j].hll)

    # ── Vocabulary / matrix invariants ────────────────────────────────────

    def vocabulary_index(self) -> TokenIndex:
        idx = TokenIndex()
        idx.insert(START)
        idx.insert(END)
        for ex in self.exchanges:
            for token in ex.ngrams(1):
                idx.insert(token)
        return idx

    def matrix_index_is_vocabulary(self) -> bool:
        vocab = {reg * 32 + tz for reg, tz in self.top.hll_1.active_positions()}
        vocab.add(bit_position(START))
        vocab.add(bit_position(END))
        return self.matrix.index_is_subset_of(vocab)

    def exchange_path_cells(self, exchange_index: int) -> List[Tuple[int, int, int]]:
        if exchange_index >= len(self.exchanges):
            return []
        out = []
        for tg in self.exchanges[exchange_index].ngrams(2):
            parts = split_ngram(tg)
            if len(parts) == 2:
                row, col = bit_position(parts[0]), bit_position(parts[1])
                out.append((row, col, self.matrix.get(row, col)))
        return out

    def suggest_continuations(self, exchange_index: int, k: int) -> List[Tuple[str, int]]:
        """Ranked next-token candidates after the last token of exchange i.

        Boundary markers are structural, not vocabulary — they are excluded
        from the suggestion list (the matrix itself keeps them).
        """
        if exchange_index >= len(self.exchanges):
            return []
        unigrams = self.exchanges[exchange_index].ngrams(1)
        if not unigrams:
            return []
        row = bit_position(unigrams[-1])
        index = self.vocabulary_index()
        out: List[Tuple[str, int]] = []
        for col, count in self.matrix.continuations(row, k * 4):
            group = index.get(col)
            if not group:
                out.append((f"bit:{col}", count))
            else:
                for token in group:
                    if not is_boundary(token):
                        out.append((token, count))
        out.sort(key=lambda item: (-item[1], item[0]))
        return out[:k]

    # ── Cross-exchange materialization (same bit, different tokens) ───────

    def bit_origins(self, bit: int) -> List[Tuple[int, List[str]]]:
        out = []
        for i, ex in enumerate(self.exchanges):
            tokens = _tokens_at_bit(ex, bit)
            if tokens:
                out.append((i, tokens))
        return out

    def shared_bit_origins(self, i: int, j: int) -> List[SharedBitOrigin]:
        if i >= len(self.exchanges) or j >= len(self.exchanges):
            return []
        a, b = self.exchanges[i], self.exchanges[j]
        shared = a.hll_1.intersection(b.hll_1)
        out = []
        for reg, tz in shared.active_positions():
            bit = reg * 32 + tz
            from_a = _tokens_at_bit(a, bit)
            from_b = _tokens_at_bit(b, bit)
            out.append(SharedBitOrigin(bit, from_a, from_b, from_a == from_b))
        out.sort(key=lambda o: o.bit)
        return out

    # ── LLM utilization ────────────────────────────────────────────────────

    def restore(self, max_paths: int = 64, max_depth: int = 512) -> "RestoredConversation":
        return restore_context(self, max_paths, max_depth)

    def to_prompt(
        self,
        include_metadata: bool = True,
        top_links: int = 5,
        predicted_continuations: int = 3,
    ) -> str:
        return build_prompt(
            self,
            include_metadata=include_metadata,
            top_links=top_links,
            predicted_continuations=predicted_continuations,
        )

    def roundtrip(self) -> "RoundTripReport":
        return roundtrip_context(self)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"ConversationContext({len(self.exchanges)} exchanges, "
            f"top={self.top.hll.popcount()} bits, "
            f"matrix={self.matrix.nonzero_cells()} cells)"
        )


# ── 3-gram De Bruijn restoration ───────────────────────────────────────────

@dataclass
class RestoredConversation:
    """Restored exchange paths from the top unions + LUT only."""

    paths: List[List[str]] = field(default_factory=list)
    confidence: float = 0.0

    def len(self) -> int:
        return len(self.paths)

    def is_empty(self) -> bool:
        return not self.paths

    def flat_tokens(self) -> List[str]:
        seen = set()
        out = []
        for path in self.paths:
            for token in path:
                if token not in seen:
                    seen.add(token)
                    out.append(token)
        return out

    def to_prompt(self, latest_query: str, first_role: str = "user") -> str:
        return build_prompt_from_restored(self, latest_query, first_role)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"RestoredConversation({len(self.paths)} paths, conf={self.confidence:.3f})"


def _candidates(hllset: hllset_py.HLLSet, lut: hllset_py.TokenLut, shape: int) -> set:
    """All LUT tokens at the active positions of ``hllset`` with the given
    n-gram shape (number of NUL-separated parts)."""
    out = set()
    for reg, tz in hllset.active_positions():
        for token in lut.lookup_position(reg, tz):
            parts = split_ngram(token)
            if len(parts) == shape and all(parts):
                out.add(token)
    return out


def _valid_1gram(token: str, grams1: set, bits1: set) -> bool:
    if is_boundary(token):
        return True
    return token in grams1 and bit_position(token) in bits1


def _valid_2gram(token: str, grams2: set, bits2: set) -> bool:
    return token in grams2 and bit_position(token) in bits2


def _is_end_node(node: str) -> bool:
    parts = split_ngram(node)
    return bool(parts) and parts[-1] == END


def _path_to_tokens(path: List[str]) -> List[str]:
    out: List[str] = []
    for i, node in enumerate(path):
        parts = split_ngram(node)
        if not parts:
            continue
        if i == 0 and not is_boundary(parts[0]):
            out.append(parts[0])
        last = parts[-1]
        if not is_boundary(last):
            out.append(last)
    return out


def _dfs_paths(
    adj: Dict[str, List[Tuple[str, str]]],
    node: str,
    used: set,
    path: List[str],
    max_paths: int,
    max_depth: int,
    results: List[List[str]],
) -> None:
    if len(results) >= max_paths or len(path) > max_depth:
        return
    if _is_end_node(node):
        results.append(list(path))
        return
    for nxt, _triple in sorted(adj.get(node, []), key=lambda pair: pair[0]):
        key = (node, nxt)
        if key in used:
            continue
        used.add(key)
        path.append(nxt)
        _dfs_paths(adj, nxt, used, path, max_paths, max_depth, results)
        path.pop()
        used.remove(key)


def restore_with_lut(
    ctx: ConversationContext,
    lut: hllset_py.TokenLut,
    max_paths: int = 64,
    max_depth: int = 512,
) -> RestoredConversation:
    """Restore exchange paths from the context top unions via a LUT.

    This is the 3-gram De Bruijn restore: nodes are 2-grams, edges are
    cross-validated 3-grams, and every START→END path is an exchange.
    """
    top = ctx.top
    bits1 = {reg * 32 + tz for reg, tz in top.hll_1.active_positions()}
    bits2 = {reg * 32 + tz for reg, tz in top.hll_2.active_positions()}

    grams1 = _candidates(top.hll_1, lut, 1)
    grams2 = _candidates(top.hll_2, lut, 2)
    grams3 = _candidates(top.hll_3, lut, 3)

    adj: Dict[str, List[Tuple[str, str]]] = {}
    for triple in grams3:
        parts = split_ngram(triple)
        if len(parts) != 3 or not all(parts):
            continue
        a, b, c = parts
        if not (
            _valid_1gram(a, grams1, bits1)
            and _valid_1gram(b, grams1, bits1)
            and _valid_1gram(c, grams1, bits1)
        ):
            continue
        ab = join_ngram([a, b])
        bc = join_ngram([b, c])
        if not (_valid_2gram(ab, grams2, bits2) and _valid_2gram(bc, grams2, bits2)):
            continue
        adj.setdefault(ab, []).append((bc, triple))

    starts = sorted(node for node in adj if split_ngram(node)[0] == START)

    results: List[List[str]] = []
    for start in starts:
        _dfs_paths(adj, start, set(), [start], max_paths, max_depth, results)

    results = [p for i, p in enumerate(results) if p not in results[:i]]
    results = results[:max_paths]

    used_triples = set()
    for path in results:
        for i in range(len(path) - 1):
            node, nxt = path[i], path[i + 1]
            for cand_next, triple in adj.get(node, []):
                if cand_next == nxt:
                    used_triples.add(triple)
                    break

    total = top.hll_3.popcount()
    confidence = (len(used_triples) / total) if results and total else 0.0

    return RestoredConversation(
        paths=[_path_to_tokens(p) for p in results],
        confidence=min(1.0, confidence),
    )


def restore_context(
    ctx: ConversationContext, max_paths: int = 64, max_depth: int = 512
) -> RestoredConversation:
    """Restore with a perfect LUT (all n-grams of all exchanges)."""
    lut = hllset_py.TokenLut()
    for ex in ctx.exchanges:
        for n in (1, 2, 3):
            lut.record_all(ex.ngrams(n))
    return restore_with_lut(ctx, lut, max_paths, max_depth)


# ── Prompt materialization ─────────────────────────────────────────────────

def build_prompt(
    ctx: ConversationContext,
    include_metadata: bool = True,
    top_links: int = 5,
    predicted_continuations: int = 3,
) -> str:
    """Materialize the live context into a role-labelled LLM prompt."""
    lines = ["You are continuing a conversation. Below is the reconstructed context.", ""]
    if include_metadata:
        lines.append(
            f"[context: {len(ctx.exchanges)} exchanges | "
            f"union popcount={ctx.top.hll.popcount()} | "
            f"cardinality~{ctx.top.hll.cardinality():.0f}]"
        )
        if top_links > 0:
            links = [
                (p, s, n)
                for p, s, n in ctx.matrix.top_links(top_links * 4)
                if not (is_boundary(p) or is_boundary(s))
            ][:top_links]
            if links:
                items = "; ".join(f"{p} → {s} ×{n}" for p, s, n in links)
                lines.append(f"[top follow links: {items}]")
        if predicted_continuations > 0 and ctx.exchanges:
            preds = ctx.suggest_continuations(
                len(ctx.exchanges) - 1, predicted_continuations
            )
            if preds:
                items = "; ".join(f"{t} ×{n}" for t, n in preds)
                lines.append(f"[predicted continuations: {items}]")
        lines.append("")
    for ex in ctx.exchanges:
        text = ex.text or " ".join(ex.tokens)
        lines.append(f"{ex.role}: {text}")
    lines.append("assistant:")
    return "\n".join(lines)


def build_prompt_from_restored(
    restored: RestoredConversation, latest_query: str, first_role: str = "user"
) -> str:
    """Materialize restored De Bruijn paths into an LLM prompt."""
    lines = [
        "You are continuing a conversation reconstructed from HLLSet context.",
        "",
    ]
    role = first_role
    for path in restored.paths:
        lines.append(f"{role}: {' '.join(path)}")
        role = _toggle_role(role)
    lines.append(f"user: {latest_query}")
    lines.append("assistant:")
    return "\n".join(lines)


# ── Round-trip composition law ─────────────────────────────────────────────

@dataclass
class RoundTripReport:
    """Result of one materialize → re-ingest round trip."""

    paths: List[List[str]] = field(default_factory=list)
    hll_1_exact: bool = False
    hll_2_exact: bool = False
    hll_3_exact: bool = False
    hll_exact: bool = False
    hll_1_jaccard: float = 0.0
    hll_2_jaccard: float = 0.0
    hll_3_jaccard: float = 0.0
    hll_jaccard: float = 0.0

    def all_exact(self) -> bool:
        return (
            self.hll_1_exact
            and self.hll_2_exact
            and self.hll_3_exact
            and self.hll_exact
        )

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"RoundTripReport(exact=1:{self.hll_1_exact} 2:{self.hll_2_exact} "
            f"3:{self.hll_3_exact} all:{self.hll_exact}, "
            f"j=({self.hll_1_jaccard:.3f},{self.hll_2_jaccard:.3f},"
            f"{self.hll_3_jaccard:.3f}), paths={len(self.paths)})"
        )


def _layer_report(
    a: hllset_py.HLLSet, b: hllset_py.HLLSet
) -> Tuple[bool, float]:
    return (a.content_key() == b.content_key(), a.jaccard(b))


def _reingest_and_compare(
    original: ConversationContext, restored: RestoredConversation
) -> RoundTripReport:
    reborn = ConversationContext()
    role = original.exchanges[0].role if original.exchanges else "user"
    for path in restored.paths:
        reborn.add_exchange(Exchange.from_tokens(role, path))
        role = _toggle_role(role)

    report = RoundTripReport(paths=[list(p) for p in restored.paths])
    report.hll_1_exact, report.hll_1_jaccard = _layer_report(
        original.top.hll_1, reborn.top.hll_1
    )
    report.hll_2_exact, report.hll_2_jaccard = _layer_report(
        original.top.hll_2, reborn.top.hll_2
    )
    report.hll_3_exact, report.hll_3_jaccard = _layer_report(
        original.top.hll_3, reborn.top.hll_3
    )
    report.hll_exact, report.hll_jaccard = _layer_report(
        original.top.hll, reborn.top.hll
    )
    return report


def roundtrip_exchange(exchange: Exchange) -> RoundTripReport:
    ctx = ConversationContext.build([exchange])
    return _reingest_and_compare(ctx, restore_context(ctx))


def roundtrip_exchange_with_extra_tokens(
    exchange: Exchange, extra_tokens: List[str]
) -> RoundTripReport:
    """Round-trip with a noisy LUT (the exchange's n-grams + distractors).

    This is where the composition law's "high probability" clause is
    observable: a false candidate must survive shape filtering *and*
    1-gram/2-gram cross-validation to inject a spurious edge.
    """
    ctx = ConversationContext.build([exchange])
    lut = hllset_py.TokenLut()
    for n in (1, 2, 3):
        lut.record_all(exchange.ngrams(n))
    lut.record_all(extra_tokens)
    return _reingest_and_compare(ctx, restore_with_lut(ctx, lut))


def roundtrip_context(ctx: ConversationContext) -> RoundTripReport:
    return _reingest_and_compare(ctx, restore_context(ctx))


__all__ = [
    # n-gram conventions
    "START",
    "END",
    "SEP",
    "bit_position",
    "generate_ngrams",
    "join_ngram",
    "split_ngram",
    # exchange
    "Exchange",
    "BitAnchors",
    # matrix
    "SparseAdjacencyMatrix",
    "TokenIndex",
    "UnfoldedCell",
    # context
    "ContextTop",
    "ConversationContext",
    "SharedBitOrigin",
    # restore
    "RestoredConversation",
    "restore_context",
    "restore_with_lut",
    # prompt
    "build_prompt",
    "build_prompt_from_restored",
    # round-trip
    "RoundTripReport",
    "roundtrip_exchange",
    "roundtrip_exchange_with_extra_tokens",
    "roundtrip_context",
]
