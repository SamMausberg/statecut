"""Hierarchical exact-cut filter with bounded exploration and raw-leaf reuse."""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as F
from typing import Callable, Generic, TypeVar
from .arithmetic import Interval, isum, certify_bf16, ReferenceDenominatorError
from .attention import exact_contribution, Contribution
from .forest import ForestCache, Node, TreeReads
from .residual import (weight_box, summary_residual, intersect, choose_cell,
                       cell_accepts)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Piece:
    node: Node
    weights: Interval | None = None
    exact: Contribution | None = None

    @property
    def denominator(self) -> Interval:
        if self.exact is not None:
            return self.exact.denominator
        assert self.weights is not None
        return self.weights.scale(self.node.summary.n)

    def numerator(self, j: int) -> Interval:
        if self.exact is not None:
            return self.exact.numerator[j]
        assert self.weights is not None
        s, a, b = self.node.summary, self.weights.lo, self.weights.hi
        return Interval(a*s.positive[j]+b*s.negative[j],
                        b*s.positive[j]+a*s.negative[j])

    def residual(self, j: int, t: F) -> Interval:
        if self.exact is not None:
            return self.exact.numerator[j]-self.exact.denominator.scale(t)
        assert self.weights is not None
        return summary_residual(self.node.summary, self.weights, j, t)


def summarize(node: Node, q: tuple[Interval, ...], reads: TreeReads) -> Piece:
    reads.summary_nodes += 1
    return Piece(node, weights=weight_box(q, node.summary))


def estimates(pieces: list[Piece]) -> tuple[F, ...]:
    nvalues = len(pieces[0].node.summary.positive)
    ds = [p.denominator.lo+p.denominator.hi for p in pieces]
    denominator = sum(ds, F(0))
    if denominator <= 0:
        return (F(0),)*nvalues
    # Untrusted proposal. Soundness comes only from the later boundary tests.
    out = []
    for j in range(nvalues):
        num = F(0)
        for p, d in zip(pieces, ds):
            if p.exact is not None:
                num += 2*p.exact.numerator[j].lo
            else:
                s = p.node.summary
                num += d*(s.positive[j]+s.negative[j])/s.n
        out.append(num/denominator)
    return tuple(out)


def centered_enclosure(pieces: list[Piece]) -> tuple[Interval, ...] | None:
    d = isum(p.denominator for p in pieces)
    if d.lo <= 0:
        return None
    centers = estimates(pieces)
    out = []
    for j, t in enumerate(centers):
        r = isum(p.residual(j, t) for p in pieces)
        box = Interval.point(t)+r/d
        hull = Interval(min(p.node.summary.vmin[j] for p in pieces),
                        max(p.node.summary.vmax[j] for p in pieces))
        original = isum(p.numerator(j) for p in pieces)/d
        out.append(intersect(intersect(box, original), hull))
    return tuple(out)


def boundary_bf16(pieces: list[Piece]) -> tuple[F, ...] | None:
    if isum(p.denominator for p in pieces).lo <= 0:
        return None
    try:
        cells = tuple(choose_cell(t) for t in estimates(pieces))
    except OverflowError:
        return None
    for j, cell in enumerate(cells):
        lo = isum(p.residual(j, cell.lo) for p in pieces)
        hi = isum(p.residual(j, cell.hi) for p in pieces)
        if not cell_accepts(lo, hi, cell):
            return None
    return tuple(c.value for c in cells)


@dataclass(frozen=True, slots=True)
class TreeVerified(Generic[T]):
    value: T
    accepted_from_bounds: bool
    used_dense_fallback: bool
    stats: TreeReads
    attention_bounds: tuple[Interval, ...]
    gate: str


def dense_tree_attention(cache: ForestCache, q: tuple[F, ...],
                         reads: TreeReads | None = None) -> tuple[F, ...]:
    reads = reads if reads is not None else TreeReads()
    contributions = [exact_contribution(q, cache.read_leaf(n, reads)) for n in cache.iter_leaves()]
    if not contributions:
        raise ValueError("empty attention")
    d = sum((c.denominator.lo for c in contributions), F(0))
    if d <= 0:
        raise ReferenceDenominatorError("reference denominator zero")
    return tuple(sum((c.numerator[j].lo for c in contributions), F(0))/d
                 for j in range(len(contributions[0].numerator)))


def root_enclosure(cache: ForestCache, q: tuple[Interval, ...],
                   reads: TreeReads | None = None) -> tuple[Interval, ...] | None:
    if cache.length == 0:
        raise ValueError("empty attention")
    reads = reads if reads is not None else TreeReads()
    pieces = [summarize(n, q, reads) for n in cache.frontier]
    reads.frontier_peak = max(reads.frontier_peak, len(pieces))
    return centered_enclosure(pieces)


def verify_tree_attention(cache: ForestCache, q: tuple[F, ...],
                          consumer: Callable[[tuple[Interval, ...]], T | None],
                          exact_consumer: Callable[[tuple[F, ...]], T], *,
                          max_expansions: int | None = 64,
                          direct_bf16: bool = False) -> TreeVerified[T]:
    """Try root summaries, refine a disjoint frontier, then fail closed.

    max_expansions bounds refinement actions (splits or leaf reads), not total
    runtime. Root setup costs O(log N); this Python scheduler is O(frontier)
    per action, NOT the proposed GPU priority-queue runtime. Fallback reads
    each raw leaf at most once per call, including previously opened leaves.
    The direct_bf16 path is permitted ONLY with the built-in BF16 consumer.
    """
    if cache.length == 0:
        raise ValueError("empty attention")
    if max_expansions is not None and max_expansions < 0:
        raise ValueError("negative expansion budget")
    if direct_bf16 and consumer is not certify_bf16:
        raise ValueError("direct_bf16 requires the built-in certify_bf16 consumer")
    reads = TreeReads()
    qi = tuple(Interval.point(x) for x in q)
    try:
        pieces = [summarize(n, qi, reads) for n in cache.frontier]
    except (ArithmeticError, OverflowError):
        # A loose summary score box may exceed the oracle domain even when
        # every actual row score is valid. No raw leaves have been opened yet.
        vals = dense_tree_attention(cache, q, reads)
        return TreeVerified(exact_consumer(vals), False, True, reads,
                            tuple(Interval.point(x) for x in vals),
                            "dense-fallback:summary-resource")
    expansions = 0
    while True:
        reads.frontier_peak = max(reads.frontier_peak, len(pieces))
        reads.consumer_calls += 1
        enc = centered_enclosure(pieces)
        if direct_bf16:
            value = boundary_bf16(pieces)
            if value is not None:
                assert enc is not None
                return TreeVerified(value, True, False, reads, enc, "direct-boundary")
        if enc is not None:
            value = consumer(enc)
            if value is not None:
                return TreeVerified(value, True, False, reads, enc, "centered-interval")
        remaining = [i for i, p in enumerate(pieces) if p.exact is None]
        if not remaining or (max_expansions is not None and expansions >= max_expansions):
            # Consume unopened subtrees without revisiting previously opened leaves.
            complete: list[Contribution] = []
            for p in pieces:
                if p.exact is not None:
                    complete.append(p.exact)
                    continue
                stack = [p.node]
                while stack:
                    node = stack.pop()
                    if node.leaf:
                        complete.append(exact_contribution(q, cache.read_leaf(node, reads)))
                    else:
                        assert node.left is not None and node.right is not None
                        stack.extend((node.right, node.left))
            d = sum((c.denominator.lo for c in complete), F(0))
            if d <= 0:
                raise ReferenceDenominatorError("reference denominator zero")
            vals = tuple(sum((c.numerator[j].lo for c in complete), F(0))/d
                         for j in range(len(complete[0].numerator)))
            return TreeVerified(exact_consumer(vals), False, True, reads,
                                tuple(Interval.point(x) for x in vals), "dense-fallback")
        def priority(i: int) -> tuple[F, int]:
            p = pieces[i]
            s = p.node.summary
            assert p.weights is not None
            return (s.n*p.weights.width*sum((u-l for l, u in zip(s.vmin, s.vmax)), F(0)), -s.n)
        index = max(remaining, key=priority)
        p = pieces[index]
        if p.node.leaf:
            c = exact_contribution(q, cache.read_leaf(p.node, reads))
            pieces[index] = Piece(p.node, exact=c)
        else:
            assert p.node.left is not None and p.node.right is not None
            try:
                children = [summarize(p.node.left, qi, reads),
                            summarize(p.node.right, qi, reads)]
            except (ArithmeticError, OverflowError):
                # Keep the existing sound frontier; the next iteration uses
                # its exact fallback and reuses all previously opened leaves.
                max_expansions = expansions
                continue
            pieces[index:index+1] = children
            reads.splits += 1
        expansions += 1
