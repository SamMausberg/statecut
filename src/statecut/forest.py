"""Immutable binary-counter KV forest; no query-time audit or flat index scan.

Exact rows are retained in leaves. A partial tail never participates in carries.
There are at most popcount(sealed_blocks) roots plus one partial tail. Appends
copy at most one fixed-size tail and the logarithmic root list, not all KV rows.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from operator import index
from .cache import Entry, Summary


def merge_summary(a: Summary, b: Summary) -> Summary:
    if len(a.kmin) != len(b.kmin) or len(a.positive) != len(b.positive):
        raise ValueError("summary shape mismatch")
    return Summary(a.n+b.n,
                   tuple(map(min, a.kmin, b.kmin)), tuple(map(max, a.kmax, b.kmax)),
                   tuple(x+y for x, y in zip(a.positive, b.positive)),
                   tuple(x+y for x, y in zip(a.negative, b.negative)),
                   tuple(map(min, a.vmin, b.vmin)), tuple(map(max, a.vmax, b.vmax)))


@dataclass(frozen=True, slots=True)
class Node:
    start: int
    summary: Summary
    left: Node | None = None
    right: Node | None = None
    _rows: tuple[Entry, ...] = field(default=(), repr=False)

    @property
    def stop(self) -> int:
        return self.start+self.summary.n

    @property
    def leaf(self) -> bool:
        return self.left is None


def parent(a: Node, b: Node) -> Node:
    if a.stop != b.start or a.summary.n != b.summary.n:
        raise ValueError("noncontiguous or unequal binary carry")
    return Node(a.start, merge_summary(a.summary, b.summary), a, b)


@dataclass(slots=True)
class TreeReads:
    summary_nodes: int = 0
    raw_entries: int = 0
    raw_scalars: int = 0
    splits: int = 0
    opened_leaves: int = 0
    frontier_peak: int = 0
    consumer_calls: int = 0


@dataclass(frozen=True, slots=True)
class ForestCache:
    identity: str
    block_size: int = 32
    roots: tuple[Node, ...] = ()
    tail: Node | None = None
    length: int = 0

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "block_size", index(self.block_size))
        except TypeError as exc:
            raise ValueError("block_size must be an integer") from exc
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")

    @property
    def frontier(self) -> tuple[Node, ...]:
        return self.roots + ((self.tail,) if self.tail is not None else ())

    def append(self, e: Entry) -> ForestCache:
        first = self.roots[0] if self.roots else self.tail
        if first is not None:
            if (len(e.k) != len(first.summary.kmin) or
                    len(e.v) != len(first.summary.positive)):
                raise ValueError("cache dimension mismatch")
        if self.tail is None:
            tail = Node(self.length, Summary.single(e), _rows=(e,))
        else:
            tail = Node(self.tail.start, self.tail.summary.append(e),
                        _rows=self.tail._rows+(e,))
        roots = list(self.roots)
        if tail.summary.n == self.block_size:
            carry = tail
            while roots and roots[-1].summary.n == carry.summary.n:
                carry = parent(roots.pop(), carry)
            roots.append(carry)
            tail = None
        return ForestCache(self.identity, self.block_size, tuple(roots), tail, self.length+1)

    @staticmethod
    def read_leaf(node: Node, reads: TreeReads) -> tuple[Entry, ...]:
        if not node.leaf or not node._rows:
            raise ValueError("raw read requires nonempty leaf")
        reads.opened_leaves += 1
        reads.raw_entries += len(node._rows)
        reads.raw_scalars += len(node._rows)*(len(node._rows[0].k)+len(node._rows[0].v))
        return node._rows

    def iter_leaves(self):
        """Metadata traversal for fallback, exports and audits, not root queries."""
        stack = list(reversed(self.frontier))
        while stack:
            node = stack.pop()
            if node.leaf:
                yield node
            else:
                assert node.left is not None and node.right is not None
                stack.extend((node.right, node.left))

    def audit(self) -> None:
        """Expensive ingestion/test audit. NEVER called by the query path."""
        dimensions = None
        def check(node: Node, sealed: bool) -> Summary:
            nonlocal dimensions
            if node.leaf:
                if node.right is not None or not node._rows:
                    raise ValueError("malformed leaf")
                if sealed and len(node._rows) != self.block_size:
                    raise ValueError("sealed leaf must contain exactly block_size rows")
                s = Summary.single(node._rows[0])
                for row in node._rows[1:]:
                    s = s.append(row)
                shape = (len(s.kmin), len(s.positive))
                if dimensions is not None and shape != dimensions:
                    raise ValueError("inconsistent leaf dimensions")
                dimensions = shape
            else:
                if node.right is None or node.left is None or node._rows:
                    raise ValueError("malformed internal node")
                if node.left.stop != node.right.start or node.left.start != node.start:
                    raise ValueError("noncontiguous children")
                if node.left.summary.n != node.right.summary.n:
                    raise ValueError("unbalanced binary-counter node")
                s = merge_summary(check(node.left, sealed), check(node.right, sealed))
            if s != node.summary:
                raise ValueError("summary provenance mismatch")
            return s
        offset = 0
        prev_blocks = None
        for node in self.roots:
            blocks, remainder = divmod(node.summary.n, self.block_size)
            if remainder or blocks <= 0 or blocks & (blocks-1):
                raise ValueError("root is not a power-of-two sealed block tree")
            if prev_blocks is not None and blocks >= prev_blocks:
                raise ValueError("root ranks not strictly descending")
            prev_blocks = blocks
        for node in self.frontier:
            if node.start != offset:
                raise ValueError("forest coverage gap or overlap")
            check(node, node is not self.tail)
            offset = node.stop
        if self.tail is not None and (not self.tail.leaf or not 0 < self.tail.summary.n < self.block_size):
            raise ValueError("invalid partial tail")
        if offset != self.length:
            raise ValueError("forest length mismatch")
