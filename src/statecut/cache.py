"""Append-only exact cache with immutable signed-moment summaries.

Summaries are built from the actual cache. They are not untrusted certificates
accepted from callers. An external adapter needs equivalent provenance checks.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as F


@dataclass(frozen=True, slots=True)
class Entry:
    k: tuple[F, ...]
    v: tuple[F, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "k", tuple(F(x) for x in self.k))
        object.__setattr__(self, "v", tuple(F(x) for x in self.v))
        if not self.k or not self.v:
            raise ValueError("zero-dimensional entry")


@dataclass(frozen=True, slots=True)
class Summary:
    n: int
    kmin: tuple[F, ...]
    kmax: tuple[F, ...]
    positive: tuple[F, ...]
    negative: tuple[F, ...]
    vmin: tuple[F, ...]
    vmax: tuple[F, ...]

    @classmethod
    def single(cls, e: Entry) -> Summary:
        return cls(1, e.k, e.k, tuple(max(x, 0) for x in e.v),
                   tuple(min(x, 0) for x in e.v), e.v, e.v)

    def append(self, e: Entry) -> Summary:
        if len(e.k) != len(self.kmin) or len(e.v) != len(self.positive):
            raise ValueError("cache dimension mismatch")
        return Summary(self.n+1,
                       tuple(min(a,b) for a,b in zip(self.kmin,e.k)),
                       tuple(max(a,b) for a,b in zip(self.kmax,e.k)),
                       tuple(a+max(b,0) for a,b in zip(self.positive,e.v)),
                       tuple(a+min(b,0) for a,b in zip(self.negative,e.v)),
                       tuple(min(a,b) for a,b in zip(self.vmin,e.v)),
                       tuple(max(a,b) for a,b in zip(self.vmax,e.v)))


@dataclass(frozen=True, slots=True)
class Block:
    summary: Summary
    _entries: tuple[Entry, ...]


@dataclass(slots=True)
class ReadCounter:
    raw_entries: int = 0
    raw_scalars: int = 0
    blocks: int = 0
    summary_blocks: int = 0


@dataclass(frozen=True, slots=True)
class Cache:
    identity: str
    block_size: int = 32
    blocks: tuple[Block, ...] = ()
    length: int = 0

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError("block_size must be positive")

    def append(self, entry: Entry) -> Cache:
        if self.blocks:
            first = self.blocks[0].summary
            if len(entry.k) != len(first.kmin) or len(entry.v) != len(first.positive):
                raise ValueError("cache dimension mismatch")
        if self.blocks and self.blocks[-1].summary.n < self.block_size:
            last = self.blocks[-1]
            b = Block(last.summary.append(entry), last._entries + (entry,))
            blocks = self.blocks[:-1] + (b,)
        else:
            blocks = self.blocks + (Block(Summary.single(entry), (entry,)),)
        return Cache(self.identity, self.block_size, blocks, self.length+1)

    def read_block(self, index: int, counter: ReadCounter) -> tuple[Entry, ...]:
        b = self.blocks[index]
        counter.blocks += 1
        counter.raw_entries += b.summary.n
        counter.raw_scalars += b.summary.n * (len(b.summary.kmin)+len(b.summary.positive))
        return b._entries

    def audit(self) -> None:
        """Full scan for TESTS / ingestion only, never a query-path check."""
        if self.length < 0 or self.length != sum(b.summary.n for b in self.blocks):
            raise ValueError("invalid coverage")
        dimensions = None
        for i, b in enumerate(self.blocks):
            shape = (len(b.summary.kmin), len(b.summary.positive))
            if dimensions is not None and shape != dimensions:
                raise ValueError("inconsistent block dimensions")
            dimensions = shape
            if not b._entries or b.summary.n != len(b._entries):
                raise ValueError("invalid block count")
            if b.summary.n > self.block_size or (i < len(self.blocks)-1 and b.summary.n != self.block_size):
                raise ValueError("invalid block partition")
            s = Summary.single(b._entries[0])
            for e in b._entries[1:]:
                s = s.append(e)
            if s != b.summary:
                raise ValueError("summary does not describe raw cache")
