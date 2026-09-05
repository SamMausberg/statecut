"""Adaptive exact-reference attention using summary-only first evaluation."""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as F
from typing import Callable, TypeVar
from .arithmetic import Interval, exp_reference, isum
from .cache import Cache, Summary, Entry, ReadCounter

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Contribution:
    denominator: Interval
    numerator: tuple[Interval, ...]


def score(q: tuple[F, ...], k: tuple[F, ...]) -> F:
    if len(q) != len(k):
        raise ValueError("query/key dimension mismatch")
    # Match Interval.point's exact interpretation of numeric inputs before
    # multiplying: otherwise a float query can make dense and filtered
    # reductions disagree through cancellation in this supposedly exact dot.
    return sum((F(a)*F(b) for a,b in zip(q,k)), F(0))


def summary_contribution(q: tuple[F, ...], s: Summary) -> Contribution:
    q = tuple(F(x) for x in q)
    if len(q) != len(s.kmin) or len(s.kmin) != len(s.kmax):
        raise ValueError("query/key dimension mismatch")
    if not s.positive or len(s.positive) != len(s.negative) or s.n < 1:
        raise ValueError("invalid summary shape/count")
    if any(a > b for a,b in zip(s.kmin,s.kmax)):
        raise ValueError("reversed key box")
    if any(p < 0 for p in s.positive) or any(m > 0 for m in s.negative):
        raise ValueError("invalid signed moments")
    sl = sum((x*(a if x >= 0 else b) for x,a,b in zip(q,s.kmin,s.kmax)), F(0))
    su = sum((x*(b if x >= 0 else a) for x,a,b in zip(q,s.kmin,s.kmax)), F(0))
    a, b = exp_reference(sl), exp_reference(su)
    # P >= 0, M <= 0. Reversing the coefficient on M is essential.
    ns = tuple(Interval(a*p+b*m, b*p+a*m) for p,m in zip(s.positive,s.negative))
    return Contribution(Interval(s.n*a,s.n*b), ns)


def exact_contribution(q: tuple[F, ...], entries: tuple[Entry, ...]) -> Contribution:
    if not entries:
        raise ValueError("empty block")
    if any(len(e.v) != len(entries[0].v) for e in entries):
        raise ValueError("inconsistent value dimensions")
    ws = tuple(exp_reference(score(q,e.k)) for e in entries)
    d = sum(ws,F(0))
    ns = tuple(Interval.point(sum((w*e.v[j] for w,e in zip(ws,entries)),F(0)))
               for j in range(len(entries[0].v)))
    return Contribution(Interval.point(d),ns)


def enclosure(cs: tuple[Contribution, ...]) -> tuple[Interval, ...] | None:
    if not cs:
        raise ValueError("empty attention")
    if not cs[0].numerator or any(len(c.numerator) != len(cs[0].numerator) for c in cs):
        raise ValueError("inconsistent contribution dimensions")
    d = isum(c.denominator for c in cs)
    if d.lo <= 0:
        return None
    return tuple(isum(c.numerator[j] for c in cs)/d for j in range(len(cs[0].numerator)))


def dense_attention(cache: Cache, q: tuple[F, ...], counter: ReadCounter | None = None) -> tuple[F, ...]:
    counter = counter if counter is not None else ReadCounter()
    cs = tuple(exact_contribution(q,cache.read_block(i,counter)) for i in range(len(cache.blocks)))
    enc = enclosure(cs)
    if enc is None:
        raise ArithmeticError("reference denominator zero")
    assert all(x.lo == x.hi for x in enc)
    return tuple(x.lo for x in enc)


@dataclass(frozen=True, slots=True)
class Verified:
    value: object
    accepted_from_bounds: bool
    used_dense_fallback: bool
    opened_blocks: tuple[int, ...]
    stats: ReadCounter
    attention_bounds: tuple[Interval, ...]


def verify_attention(cache: Cache, q: tuple[F, ...],
                     consumer: Callable[[tuple[Interval, ...]], T | None],
                     exact_consumer: Callable[[tuple[F, ...]], T],
                     max_refinements: int | None = None) -> Verified:
    """A consumer may accept ONLY a sound certificate, not an approximation.

    User-defined consumer soundness is a proof obligation. Package consumers
    are exact BF16 endpoint agreement and tie-aware logit separation.
    max_refinements limits summary/refinement work, not correctness.
    Dense fallback reuses previously opened blocks; each raw block read once.
    """
    if not cache.blocks:
        raise ValueError("empty attention")
    if max_refinements is not None and max_refinements < 0:
        raise ValueError("negative budget")
    counter = ReadCounter()
    try:
        cs = []
        for block in cache.blocks:
            counter.summary_blocks += 1
            cs.append(summary_contribution(q, block.summary))
    except (ArithmeticError, OverflowError):
        # Correlations can keep every raw score inside the oracle domain even
        # when a loose coordinate box exceeds it. No raw block is open yet.
        vals = dense_attention(cache, q, counter)
        return Verified(exact_consumer(vals), False, True,
                        tuple(range(len(cache.blocks))), counter,
                        tuple(Interval.point(x) for x in vals))
    remaining = set(range(len(cs)))
    opened: list[int] = []
    while True:
        enc = enclosure(tuple(cs))
        if enc is not None:
            value = consumer(enc)
            if value is not None:
                return Verified(value,True,False,tuple(opened),counter,enc)
        if not remaining:
            if enc is None:
                raise ArithmeticError("reference denominator zero")
            if any(x.lo != x.hi for x in enc):
                raise ArithmeticError("fallback did not recover exact reference values")
            value = exact_consumer(tuple(x.lo for x in enc))
            return Verified(value,False,True,tuple(opened),counter,enc)
        if max_refinements is not None and len(opened) >= max_refinements:
            # Fail closed. Do not emit the draft or commit approximate state.
            for i in sorted(remaining):
                cs[i] = exact_contribution(q,cache.read_block(i,counter))
                opened.append(i)
            enc = enclosure(tuple(cs))
            if enc is None:
                raise ArithmeticError("reference denominator zero")
            if any(x.lo != x.hi for x in enc):
                raise ArithmeticError("fallback did not recover exact reference values")
            value = exact_consumer(tuple(x.lo for x in enc))
            return Verified(value,False,True,tuple(opened),counter,enc)
        # Scheduling is untrusted: any choice is sound and raw blocks are unique.
        def priority(i: int) -> F:
            s = cache.blocks[i].summary
            vmax = max((abs(x) for x in s.vmin+s.vmax), default=F(0))
            return sum((x.width for x in cs[i].numerator),F(0))+vmax*cs[i].denominator.width
        i = max(remaining,key=lambda j:(priority(j),-j))
        cs[i] = exact_contribution(q,cache.read_block(i,counter))
        remaining.remove(i)
        opened.append(i)
