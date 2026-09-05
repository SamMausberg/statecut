"""Denominator-free, value-translation-invariant attention certificates.

All arithmetic is exact rational. The weight semantics are RATIONAL_BF16_V1.
An independent floating-point attention backend needs its own proven bridge.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as F
from .arithmetic import (Interval, bf16_value, round_bf16_bits, exp_reference,
                         isum)
from .cache import Summary


def intersect(a: Interval, b: Interval) -> Interval:
    lo, hi = max(a.lo, b.lo), min(a.hi, b.hi)
    if lo > hi:
        raise ArithmeticError("inconsistent sound enclosures: provenance/arithmetic defect")
    return Interval(lo, hi)


def weight_box(q: tuple[Interval, ...], s: Summary) -> Interval:
    if len(q) != len(s.kmin) or not q:
        raise ValueError("query/key dimension mismatch")
    z = isum(x * Interval(l, u) for x, l, u in zip(q, s.kmin, s.kmax))
    return Interval(exp_reference(z.lo), exp_reference(z.hi))


def chord_abs_sum(n: int, total: F, lo: F, hi: F, t: F) -> F:
    """Upper bound on sum |v_i-t| using count, sum and range only.

    This is the convex chord of absolute value, not a norm heuristic.
    The l == u case is handled without dividing by zero.
    """
    if n <= 0 or lo > hi or not n*lo <= total <= n*hi:
        raise ValueError("inconsistent count/sum/value range")
    if lo == hi:
        return n * abs(lo-t)
    return ((n*hi-total)*abs(lo-t) + (total-n*lo)*abs(hi-t))/(hi-lo)


def moment_residual(n: int, total: F, lo: F, hi: F,
                    weights: Interval, t: F) -> Interval:
    """Enclose R(t) = sum w_i (v_i-t), retaining shared normalization."""
    a, b = weights.lo, weights.hi
    if a < 0:
        raise ValueError("negative attention weight")
    bound = chord_abs_sum(n, total, lo, hi, t)
    center = (a+b)/2 * (total-n*t)
    radius = (b-a)/2 * bound
    return Interval(center-radius, center+radius)


def summary_residual(s: Summary, weights: Interval, j: int, t: F) -> Interval:
    """Intersection with the original signed-moment residual is always safe."""
    p, m = s.positive[j], s.negative[j]
    direct = moment_residual(s.n, p+m, s.vmin[j], s.vmax[j], weights, t)
    a, b = weights.lo, weights.hi
    numerator = Interval(a*p+b*m, b*p+a*m)
    independent = numerator - Interval(s.n*a, s.n*b).scale(t)
    return intersect(direct, independent)


@dataclass(frozen=True, slots=True)
class RoundingCell:
    bits: int
    value: F
    lo: F
    hi: F
    closed: bool


def bf16_cell(bits: int) -> RoundingCell:
    """Exact finite RNE cell, including subnormal, negative and overflow edges.

    Canonical +0 is the only permitted zero encoding in this reference ABI.
    Both endpoints belong to the cell precisely when its significand is even.
    """
    if bits == 0x8000:
        raise ValueError("negative zero is not canonical")
    value = bf16_value(bits)
    if bits == 0:
        return RoundingCell(bits, F(0), -F(1, 1 << 134), F(1, 1 << 134), True)
    negative = bool(bits & 0x8000)
    magnitude = bits & 0x7fff
    y = bf16_value(magnitude)
    prev = bf16_value(magnitude-1)
    nxt = (bf16_value(magnitude+1) if magnitude < 0x7f7f else y+(y-prev))
    lo, hi = (prev+y)/2, (y+nxt)/2
    if negative:
        lo, hi = -hi, -lo
    return RoundingCell(bits, value, lo, hi, magnitude % 2 == 0)


def cell_accepts(lower_residual: Interval, upper_residual: Interval,
                 cell: RoundingCell) -> bool:
    """Caller must separately establish a strictly positive denominator."""
    if cell.closed:
        return lower_residual.lo >= 0 and upper_residual.hi <= 0
    return lower_residual.lo > 0 and upper_residual.hi < 0


def choose_cell(estimate: F) -> RoundingCell:
    return bf16_cell(round_bf16_bits(estimate))
