"""Exact rational arithmetic and outward enclosures. No float-based acceptance.

The reference is RATIONAL_BF16_V1, NOT a PyTorch/FlashAttention kernel.
See docs/REFERENCE.md. Python and its integer implementation remain trusted.
"""
from __future__ import annotations
from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
from math import isqrt
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Interval:
    lo: F
    hi: F

    def __post_init__(self) -> None:
        object.__setattr__(self, "lo", F(self.lo))
        object.__setattr__(self, "hi", F(self.hi))
        if self.lo > self.hi:
            raise ValueError("reversed interval")

    @classmethod
    def point(cls, x: F | int) -> Interval:
        return cls(F(x), F(x))

    def __add__(self, other: Interval) -> Interval:
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __neg__(self) -> Interval:
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other: Interval) -> Interval:
        return self + (-other)

    def __mul__(self, other: Interval) -> Interval:
        xs = (self.lo * other.lo, self.lo * other.hi,
              self.hi * other.lo, self.hi * other.hi)
        return Interval(min(xs), max(xs))

    def __truediv__(self, other: Interval) -> Interval:
        if other.lo <= 0 <= other.hi:
            raise ZeroDivisionError("interval denominator includes zero")
        return self * Interval(1 / other.hi, 1 / other.lo)

    def scale(self, x: F | int) -> Interval:
        return self * Interval.point(x)

    def square(self) -> Interval:
        lo = F(0) if self.lo <= 0 <= self.hi else min(self.lo**2, self.hi**2)
        return Interval(lo, max(self.lo**2, self.hi**2))

    @property
    def width(self) -> F:
        return self.hi - self.lo

    def contains(self, x: F) -> bool:
        return self.lo <= x <= self.hi


def isum(xs: Iterable[Interval]) -> Interval:
    return sum(xs, Interval.point(0))


def dyadic_outward(x: Interval, bits: int) -> Interval:
    if bits < 1:
        raise ValueError("positive precision required")
    s = 1 << bits
    lo = (x.lo.numerator * s) // x.lo.denominator
    hi = -((-x.hi.numerator * s) // x.hi.denominator)
    return Interval(F(lo, s), F(hi, s))


def exp_enclosure(x: F, bits: int = 96) -> Interval:
    """Enclose REAL exp(x) with a proved Taylor-tail bound.

    Nonnegative range reduction to t <= 1/2. Sum positive series, bound
    remaining tail geometrically, then outward-square. Negative x uses
    reciprocal. No libm and no empirical tolerance is used.
    """
    x = F(x)
    if bits < 8:
        raise ValueError("bits must be >= 8")
    if abs(x) > 1024:
        raise OverflowError("oracle resource guard: |score| > 1024")
    if x == 0:
        return Interval.point(1)
    t, squares = abs(x), 0
    while t > F(1, 2):
        t /= 2
        squares += 1
    workbits = bits + 32 + squares
    target = F(1, 1 << workbits)
    term = total = F(1)
    m = 0
    while True:
        nxt = term * t / (m + 1)
        tail = nxt / (1 - t / (m + 2))
        if tail <= target:
            break
        total += nxt
        term = nxt
        m += 1
    result = dyadic_outward(Interval(total, total + tail), workbits)
    for _ in range(squares):
        result = dyadic_outward(result.square(), workbits)
    if x < 0:
        result = Interval.point(1) / result
    return dyadic_outward(result, bits)


def rne_integer(x: F) -> int:
    """Nearest integer, ties to even, including negative inputs."""
    n = x.numerator // x.denominator
    frac = x - n
    if frac < F(1, 2):
        return n
    if frac > F(1, 2):
        return n + 1
    return n if n % 2 == 0 else n + 1


@lru_cache(maxsize=16384)
def exp_reference(x: F, fractional_bits: int = 24) -> F:
    """E_p(x) = 2^-p RNE(2^p exp(x)), evaluated by rigorous enclosures.

    Not a CUDA exp implementation. Positive inputs to E may round to zero
    for sufficiently negative scores. A zero denominator is rejected.
    A resource limit raises, never silently approximates.
    """
    if not 1 <= fractional_bits <= 256:
        raise ValueError("fractional_bits must be in [1,256]")
    scale = 1 << fractional_bits
    for bits in (64, 128, 256, 512, 1024):
        b = exp_enclosure(F(x), bits)
        a, c = rne_integer(b.lo * scale), rne_integer(b.hi * scale)
        if a == c:
            return F(a, scale)
    raise ArithmeticError("exp rounding unresolved: increase precision, do not accept")


def bf16_value(bits: int) -> F:
    if not 0 <= bits <= 65535:
        raise ValueError("not a uint16 BF16 encoding")
    sign = -1 if bits & 0x8000 else 1
    e, m = (bits >> 7) & 255, bits & 127
    if e == 255:
        raise ValueError("NaN and infinity are outside the reference domain")
    if e == 0:
        return sign * F(m, 1 << 133)
    return sign * F(128 + m) * F(2) ** (e - 134)


@lru_cache(maxsize=1)
def _positive_bf16() -> tuple[F, ...]:
    return tuple(bf16_value(i) for i in range(0x7f80))


def round_bf16_bits(x: F) -> int:
    """Exact rational -> IEEE BF16 nearest/even, finite-only, +0 canonical.

    No intermediate float conversion or double rounding. Overflow fails
    closed. Signed zero is deliberately not part of this reference ABI.
    """
    x = F(x)
    negative = x < 0
    y = abs(x)
    table = _positive_bf16()
    i = bisect_left(table, y)
    if i == len(table):
        threshold = table[-1] + (table[-1] - table[-2]) / 2
        if y >= threshold:
            raise OverflowError("reference BF16 overflow")
        code = len(table) - 1
    elif table[i] == y or i == 0:
        code = i
    else:
        a, b = y - table[i-1], table[i] - y
        code = i-1 if a < b else i if b < a else (i-1 if (i-1) % 2 == 0 else i)
    return code | (0x8000 if negative and code else 0)


def bf16(x: F) -> F:
    return bf16_value(round_bf16_bits(x))


def rounded_interval(x: Interval) -> Interval:
    # RNE is monotone on the finite domain, with canonical zero.
    return Interval(bf16(x.lo), bf16(x.hi))


def certify_bf16(xs: tuple[Interval, ...]) -> tuple[F, ...] | None:
    out: list[F] = []
    try:
        for x in xs:
            a, b = round_bf16_bits(x.lo), round_bf16_bits(x.hi)
            if a != b:
                return None
            out.append(bf16_value(a))
    except OverflowError:
        return None
    return tuple(out)


def sqrt_enclosure(x: F, bits: int = 96) -> Interval:
    x = F(x)
    if x < 0:
        raise ValueError("negative radicand")
    if bits < 1:
        raise ValueError("positive precision required")
    s = 1 << bits
    n = isqrt((x.numerator * s * s) // x.denominator)
    lo = F(n, s)
    return Interval(lo, lo if lo * lo == x else F(n + 1, s))


def sqrt_interval(x: Interval, bits: int = 96) -> Interval:
    return Interval(sqrt_enclosure(x.lo, bits).lo, sqrt_enclosure(x.hi, bits).hi)


def greedy(logits: tuple[F, ...]) -> int:
    if not logits:
        raise ValueError("empty vocabulary")
    return max(range(len(logits)), key=lambda i: (logits[i], -i))


def certify_argmax(xs: tuple[Interval, ...]) -> int | None:
    """Smallest-index tie rule. Equality only allowed against later indices."""
    if not xs:
        raise ValueError("empty vocabulary")
    c = max(range(len(xs)), key=lambda i: (xs[i].lo, -i))
    if all((xs[c].lo > x.hi if j < c else xs[c].lo >= x.hi)
           for j, x in enumerate(xs) if j != c):
        return c
    return None


def bf16_rsqrt(numerator: F, radicand: F) -> F:
    """Correctly round numerator/sqrt(radicand) without an approximate sqrt.

    All ordering and midpoint tests reduce to exact rational square tests.
    This terminates even when the result is exactly a BF16 midpoint.
    """
    if radicand <= 0:
        raise ValueError("strictly positive radicand required")
    n = abs(F(numerator))
    table = _positive_bf16()
    target = n*n
    left,right = 0,len(table)
    while left < right:
        mid = (left+right)//2
        if table[mid]*table[mid]*radicand < target:
            left = mid+1
        else:
            right = mid
    i = left
    if i == len(table):
        boundary = table[-1]+(table[-1]-table[-2])/2
        if target >= boundary*boundary*radicand:
            raise OverflowError("reference BF16 overflow")
        code = len(table)-1
    elif table[i]*table[i]*radicand == target or i == 0:
        code = i
    else:
        boundary = (table[i-1]+table[i])/2
        compare = boundary*boundary*radicand
        code = i-1 if target < compare else i if target > compare else (i-1 if (i-1)%2 == 0 else i)
    return -table[code] if numerator < 0 else table[code]
