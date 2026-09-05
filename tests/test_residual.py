from fractions import Fraction as F
from itertools import combinations_with_replacement
from random import Random
import pytest
from statecut.arithmetic import Interval, bf16, bf16_value, round_bf16_bits
from statecut.residual import (chord_abs_sum, integer_abs_sum, moment_residual,
                              bf16_cell, cell_accepts)


def test_chord_and_residual_randomized():
    rng = Random(1841)
    for _ in range(600):
        n = rng.randrange(1, 25)
        vs = [F(rng.randrange(-100, 101), 16) for _ in range(n)]
        a = F(rng.randrange(0, 20), 7)
        b = a+F(rng.randrange(0, 20), 11)
        ws = [a+(b-a)*F(rng.randrange(0, 17), 16) for _ in vs]
        t = F(rng.randrange(-120, 121), 13)
        total, lo, hi = sum(vs), min(vs), max(vs)
        A = chord_abs_sum(n, total, lo, hi, t)
        assert A >= sum(abs(v-t) for v in vs)
        sharp = integer_abs_sum(n, total, lo, hi, t)
        assert sum(abs(v-t) for v in vs) <= sharp <= A
        out = moment_residual(n, total, lo, hi, Interval(a, b), t)
        assert out.contains(sum(w*(v-t) for w, v in zip(ws, vs)))
        # Exact translation equivariance, including very large offsets.
        shift = F(rng.randrange(-10**9, 10**9), 3)
        shifted = moment_residual(n, total+n*shift, lo+shift, hi+shift,
                                  Interval(a, b), t+shift)
        assert out == shifted


def test_symmetric_endpoint_envelope_is_attained():
    a, b = F(3, 4), F(5, 4)
    vs = [F(63, 4)]*17+[F(65, 4)]*17
    for t in (F(63, 4), F(16), F(65, 4), F(-3), F(100)):
        out = moment_residual(len(vs), sum(vs), min(vs), max(vs), Interval(a, b), t)
        maximum = sum((b if v >= t else a)*(v-t) for v in vs)
        minimum = sum((a if v >= t else b)*(v-t) for v in vs)
        assert out == Interval(minimum, maximum)


def test_integer_envelope_against_exhaustive_fixed_sum_optimization():
    # Enumerate the feasible row multisets independently of the formula.
    # This grid contains the maximizing endpoint/remainder construction for
    # every enumerated moment, so both residual extremes must match exactly.
    grid = tuple(F(i, 2) for i in range(-2, 3))
    a, b = F(2, 3), F(7, 5)
    for n in range(1, 6):
        rows = tuple(combinations_with_replacement(grid, n))
        for t in (F(-2), F(-3, 4), F(-1, 8), F(0), F(1, 3), F(1), F(2)):
            extrema = {}
            abs_max = {}
            for vs in rows:
                total = sum(vs, F(0))
                minimum = sum((a if v >= t else b)*(v-t) for v in vs)
                maximum = sum((b if v >= t else a)*(v-t) for v in vs)
                old = extrema.get(total, (minimum, maximum))
                extrema[total] = (min(old[0], minimum), max(old[1], maximum))
                abs_max[total] = max(abs_max.get(total, F(0)), sum(abs(v-t) for v in vs))
            for total, (minimum, maximum) in extrema.items():
                sharp = integer_abs_sum(n, total, F(-1), F(1), t)
                assert sharp == abs_max[total]
                assert sharp <= chord_abs_sum(n, total, F(-1), F(1), t)
                assert moment_residual(n, total, F(-1), F(1), Interval(a, b), t) == Interval(minimum, maximum)
                shift = F(10**30, 7)
                assert integer_abs_sum(n, total+n*shift, F(-1)+shift, F(1)+shift, t+shift) == sharp


def test_integer_inputs_never_round_residual_bounds_through_float():
    endpoint = 2**53+1
    expected = 2*endpoint
    assert chord_abs_sum(2, 0, -endpoint, endpoint, 0) == expected
    assert isinstance(chord_abs_sum(2, 0, -endpoint, endpoint, 0), F)
    assert moment_residual(2, 0, -endpoint, endpoint, Interval(0, 2), 0) == Interval(-expected, expected)
    for n in (0, -1, F(3, 2), 2.0):
        with pytest.raises(ValueError):
            integer_abs_sum(n, 0, -1, 1, 0)


def test_integer_envelope_strictly_improves_interior_remainder():
    assert chord_abs_sum(3, 0, -1, 1, 0) == 3
    assert integer_abs_sum(3, 0, -1, 1, 0) == 2
    assert moment_residual(3, 0, -1, 1, Interval(0, 2), 0) == Interval(-2, 2)


@pytest.mark.parametrize("bits", [0, 1, 2, 127, 128, 129, 0x3f80, 0x3f81, 0x4180,
                                 0x7f7e, 0x7f7f, 0x8001, 0x807f, 0xbf80, 0xbf81, 0xff7f])
def test_rounding_cells(bits):
    cell = bf16_cell(bits)
    assert round_bf16_bits((cell.lo+cell.hi)/2) == bits
    assert round_bf16_bits(cell.value) == bits
    for edge in (cell.lo, cell.hi):
        try:
            rounds_here = round_bf16_bits(edge) == bits
        except OverflowError:
            rounds_here = False
        assert rounds_here == cell.closed
    for sign in (-1, 1):
        x = cell.lo if sign < 0 else cell.hi
        delta = (cell.hi-cell.lo)/1024
        assert round_bf16_bits(x-sign*delta) == bits


def test_every_finite_canonical_bf16_cell_center():
    for bits in range(65536):
        if bits == 0x8000 or (bits & 0x7f80) == 0x7f80:
            continue
        cell = bf16_cell(bits)
        assert cell.lo < cell.value < cell.hi
        assert round_bf16_bits(cell.value) == bits


def test_invalid_and_tie_gates():
    for bits in (0x8000, 0x7f80, 0x7fc1):
        with pytest.raises(ValueError):
            bf16_cell(bits)
    z = Interval.point(0)
    assert cell_accepts(z, z, bf16_cell(0x4180))
    assert not cell_accepts(z, z, bf16_cell(0x4181))
    with pytest.raises(ValueError):
        chord_abs_sum(2, F(100), F(0), F(1), F(0))
    with pytest.raises(ValueError):
        moment_residual(1, F(0), F(0), F(0), Interval(-1, 1), F(0))
