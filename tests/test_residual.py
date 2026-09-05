from fractions import Fraction as F
from random import Random
import pytest
from statecut.arithmetic import Interval, bf16, bf16_value, round_bf16_bits
from statecut.residual import chord_abs_sum, moment_residual, bf16_cell, cell_accepts


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
