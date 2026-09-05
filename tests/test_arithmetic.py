from fractions import Fraction as F
from random import Random
import mpmath as mp
import pytest
from statecut.arithmetic import *


def test_reversed_and_zero_divisor():
    with pytest.raises(ValueError): Interval(1,0)
    with pytest.raises(ZeroDivisionError): Interval.point(1)/Interval(-1,1)


def test_interval_corners_random():
    r=Random(11)
    for _ in range(300):
        a,b=sorted(F(r.randrange(-80,81),8) for _ in range(2))
        c,d=sorted(F(r.randrange(-80,81),8) for _ in range(2))
        x,y=Interval(a,b),Interval(c,d)
        for u in (a,(a+b)/2,b):
            for v in (c,(c+d)/2,d):
                assert (x+y).contains(u+v)
                assert (x*y).contains(u*v)
                if not c<=0<=d: assert (x/y).contains(u/v)
        for u in (a,(a+b)/2,b): assert x.square().contains(u*u)


def test_exp_enclosures_independent_high_precision():
    mp.mp.dps=150
    vals=[F(0),F(1,2),F(-1,2),F(20),F(-20),F(100),F(-100)]
    vals += [F(x,13) for x in range(-100,101,7)]
    for x in vals:
        b=exp_enclosure(x,96)
        truth=mp.exp(mp.mpf(x.numerator)/x.denominator)
        lo=mp.mpf(b.lo.numerator)/b.lo.denominator
        hi=mp.mpf(b.hi.numerator)/b.hi.denominator
        assert lo <= truth <= hi
    # This is a numerical cross-check, not the proof of exp enclosure.


def test_exp_reference_rounding_and_monotonicity():
    vals=[F(x,16) for x in range(-64,65)]
    ys=[exp_reference(x) for x in vals]
    assert ys==sorted(ys)
    assert exp_reference(F(0))==1
    for x,y in zip(vals,ys):
        b=exp_enclosure(x,128)
        assert b.lo-F(1,2**25)<=y<=b.hi+F(1,2**25)


def test_bf16_ties_subnormals_and_overflow():
    a,b=bf16_value(0x3f80),bf16_value(0x3f81)
    assert round_bf16_bits((a+b)/2)==0x3f80
    c=bf16_value(0x3f82)
    assert round_bf16_bits((b+c)/2)==0x3f82
    assert round_bf16_bits(F(1,2**134))==0
    assert round_bf16_bits(F(3,2**134))==2
    assert round_bf16_bits(-F(1,2**134))==0
    assert certify_bf16((Interval(a,(a+b)/2),))==(a,)
    assert certify_bf16((Interval(a,b),)) is None
    with pytest.raises(OverflowError): bf16(F(2)**128)
    with pytest.raises(ValueError): bf16_value(0x7fc0)


def test_bf16_every_finite_encoding_roundtrip():
    for code in range(65536):
        if ((code>>7)&255)==255: continue
        actual=round_bf16_bits(bf16_value(code))
        assert actual==(0 if code==0x8000 else code)


def test_exact_rsqrt_rounding_at_midpoint():
    for code in [0,1,2,127,128,0x3f7f,0x3f80,0x3f81,0x4000]:
        a,b=bf16_value(code),bf16_value(code+1)
        midpoint=(a+b)/2
        # sqrt(1/9) is not dyadic. Interval iteration alone can stall here.
        expected=bf16_value(code if code%2==0 else code+1)
        assert bf16_rsqrt(midpoint/3,F(1,9))==expected
        assert bf16_rsqrt(-midpoint/3,F(1,9))==-expected


def test_tie_aware_argmax():
    assert certify_argmax((Interval.point(2),Interval.point(2)))==0
    assert certify_argmax((Interval(1,2),Interval.point(2))) is None
    assert certify_argmax((Interval.point(1),Interval(2,3)))==1
    assert greedy((F(2),F(2)))==0


def test_direct_bf16_rounding_must_not_double_round_via_fp32():
    """Exact x is above a BF16 midpoint but binary32 rounding loses that fact."""
    import struct
    x=F(1)+F(1,256)+F(1,1<<30)
    via_fp32=F(struct.unpack("f",struct.pack("f",float(x)))[0])
    assert bf16(x)==F(1)+F(1,128)
    assert bf16(via_fp32)==F(1)


def test_rsqrt_normalizes_float_radicand_before_midpoint_comparison():
    radicand = 0.1
    lower, upper = bf16_value(0x3f80), bf16_value(0x3f81)
    midpoint = (lower+upper)/2
    # The numerator is strictly above the true midpoint*sqrt(radicand).
    # Computing midpoint**2*radicand in float moves that boundary enough to
    # reverse this exact decision, even though the radicand bits are fixed.
    numerator = midpoint*sqrt_enclosure(F(radicand), 256).hi
    assert numerator*numerator > midpoint*midpoint*F(radicand)
    assert bf16_rsqrt(numerator, radicand) == upper
    assert bf16_rsqrt(-numerator, radicand) == -upper


def test_large_positive_e24_uses_enough_absolute_precision():
    # 2048 needs nearly 3000 integer bits before the fixed fractional grid.
    # The original fixed maximum of 1024 working bits could not resolve even
    # exp(1024), despite accepting that score as inside its resource domain.
    scores = (F(1024), F(167805713, 131072), F(2048), F(-2048))
    with mp.workdps(1400):
        for x in scores:
            truth = mp.exp(mp.mpf(x.numerator)/x.denominator)
            for fractional_bits in (1, 24, 256):
                expected = F(int(mp.nint(truth*(1 << fractional_bits))), 1 << fractional_bits)
                assert exp_reference(x, fractional_bits) == expected
    # The independent numerical cross-check does not replace the enclosure
    # proof; unsupported values still raise instead of inventing a weight.
    for x in (F(4097, 2), F(-4097, 2)):
        with pytest.raises(OverflowError, match="resource guard"):
            exp_reference(x)
