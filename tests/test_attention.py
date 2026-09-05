from fractions import Fraction as F
from random import Random
import pytest
from statecut.arithmetic import Interval,bf16,certify_bf16
from statecut.cache import Cache,Entry,Summary,Block,ReadCounter
from statecut.attention import *


def make_cache(seed,n=33,d=4,block=8):
    r=Random(seed)
    c=Cache("test",block)
    for _ in range(n):
        c=c.append(Entry(tuple(F(r.randrange(-8,9),8) for _ in range(d)),
                         tuple(F(r.randrange(-32,33),8) for _ in range(d))))
    q=tuple(F(r.randrange(-8,9),8) for _ in range(d))
    return c,q


@pytest.mark.parametrize("seed",range(48))
def test_enclosure_and_exact_bf16(seed):
    cache,q=make_cache(seed,n=1+seed%33)
    cache.audit()
    exact=dense_attention(cache,q)
    cs=tuple(summary_contribution(q,b.summary) for b in cache.blocks)
    enc=enclosure(cs)
    assert enc is not None
    assert all(x.contains(y) for x,y in zip(enc,exact))
    result=verify_attention(cache,q,certify_bf16,lambda a:tuple(bf16(x) for x in a))
    assert result.value==tuple(bf16(x) for x in exact)
    assert result.stats.raw_entries<=cache.length
    assert len(result.opened_blocks)==len(set(result.opened_blocks))


def test_zero_raw_reads_for_constant_keys():
    c=Cache("identical-keys",64)
    for j in range(4096):
        c=c.append(Entry((F(1),F(-1)),(F(j%7-3),F(j%3-1))))
    out=verify_attention(c,(F(1),F(2)),certify_bf16,lambda a:tuple(bf16(x) for x in a))
    assert out.stats.raw_entries==0
    assert out.value==tuple(bf16(x) for x in dense_attention(c,(F(1),F(2))))


def test_negative_values_need_reversed_weight_bounds():
    c=Cache("signed",8)
    c=c.append(Entry((F(-1),),(F(-100),)))
    c=c.append(Entry((F(1),),(F(-1),)))
    s=summary_contribution((F(1),),c.blocks[0].summary)
    exact=exact_contribution((F(1),),c.blocks[0]._entries)
    assert s.numerator[0].contains(exact.numerator[0].lo)


def test_zero_refinement_budget_fails_closed():
    c,q=make_cache(81)
    result=verify_attention(c,q,lambda _:None,lambda a:tuple(bf16(x) for x in a),0)
    assert result.used_dense_fallback
    assert result.stats.raw_entries==c.length
    assert result.value==tuple(bf16(x) for x in dense_attention(c,q))


def test_append_and_tamper_and_shape_guards():
    c,q=make_cache(12)
    before=c
    nxt=c.append(Entry((F(0),)*4,(F(0),)*4))
    assert before.length+1==nxt.length
    before.audit(); nxt.audit()
    with pytest.raises(ValueError): c.append(Entry((F(0),),(F(0),)))
    with pytest.raises(ValueError): dense_attention(c,(F(0),))
    b=c.blocks[0]
    s=b.summary
    wrong=Summary(s.n,s.kmin,s.kmax,tuple(x+1 for x in s.positive),s.negative,s.vmin,s.vmax)
    corrupt=Cache(c.identity,c.block_size,(Block(wrong,b._entries),)+c.blocks[1:],c.length)
    with pytest.raises(ValueError): corrupt.audit()


def test_all_zero_reference_weights_rejected():
    c=Cache("zero",8).append(Entry((F(-100),),(F(1),)))
    with pytest.raises(ArithmeticError):
        verify_attention(c,(F(1),),certify_bf16,lambda a:a)


def test_nonconstant_keys_open_only_dominant_block():
    """A non-degenerate example: every block has varying keys; only one is read."""
    rng=Random(10)
    c=Cache("dominant",32)
    for i in range(512):
        if i<480:
            k=(F(-14)+F(rng.randrange(-4,5),32),F(0),F(0),F(0))
            v=tuple(F(rng.randrange(-8,9),16) for _ in range(4))
        else:
            k=(F(rng.randrange(-4,5),8),F(0),F(0),F(0))
            v=(F(1,2),)*4
        c=c.append(Entry(k,v))
    q=(F(1),F(0),F(0),F(0))
    out=verify_attention(c,q,certify_bf16,lambda a:tuple(bf16(x) for x in a),4)
    assert out.value==tuple(bf16(x) for x in dense_attention(c,q))
    assert out.opened_blocks==(15,)
    assert out.stats.raw_entries==32
    assert not out.used_dense_fallback


def test_malformed_shapes_rejected_not_silently_zipped():
    a=Entry((F(0),),(F(1),))
    b=Entry((F(0),),(F(1),F(2)))
    bad=Cache("shape",1,(Block(Summary.single(a),(a,)),Block(Summary.single(b),(b,))),2)
    with pytest.raises(ValueError): bad.audit()
    with pytest.raises(ValueError): exact_contribution((F(0),),(a,b))
    with pytest.raises(ValueError):
        enclosure((Contribution(Interval.point(1),(Interval.point(0),)),
                   Contribution(Interval.point(1),(Interval.point(0),Interval.point(1)))))


def test_summary_invalid_signs_are_rejected():
    a=Summary(1,(F(0),),(F(0),),(F(-1),),(F(0),),(F(-1),),(F(-1),))
    with pytest.raises(ValueError): summary_contribution((F(0),),a)


def test_flat_summary_resource_failure_falls_back_to_valid_correlated_scores():
    cache = Cache("correlated", 2)
    for k, v in (((2048, -2048), 1), ((-2048, 2048), 3)):
        cache = cache.append(Entry(k, (v,)))
    query = (F(1), F(1))
    report = verify_attention(cache, query, certify_bf16, lambda a: tuple(map(bf16, a)))
    assert report.value == tuple(map(bf16, dense_attention(cache, query))) == (F(2),)
    assert report.used_dense_fallback
    assert report.stats.raw_entries == 2
    assert report.opened_blocks == (0,)
