from fractions import Fraction as F
from random import Random
import pytest
from statecut.arithmetic import Interval, bf16, certify_bf16
from statecut.cache import Cache, Entry
from statecut.attention import dense_attention
from statecut.forest import ForestCache, TreeReads
from statecut.tree_attention import verify_tree_attention, root_enclosure


def build(rows, block_size=8):
    f, c = ForestCache("f", block_size), Cache("c", block_size)
    for row in rows:
        f, c = f.append(row), c.append(row)
    return f, c


def test_nonconstant_keys_and_values_single_root_no_raw_reads():
    rows = [Entry((F((-1)**i, 16),), (F(16)+F((-1)**(i//2), 4),)) for i in range(2048)]
    tree, dense = build(rows, 32)
    report = verify_tree_attention(tree, (F(1),), certify_bf16,
                                   lambda a: tuple(map(bf16, a)), direct_bf16=True)
    assert report.value == tuple(map(bf16, dense_attention(dense, (F(1),)))) == (F(16),)
    assert report.stats.raw_entries == 0
    assert report.stats.summary_nodes == 1
    assert report.gate == "direct-boundary"


@pytest.mark.parametrize("budget", [0, 1, 2, 9, None])
def test_randomized_refinement_never_disagrees(budget):
    rng = Random(817)
    for case in range(25):
        n = rng.randrange(1, 60)
        rows = [Entry(tuple(F(rng.randrange(-20, 21), 16) for _ in range(3)),
                      tuple(F(rng.randrange(-50, 51), 32) for _ in range(4))) for _ in range(n)]
        tree, dense = build(rows, 5)
        q = tuple(F(rng.randrange(-10, 11), 8) for _ in range(3))
        exact = dense_attention(dense, q)
        report = verify_tree_attention(tree, q, certify_bf16, lambda a: tuple(map(bf16, a)),
                                       max_expansions=budget, direct_bf16=True)
        assert report.value == tuple(map(bf16, exact))
        assert all(box.contains(v) for box, v in zip(report.attention_bounds, exact))
        assert report.stats.raw_entries <= n


def test_forced_fallback_reuses_raw_leaves():
    rows = [Entry((F(i, 11),), (F((-1)**i, 7),)) for i in range(45)]
    tree, dense = build(rows, 4)
    for budget in (0, 3, 8, None):
        report = verify_tree_attention(tree, (F(1),), lambda box: None, lambda a: a,
                                       max_expansions=budget)
        assert report.used_dense_fallback
        assert report.value == dense_attention(dense, (F(1),))
        assert report.stats.raw_entries == len(rows)
        assert report.stats.opened_leaves == len(list(tree.iter_leaves()))


def test_interval_queries_enclose_all_corners():
    rows = [Entry((F(i, 10), F((-1)**i)), (F(i%5), F(-i%7))) for i in range(17)]
    tree, dense = build(rows, 4)
    qbox = (Interval(F(-1, 2), F(1, 2)), Interval(F(1, 4), F(3, 4)))
    enc = root_enclosure(tree, qbox)
    assert enc is not None
    for x in (qbox[0].lo, F(0), qbox[0].hi):
        for y in (qbox[1].lo, F(1, 2), qbox[1].hi):
            exact = dense_attention(dense, (x, y))
            assert all(b.contains(v) for b, v in zip(enc, exact))


def test_zero_weight_denominator_and_no_custom_direct_gate():
    tree, _ = build([Entry((F(-100),), (F(1),))])
    with pytest.raises(ArithmeticError):
        verify_tree_attention(tree, (F(1),), certify_bf16, lambda a: a, max_expansions=0)
    with pytest.raises(ValueError):
        verify_tree_attention(tree, (F(1),), lambda x: 1, lambda a: 1, direct_bf16=True)


def test_direct_boundary_retains_more_information_than_divided_interval():
    # Slightly wider score box makes the residual/D enclosure cross a cell.
    # Boundary residual tests still prove the cut without any raw reads.
    rows = [Entry((F((-1)**i,8),), (F(16)+F((-1)**(i//2),4),)) for i in range(128)]
    tree, dense = build(rows,32)
    direct = verify_tree_attention(tree,(F(1),),certify_bf16,lambda a:tuple(map(bf16,a)),
                                   max_expansions=0,direct_bf16=True)
    divided = verify_tree_attention(tree,(F(1),),certify_bf16,lambda a:tuple(map(bf16,a)),
                                    max_expansions=0,direct_bf16=False)
    assert direct.value == divided.value == (F(16),)
    assert direct.stats.raw_entries == 0 and direct.gate == "direct-boundary"
    assert certify_bf16(direct.attention_bounds) is None
    assert divided.used_dense_fallback and divided.stats.raw_entries == len(rows)


def test_summary_resource_failure_falls_back_when_actual_scores_are_valid():
    # The box allows scores +/-2048 (outside the oracle guard); each actual
    # correlated row has score zero. A failed bound must not reject this target.
    rows=[Entry((F(1024),F(-1024)),(F(1),)),
          Entry((F(-1024),F(1024)),(F(3),))]
    tree,dense=build(rows,2)
    r=verify_tree_attention(tree,(F(1),F(1)),certify_bf16,
                            lambda a:tuple(map(bf16,a)),direct_bf16=True)
    assert r.value == (F(2),)
    assert r.used_dense_fallback and r.stats.raw_entries == 2
    assert r.gate == "dense-fallback:summary-resource"


def test_integer_count_gate_accepts_where_continuous_chord_rejects():
    from statecut.residual import (bf16_cell, cell_accepts, chord_abs_sum,
                                   weight_box, summary_residual, intersect)
    rows = [Entry((F(-3, 20),), (F(63, 4),)),
            Entry((F(0),), (F(16),)),
            Entry((F(3, 20),), (F(65, 4),))]
    tree, dense = build(rows, 3)
    s = tree.roots[0].summary
    w = weight_box((Interval.point(1),), s)
    cell = bf16_cell(0x4180)
    def old_residual(t):
        total = s.positive[0]+s.negative[0]
        center = (w.lo+w.hi)/2*(total-s.n*t)
        radius = w.width/2*chord_abs_sum(s.n, total, s.vmin[0], s.vmax[0], t)
        old = Interval(center-radius, center+radius)
        # Include the original signed-moment intersection in the comparison.
        numerator = Interval(w.lo*s.positive[0]+w.hi*s.negative[0],
                             w.hi*s.positive[0]+w.lo*s.negative[0])
        return intersect(old, numerator-w.scale(s.n*t))
    assert not cell_accepts(old_residual(cell.lo), old_residual(cell.hi), cell)
    assert cell_accepts(summary_residual(s, w, 0, cell.lo), summary_residual(s, w, 0, cell.hi), cell)
    report = verify_tree_attention(tree, (F(1),), certify_bf16,
                                   lambda a: tuple(map(bf16, a)), max_expansions=0,
                                   direct_bf16=True)
    assert report.value == tuple(map(bf16, dense_attention(dense, (F(1),)))) == (F(16),)
    assert report.gate == "direct-boundary" and report.stats.raw_entries == 0


def test_float_queries_use_the_same_exact_reduction_in_dense_and_filtered_paths():
    from statecut.attention import score, verify_attention
    # All query coordinates are exactly representable. A floating reduction
    # would lose the middle 1 between +2**54 and -2**54, changing E24 weights.
    rows = [Entry((0, 0, 0), (0,)), Entry((0, 0, 0), (0,)),
            Entry((2**54, 1, -2**54), (1,))]
    tree, dense = build(rows, 1)
    query = (1.0, 1.0, 1.0)
    exact_query = tuple(map(F, query))
    assert score(query, rows[-1].k) == F(1)
    exact = dense_attention(dense, exact_query)
    assert dense_attention(dense, query) == exact
    for direct in (False, True):
        report = verify_tree_attention(tree, query, certify_bf16,
                                       lambda a: tuple(map(bf16, a)),
                                       max_expansions=0, direct_bf16=direct)
        assert report.value == tuple(map(bf16, exact)) == (F(147, 256),)
        assert report.stats.raw_entries == 0
    old = verify_attention(dense, query, certify_bf16, lambda a: tuple(map(bf16, a)))
    assert old.value == tuple(map(bf16, exact)) and old.stats.raw_entries == 0
