from dataclasses import replace
from fractions import Fraction as F
import pytest
from statecut.cache import Cache, Entry, Summary
from statecut.forest import ForestCache, Node, parent


def test_append_all_prefixes_and_immutable_snapshots():
    cache = ForestCache("x", 7)
    snapshots = []
    for i in range(500):
        snapshots.append(cache)
        cache = cache.append(Entry((F(i, 7),), (F((-1)**i*i, 3),)))
        sealed = cache.length//7
        assert len(cache.roots) == sealed.bit_count()
        assert len(cache.frontier) <= sealed.bit_length()+1
        cache.audit()
    for i in (0, 1, 6, 7, 27, 63, 199, 499):
        assert snapshots[i].length == i
        assert sum(len(n._rows) for n in snapshots[i].iter_leaves()) == i
        snapshots[i].audit()


def test_foreign_shapes_and_corrupt_provenance():
    c = ForestCache("x", 2).append(Entry((F(1),), (F(2),)))
    with pytest.raises(ValueError):
        c.append(Entry((F(1), F(2)), (F(2),)))
    with pytest.raises(ValueError):
        replace(c, length=10).audit()
    assert c.tail is not None
    bad = replace(c.tail, start=2)
    with pytest.raises(ValueError):
        replace(c, tail=bad).audit()


@pytest.mark.parametrize("cache_type", [Cache, ForestCache])
@pytest.mark.parametrize("block_size", [0, -1, F(3, 2), 2.0])
def test_block_size_must_be_a_positive_integer(cache_type, block_size):
    with pytest.raises(ValueError):
        cache_type("invalid-size", block_size)


def test_forest_audit_checks_dimensions_across_roots_and_tail():
    c = ForestCache("shape", 2)
    for value in (1, 2):
        c = c.append(Entry((0,), (value,)))
    different = Entry((0,), (1, 2))
    tail = Node(2, Summary.single(different), _rows=(different,))
    with pytest.raises(ValueError, match="dimensions"):
        replace(c, tail=tail, length=3).audit()


def test_forest_audit_rejects_wrong_sealed_leaf_sizes():
    e = Entry((0,), (1,))
    s = Summary.single(e).append(e)
    oversized = Node(0, s, _rows=(e, e))
    with pytest.raises(ValueError, match="sealed leaf"):
        ForestCache("oversized", 1, (oversized,), length=2).audit()
    undersized = parent(Node(0, s, _rows=(e, e)), Node(2, s, _rows=(e, e)))
    with pytest.raises(ValueError, match="sealed leaf"):
        ForestCache("undersized", 4, (undersized,), length=4).audit()
