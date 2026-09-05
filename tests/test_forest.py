from dataclasses import replace
from fractions import Fraction as F
import pytest
from statecut.cache import Entry
from statecut.forest import ForestCache


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
