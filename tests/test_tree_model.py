from dataclasses import replace
from fractions import Fraction as F
import pytest
from statecut.model import small_model, Model, Layer
from statecut.tree_model import TreeModel


@pytest.mark.parametrize("strategy", ["cuts", "writes"])
@pytest.mark.parametrize("seed", [0, 1, 8, 19])
def test_complete_future_state_equality(strategy, seed):
    m = TreeModel(small_model(seed))
    dense, filtered = m.initial(4), m.initial(4)
    token = seed % 8
    for _ in range(24):
        a = m.step(dense, token, strategy="dense")
        b = m.step(filtered, token, strategy=strategy, max_expansions=3)
        assert a.token == b.token
        assert a.state == b.state  # Every exact KV entry, prefix, root, summary.
        dense, filtered, token = a.state, b.state, a.token


def state_frontier_fixture():
    """Non-singleton transient attention, but exact later persistent writes.

    This is a constructed frozen decoder, NOT a pretrained model. Layer 1
    has genuine variable K/V attention. Layer 2's K/V projections vanish.
    A positive terminal margin permits success without fixing layer 1's
    transient attention output. This separates write gates from activation gates.
    """
    d = 4
    z = tuple((F(0),)*d for _ in range(d))
    eye = tuple(tuple(F(i == j) for j in range(d)) for i in range(d))
    one = (F(1),)*d
    layer1 = Layer(eye, eye, eye, eye, z, z, z, one, one)
    layer2 = Layer(z, z, z, z, z, z, z, one, one)
    embedding = ((F(2), F(1), F(0), F(0)),
                 (F(2), F(-1), F(2), F(0)),
                 (F(2), F(2), F(-1), F(1)))
    head = ((F(1), F(0), F(0), F(0)), (F(0),)*d, (F(0),)*d)
    return TreeModel(Model(embedding, (layer1, layer2), one, head))


def test_write_frontier_strictly_weaker_than_exact_attention_cut():
    m = state_frontier_fixture()
    state = m.initial(8)
    for i in range(32):
        state = m.step(state, i % 3, strategy="dense").state
    out = m.step(state, 1, strategy="writes")
    dense = m.step(state, 1, strategy="dense")
    assert out.write_frontier_accepted
    assert out.uncertain_attention_cuts >= 1
    assert out.attempt_stats.raw_entries == 0
    assert out.state == dense.state
    assert out.token == dense.token
    # Continue from the certified state, including tokens absent from its draft.
    for i in range(12):
        a = m.step(out.state, (i+2) % 3, strategy="writes")
        b = m.step(dense.state, (i+2) % 3, strategy="dense")
        assert a.state == b.state and a.token == b.token
        out, dense = a, b


def test_foreign_state_failure_and_snapshot_no_mutation():
    m = TreeModel(small_model(1))
    state = m.initial(4)
    old = state
    with pytest.raises(ValueError):
        m.step(state, 100)
    assert state == old
    with pytest.raises(ValueError):
        TreeModel(small_model(2)).step(state, 1)
    with pytest.raises(ValueError):
        m.step(replace(state, position=1), 1)
    with pytest.raises(ValueError):
        m.step(state, 0, strategy="unsafe-draft")


@pytest.mark.parametrize("strategy", ["dense", "cuts", "writes"])
def test_v2_preserves_original_v1_dense_reference(strategy):
    old_model=small_model(17)
    model=TreeModel(old_model)
    old,new=old_model.initial(4),model.initial(4)
    for i in range(18):
        # Arbitrary supplied continuations, not just model-generated tokens.
        token=(i*3+1)%8
        a=old_model.step(old,token,certified=False)
        b=model.step(new,token,strategy=strategy,max_expansions=2)
        assert a.token == b.token
        assert a.state.position == b.state.position and a.state.tokens == b.state.tokens
        for oc,nc in zip(a.state.caches,b.state.caches):
            original_rows=tuple(e for block in oc.blocks for e in block._entries)
            forest_rows=tuple(e for leaf in nc.iter_leaves() for e in leaf._rows)
            assert original_rows == forest_rows
        old,new=a.state,b.state
