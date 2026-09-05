from dataclasses import replace
from fractions import Fraction as F
import pytest
from statecut.model import small_model,State
from statecut.arithmetic import Interval


@pytest.mark.parametrize("seed",range(8))
@pytest.mark.parametrize("terminal",[False,True])
def test_future_state_and_tokens_equal(seed,terminal):
    m=small_model(seed)
    dense,fast=m.initial(4),m.initial(4)
    token=seed%8
    for t in range(12):
        a=m.step(dense,token,certified=False)
        b=m.step(fast,token,certified=True,terminal_argmax=terminal,max_refinements=2)
        assert a.token==b.token
        assert a.state==b.state  # every historical and newly appended K/V
        dense,fast=a.state,b.state
        token=a.token


def test_foreign_model_and_position_rejected_without_mutation():
    m=small_model(0)
    s=m.initial()
    with pytest.raises(ValueError): small_model(1).step(s,0)
    with pytest.raises(ValueError): m.step(replace(s,position=1),0)
    assert s.position==0 and all(c.length==0 for c in s.caches)


def test_terminal_interval_suffix_contains_reference():
    m=small_model(17)
    h=m.embedding[0]
    for layer in m.layers:
        a=(F(1,2),F(-1,2),F(1,4),F(-1,4))
        bounds=tuple(Interval(x-F(1,1000),x+F(1,1000)) for x in a)
        box=layer.isuffix(h,bounds)
        y=layer.suffix(h,a)
        assert all(b.contains(x) for b,x in zip(box,y))


def test_same_token_different_state_counterexample():
    # current argmax identical, but a state perturbation changes next decision.
    def step(s):
        return (0 if s>=0 else 1), s-F(1,2)
    assert step(F(1,4))[0]==step(F(3,4))[0]
    assert step(step(F(1,4))[1])[0]!=step(step(F(3,4))[1])[0]


def test_interrupted_step_does_not_publish_partial_state(monkeypatch):
    from statecut.model import Layer
    m=small_model(3)
    s=m.initial()
    old_caches=s.caches
    def interrupt(*_):
        raise RuntimeError("injected after staging the first layer KV")
    monkeypatch.setattr(Layer,"suffix",interrupt)
    with pytest.raises(RuntimeError,match="injected"):
        m.step(s,0,certified=False)
    assert s.caches is old_caches
    assert s.position==0 and s.tokens==()
    assert all(c.length==0 and c.blocks==() for c in s.caches)
