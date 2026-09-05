"""Frozen small pre-norm causal decoder for exact-state integration testing.

This is NOT a pretrained model adapter. It has two attention/FFN layers,
RMSNorm and a SiLU-like gate defined by the explicit exp reference. All
weights are frozen dyadic rationals and all materialization cuts are BF16.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as F
from hashlib import sha256
from random import Random
from .arithmetic import (Interval,bf16,bf16_rsqrt,exp_reference,certify_bf16,
                         certify_argmax,rounded_interval,sqrt_interval,isum,greedy)
from .cache import Cache,Entry,ReadCounter
from .attention import dense_attention,verify_attention,Verified

Vec = tuple[F,...]
Mat = tuple[Vec,...]
IVec = tuple[Interval,...]
EPS = F(1,256)


def linear(w: Mat,x: Vec) -> Vec:
    if any(len(row)!=len(x) for row in w):
        raise ValueError("linear shape mismatch")
    return tuple(bf16(sum((a*b for a,b in zip(row,x)),F(0))) for row in w)


def ilinear(w: Mat,x: IVec) -> IVec:
    if any(len(row)!=len(x) for row in w):
        raise ValueError("linear shape mismatch")
    return tuple(rounded_interval(isum(b.scale(a) for a,b in zip(row,x))) for row in w)


def add(x: Vec,y: Vec) -> Vec:
    if len(x)!=len(y):
        raise ValueError("residual shape mismatch")
    return tuple(bf16(a+b) for a,b in zip(x,y))


def iadd(x: IVec,y: IVec) -> IVec:
    if len(x)!=len(y):
        raise ValueError("residual shape mismatch")
    return tuple(rounded_interval(a+b) for a,b in zip(x,y))


def norm(x: Vec,gamma: Vec) -> Vec:
    if not x or len(x)!=len(gamma):
        raise ValueError("norm shape mismatch")
    r = sum((a*a for a in x),F(0))/len(x)+EPS
    return tuple(bf16_rsqrt(a*g,r) for a,g in zip(x,gamma))


def inorm(x: IVec,gamma: Vec) -> IVec:
    if not x or len(x)!=len(gamma):
        raise ValueError("norm shape mismatch")
    r = isum(a.square() for a in x).scale(F(1,len(x)))+Interval.point(EPS)
    root = sqrt_interval(r,128)
    return tuple(rounded_interval(a.scale(g)/root) for a,g in zip(x,gamma))


def silu(x: F) -> F:
    return bf16(x/(1+exp_reference(-x)))


def isilu(x: Interval) -> Interval:
    # SiLU is NOT monotone on R. Interval multiplication is deliberate.
    sigmoid = Interval(F(1)/(1+exp_reference(-x.lo)), F(1)/(1+exp_reference(-x.hi)))
    return rounded_interval(x*sigmoid)


@dataclass(frozen=True)
class Layer:
    q: Mat
    k: Mat
    v: Mat
    o: Mat
    gate: Mat
    up: Mat
    down: Mat
    norm1: Vec
    norm2: Vec

    def suffix(self,h: Vec,a: Vec) -> Vec:
        z = add(h,linear(self.o,a))
        r = norm(z,self.norm2)
        g,u = linear(self.gate,r),linear(self.up,r)
        f = tuple(bf16(silu(x)*y) for x,y in zip(g,u))
        return add(z,linear(self.down,f))

    def isuffix(self,h: Vec,a: IVec) -> IVec:
        z = iadd(tuple(Interval.point(x) for x in h),ilinear(self.o,a))
        r = inorm(z,self.norm2)
        g,u = ilinear(self.gate,r),ilinear(self.up,r)
        f = tuple(rounded_interval(isilu(x)*y) for x,y in zip(g,u))
        return iadd(z,ilinear(self.down,f))


@dataclass(frozen=True)
class State:
    model_id: str
    caches: tuple[Cache,...]
    position: int = 0
    tokens: tuple[int,...] = ()


@dataclass(frozen=True)
class Step:
    token: int
    state: State
    certificates: tuple[Verified,...]
    dense_reads: int


@dataclass(frozen=True)
class Model:
    embedding: Mat
    layers: tuple[Layer,...]
    final_norm: Vec
    head: Mat
    query_scale: F = F(1,2)

    @property
    def identity(self) -> str:
        return sha256(("RATIONAL_BF16_V1|"+repr(self)).encode()).hexdigest()

    def initial(self,block_size: int = 8) -> State:
        ident = self.identity
        return State(ident,tuple(Cache(f"{ident}:layer:{i}",block_size) for i in range(len(self.layers))))

    def _check(self,state: State,token: int) -> None:
        if not self.layers:
            raise ValueError("at least one layer required")
        if not 0 <= token < len(self.embedding):
            raise ValueError("token out of range")
        if state.model_id != self.identity or len(state.caches)!=len(self.layers):
            raise ValueError("stale or foreign model/cache")
        if state.position != len(state.tokens):
            raise ValueError("position/token-prefix mismatch")
        for i,c in enumerate(state.caches):
            if c.identity != f"{self.identity}:layer:{i}" or c.length != state.position:
                raise ValueError("cache identity, layer, or visible-prefix mismatch")

    def logits(self,h: Vec) -> Vec:
        return linear(self.head,norm(h,self.final_norm))

    def ilogits(self,h: IVec) -> IVec:
        return ilinear(self.head,inorm(h,self.final_norm))

    def step(self,state: State,token: int, *, certified: bool = True,
             terminal_argmax: bool = True,max_refinements: int | None = None) -> Step:
        self._check(state,token)
        # Persistent immutable snapshots: an exception cannot mutate state.
        h = self.embedding[token]
        staged: list[Cache] = []
        reports: list[Verified] = []
        dense_reads = 0
        next_token: int | None = None
        for i,layer in enumerate(self.layers):
            r = norm(h,layer.norm1)
            q = tuple(x*self.query_scale for x in linear(layer.q,r))
            k,v = linear(layer.k,r),linear(layer.v,r)
            current = state.caches[i].append(Entry(k,v))
            staged.append(current)  # local only until the entire step succeeds
            final = i == len(self.layers)-1
            if certified and final and terminal_argmax:
                def consume(bounds: tuple[Interval,...]) -> int | None:
                    try:
                        aa = tuple(rounded_interval(x) for x in bounds)
                        return certify_argmax(self.ilogits(layer.isuffix(h,aa)))
                    except (OverflowError,ArithmeticError):
                        return None
                def exact_consume(a: Vec) -> int:
                    return greedy(self.logits(layer.suffix(h,tuple(bf16(x) for x in a))))
                report = verify_attention(current,q,consume,exact_consume,max_refinements)
                reports.append(report)
                next_token = int(report.value)
            else:
                if certified:
                    report = verify_attention(current,q,certify_bf16,
                                              lambda a: tuple(bf16(x) for x in a),max_refinements)
                    reports.append(report)
                    a = report.value
                    if not isinstance(a,tuple):
                        raise TypeError("invalid attention consumer result")
                else:
                    counter = ReadCounter()
                    a = tuple(bf16(x) for x in dense_attention(current,q,counter))
                    dense_reads += counter.raw_entries
                h = layer.suffix(h,a)
                if final:
                    next_token = greedy(self.logits(h))
        assert next_token is not None
        committed = State(state.model_id,tuple(staged),state.position+1,state.tokens+(token,))
        return Step(next_token,committed,tuple(reports),dense_reads)


def small_model(seed: int = 0) -> Model:
    """A deterministic frozen test fixture, not a pretrained checkpoint."""
    rng = Random(seed)
    d,ff,vocab = 4,6,8
    def matrix(n: int,m: int,denom: int = 32) -> Mat:
        return tuple(tuple(bf16(F(rng.randrange(-6,7),denom)) for _ in range(m)) for _ in range(n))
    one = (F(1),)*d
    layers = tuple(Layer(matrix(d,d),matrix(d,d),matrix(d,d),matrix(d,d),
                         matrix(ff,d),matrix(ff,d),matrix(d,ff),one,one) for _ in range(2))
    return Model(matrix(vocab,d,8),layers,one,matrix(vocab,d,8))
