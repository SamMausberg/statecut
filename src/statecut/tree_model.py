"""Whole-transition integration for the frozen reference decoder.

Two sound strategies are implemented:
  cuts: require exact attention cuts, except for the state-free final suffix;
  writes: propagate intervals and require only persistent KV writes to be
          singleton. A failed attempt is discarded and the dense step rerun.
Neither strategy is an adapter for PyTorch/FlashAttention arithmetic.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction as F
from .arithmetic import (Interval, bf16, certify_bf16, certify_argmax,
                         rounded_interval, greedy)
from .cache import Entry
from .forest import ForestCache, TreeReads
from .tree_attention import (verify_tree_attention, dense_tree_attention,
                             root_enclosure, TreeVerified)
from .model import (Model, Vec, IVec, Mat, Layer, norm, inorm, linear,
                    ilinear, iadd, isilu)


def points(v: Vec) -> IVec:
    return tuple(Interval.point(x) for x in v)


def singleton(v: IVec) -> Vec | None:
    return tuple(x.lo for x in v) if all(x.lo == x.hi for x in v) else None


def norm_box(x: IVec, gamma: Vec) -> IVec:
    known = singleton(x)
    return points(norm(known, gamma)) if known is not None else inorm(x, gamma)


def linear_box(w: Mat, x: IVec) -> IVec:
    known = singleton(x)
    return points(linear(w, known)) if known is not None else ilinear(w, x)


def suffix_box(layer: Layer, h: IVec, a: IVec) -> IVec:
    known_h, known_a = singleton(h), singleton(a)
    if known_h is not None and known_a is not None:
        return points(layer.suffix(known_h, known_a))
    z = iadd(h, linear_box(layer.o, a))
    r = norm_box(z, layer.norm2)
    gate, up = linear_box(layer.gate, r), linear_box(layer.up, r)
    f = tuple(rounded_interval(isilu(x)*y) for x, y in zip(gate, up))
    return iadd(z, linear_box(layer.down, f))


@dataclass(frozen=True, slots=True)
class TreeState:
    model_id: str
    caches: tuple[ForestCache, ...]
    position: int = 0
    tokens: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class TreeStep:
    token: int
    state: TreeState
    certificates: tuple[TreeVerified, ...] = ()
    dense_reads: int = 0
    write_frontier_accepted: bool = False
    uncertain_attention_cuts: int = 0
    attempt_stats: TreeReads | None = None
    fallback_reason: str | None = None


@dataclass(frozen=True)
class TreeModel:
    model: Model
    _model_id: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Fingerprint once at construction, never scan model weights per token.
        object.__setattr__(self, "_model_id", self.model.identity)

    def initial(self, block_size: int = 32) -> TreeState:
        return TreeState(self._model_id,
                         tuple(ForestCache(f"{self._model_id}:layer:{i}", block_size)
                               for i in range(len(self.model.layers))))

    def _check(self, state: TreeState, token: int) -> None:
        if not self.model.layers or not 0 <= token < len(self.model.embedding):
            raise ValueError("invalid model or token")
        if state.model_id != self._model_id or len(state.caches) != len(self.model.layers):
            raise ValueError("foreign state")
        if state.position != len(state.tokens):
            raise ValueError("prefix length mismatch")
        for i, cache in enumerate(state.caches):
            if cache.length != state.position or cache.identity != f"{self._model_id}:layer:{i}":
                raise ValueError("foreign layer or stale prefix")

    def step(self, state: TreeState, token: int, *, strategy: str = "cuts",
             max_expansions: int | None = 64) -> TreeStep:
        self._check(state, token)
        if strategy not in ("cuts", "writes", "dense"):
            raise ValueError("strategy must be cuts, writes or dense")
        if strategy == "writes":
            return self._writes(state, token)
        h = self.model.embedding[token]
        staged = []
        reports = []
        dense_reads = 0
        output = None
        for i, layer in enumerate(self.model.layers):
            r = norm(h, layer.norm1)
            q = tuple(x*self.model.query_scale for x in linear(layer.q, r))
            k, v = linear(layer.k, r), linear(layer.v, r)
            cache = state.caches[i].append(Entry(k, v))
            staged.append(cache)
            final = i == len(self.model.layers)-1
            if strategy == "cuts" and final:
                def consume(bounds: IVec):
                    try:
                        aa = tuple(rounded_interval(x) for x in bounds)
                        return certify_argmax(self.model.ilogits(layer.isuffix(h, aa)))
                    except (ArithmeticError, OverflowError):
                        return None
                def exact_consume(a: Vec):
                    return greedy(self.model.logits(layer.suffix(h, tuple(map(bf16, a)))))
                report = verify_tree_attention(cache, q, consume, exact_consume,
                                               max_expansions=max_expansions)
                reports.append(report)
                output = int(report.value)
            else:
                if strategy == "dense":
                    reads = TreeReads()
                    a = tuple(map(bf16, dense_tree_attention(cache, q, reads)))
                    dense_reads += reads.raw_entries
                else:
                    report = verify_tree_attention(cache, q, certify_bf16,
                                                   lambda a: tuple(map(bf16, a)),
                                                   max_expansions=max_expansions,
                                                   direct_bf16=True)
                    reports.append(report)
                    a = report.value
                h = layer.suffix(h, a)
                if final:
                    output = greedy(self.model.logits(h))
        assert output is not None
        committed = TreeState(state.model_id, tuple(staged), state.position+1, state.tokens+(token,))
        return TreeStep(output, committed, tuple(reports), dense_reads)

    def _writes(self, state: TreeState, token: int) -> TreeStep:
        """Certificate at actual write boundaries, not every activation.

        No raw cache reads on a successful attempt. On failure ALL temporary
        writes are discarded. Fallback starts from the original snapshot.
        Recomputed current-token projections/summary maintenance are real costs.
        """
        h = points(self.model.embedding[token])
        staged = []
        reads = TreeReads()
        uncertain = 0
        reason = None
        try:
            for i, layer in enumerate(self.model.layers):
                r = norm_box(h, layer.norm1)
                q = tuple(x.scale(self.model.query_scale) for x in linear_box(layer.q, r))
                kb, vb = linear_box(layer.k, r), linear_box(layer.v, r)
                k, v = singleton(kb), singleton(vb)
                if k is None or v is None:
                    reason = f"layer-{i}:non-singleton-KV-write"
                    break
                cache = state.caches[i].append(Entry(k, v))
                staged.append(cache)
                ab = root_enclosure(cache, q, reads)
                if ab is None:
                    reason = f"layer-{i}:nonpositive-denominator-bound"
                    break
                aa = tuple(rounded_interval(x) for x in ab)
                if singleton(aa) is None:
                    uncertain += 1
                h = suffix_box(layer, h, aa)
            if reason is None:
                logits = linear_box(self.model.head, norm_box(h, self.model.final_norm))
                output = certify_argmax(logits)
                if output is not None:
                    committed = TreeState(state.model_id, tuple(staged), state.position+1,
                                          state.tokens+(token,))
                    return TreeStep(output, committed, write_frontier_accepted=True,
                                    uncertain_attention_cuts=uncertain, attempt_stats=reads)
                reason = "terminal-argmax-ambiguous"
        except (ArithmeticError, OverflowError) as exc:
            reason = "arithmetic-domain:"+type(exc).__name__
        dense = self.step(state, token, strategy="dense")
        return TreeStep(dense.token, dense.state, dense_reads=dense.dense_reads,
                        uncertain_attention_cuts=uncertain, attempt_stats=reads,
                        fallback_reason=reason)
