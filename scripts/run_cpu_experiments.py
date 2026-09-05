#!/usr/bin/env python3
"""Reproducible CPU correctness and logical-read experiments, not GPU timings."""
import json,platform,sys,time
from pathlib import Path
from fractions import Fraction as F
from random import Random
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from statecut import Cache,Entry,bf16,certify_bf16,dense_attention,verify_attention
from statecut.model import small_model


def main():
    rows=[]
    for kind in ("constant_keys","clustered_keys","unstructured"):
        rng=Random(614)
        c=Cache(kind,32)
        for i in range(512):
            denom={"constant_keys":None,"clustered_keys":65536,"unstructured":8}[kind]
            k=tuple(F(1,4)+(F(rng.randrange(-8,9),denom) if denom else 0) for _ in range(4))
            v=tuple(F(rng.randrange(-8,9),8) for _ in range(4))
            c=c.append(Entry(k,v))
        q=(F(1),F(-1,2),F(1,4),F(-1,4))
        t=time.perf_counter()
        r=verify_attention(c,q,certify_bf16,lambda a:tuple(bf16(x) for x in a),max_refinements=4)
        elapsed=time.perf_counter()-t
        exact=tuple(bf16(x) for x in dense_attention(c,q))
        assert r.value==exact
        rows.append(dict(case=kind,entries=c.length,summary_blocks=r.stats.summary_blocks,
                         raw_entries_read=r.stats.raw_entries,raw_fraction=r.stats.raw_entries/c.length,
                         equal_reference=True,dense_fallback=r.used_dense_fallback,
                         cpu_seconds=elapsed))
    # Distinct nonconstant-key case. Read only the high-weight block.
    rng=Random(10)
    c=Cache("dominant_block",32)
    for i in range(512):
        if i<480:
            k=(F(-14)+F(rng.randrange(-4,5),32),F(0),F(0),F(0))
            v=tuple(F(rng.randrange(-8,9),16) for _ in range(4))
        else:
            k=(F(rng.randrange(-4,5),8),F(0),F(0),F(0))
            v=(F(1,2),)*4
        c=c.append(Entry(k,v))
    q=(F(1),F(0),F(0),F(0))
    r=verify_attention(c,q,certify_bf16,lambda a:tuple(bf16(x) for x in a),4)
    assert r.value==tuple(bf16(x) for x in dense_attention(c,q))
    rows.append(dict(case="dominant_block",entries=c.length,
                     summary_blocks=r.stats.summary_blocks,
                     raw_entries_read=r.stats.raw_entries,
                     raw_fraction=r.stats.raw_entries/c.length,
                     equal_reference=True,dense_fallback=r.used_dense_fallback))
    trajectories=[]
    for seed in range(4):
        m=small_model(seed)
        d,f=m.initial(8),m.initial(8)
        token=seed
        reads=total=terminal_early=0
        for t in range(20):
            a=m.step(d,token,certified=False)
            b=m.step(f,token,max_refinements=2)
            assert a.token==b.token and a.state==b.state
            total+=a.dense_reads
            reads+=sum(c.stats.raw_entries for c in b.certificates)
            terminal_early+=int(b.certificates[-1].stats.raw_entries<b.state.position)
            d,f,token=a.state,b.state,a.token
        trajectories.append(dict(seed=seed,steps=20,all_tokens_equal=True,all_kv_equal=True,
                                 reference_raw_entries=total,certified_raw_entries=reads,
                                 terminal_early_accepts=terminal_early))
    result=dict(python=platform.python_version(),platform=platform.platform(),
                semantic_profile="RATIONAL_BF16_V1",pretrained_model=False,
                gpu_present=False,lean_compiled=False,attention=rows,trajectories=trajectories,
                note="Logical raw reads exclude metadata, weights, construction, and writes. CPU timings are not GH200 speedups or controlled comparisons: cold BF16 table initialization and memoized exponentials differ between cases.")
    out=Path(__file__).resolve().parents[1]/"results"/"cpu_experiments.json"
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
