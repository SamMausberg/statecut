#!/usr/bin/env python3
"""Paired exact-reference experiments. Logical reads, not GPU speedups."""
from dataclasses import asdict
from fractions import Fraction as F
import argparse
import json
from pathlib import Path
from random import Random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from statecut import Cache, Entry, bf16, certify_bf16, dense_attention, verify_attention
from statecut.forest import ForestCache
from statecut.tree_attention import verify_tree_attention
from statecut.model import Model, Layer, small_model
from statecut.tree_model import TreeModel


def fixture():
    d=4
    z=tuple((F(0),)*d for _ in range(d))
    eye=tuple(tuple(F(i == j) for j in range(d)) for i in range(d))
    one=(F(1),)*d
    a=Layer(eye,eye,eye,eye,z,z,z,one,one)
    b=Layer(z,z,z,z,z,z,z,one,one)
    emb=((F(2),F(1),F(0),F(0)),(F(2),F(-1),F(2),F(0)),(F(2),F(2),F(-1),F(1)))
    head=((F(1),F(0),F(0),F(0)),(F(0),)*d,(F(0),)*d)
    return TreeModel(Model(emb,(a,b),one,head))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,
                        default=Path(__file__).resolve().parents[1]/"results/v2_experiments.json")
    args=parser.parse_args()
    reports=[]
    for n in (128,512,2048,8192):
        tree,flat=ForestCache("tree",32),Cache("flat",32)
        start=time.perf_counter()
        for i in range(n):
            row=Entry((F((-1)**i,8),),(F(16)+F((-1)**(i//2),4),))
            tree,flat=tree.append(row),flat.append(row)
        build=time.perf_counter()-start
        q=(F(1),)
        exact=tuple(map(bf16,dense_attention(flat,q)))
        for method in ("v1-independent-flat", "v2-centered-divided-tree", "v2-direct-residual-tree"):
            if method.startswith("v1"):
                r=verify_attention(flat,q,certify_bf16,lambda a:tuple(map(bf16,a)),max_refinements=0)
                nodes=r.stats.summary_blocks
            else:
                r=verify_tree_attention(tree,q,certify_bf16,lambda a:tuple(map(bf16,a)),
                                        max_expansions=0,direct_bf16=method=="v2-direct-residual-tree")
                nodes=r.stats.summary_nodes
            assert r.value == exact == (F(16),)
            reports.append({"case":"balanced-nonconstant", "method":method,"entries":n,
                            "root_count":len(tree.frontier),"summary_records_read":nodes,
                            "raw_entries_read":r.stats.raw_entries,"fallback":r.used_dense_fallback,
                            "equal_e24_dense":True,"both_indexes_construction_cpu_seconds":build})
    rng=Random(20260905)
    tree,flat=ForestCache("adversarial",32),Cache("adversarial",32)
    for i in range(512):
        row=Entry(tuple(F(rng.randrange(-16,17),8) for _ in range(4)),
                  tuple(F(rng.randrange(-32,33),16) for _ in range(4)))
        tree,flat=tree.append(row),flat.append(row)
    q=(F(1),F(-1,2),F(1,4),F(-1,4))
    r=verify_tree_attention(tree,q,certify_bf16,lambda a:tuple(map(bf16,a)),
                            max_expansions=9,direct_bf16=True)
    assert r.value == tuple(map(bf16,dense_attention(flat,q)))
    reports.append({"case":"unstructured", "method":"v2-direct-residual-tree","entries":512,
                    "summary_records_read":r.stats.summary_nodes,"raw_entries_read":r.stats.raw_entries,
                    "fallback":r.used_dense_fallback,"equal_e24_dense":True})
    trajectories=[]
    for strategy in ("cuts","writes"):
        for seed in (0,1,8,19):
            model=TreeModel(small_model(seed))
            dense,filtered=model.initial(4),model.initial(4)
            token=seed%8
            accepts=raw=0
            for _ in range(24):
                a=model.step(dense,token,strategy="dense")
                b=model.step(filtered,token,strategy=strategy,max_expansions=3)
                assert a.state == b.state and a.token == b.token
                accepts+=int(b.write_frontier_accepted)
                raw+=b.dense_reads+sum(x.stats.raw_entries for x in b.certificates)
                dense,filtered,token=a.state,b.state,a.token
            trajectories.append({"strategy":strategy,"seed":seed,"steps":24,
                                 "all_tokens_and_states_equal":True,"write_frontier_accepts":accepts,
                                 "raw_rows_all_layers_steps":raw})
    model=fixture();state=model.initial(8)
    for i in range(32):
        state=model.step(state,i%3,strategy="dense").state
    a=model.step(state,1,strategy="dense");b=model.step(state,1,strategy="writes")
    assert a.state == b.state and a.token == b.token and b.write_frontier_accepted
    report={"scope":"synthetic E24 reference; no pretrained/GPU results",
            "attention":reports,"state_trajectories":trajectories,
            "write_frontier_separation":{"accepted":True,"equal_token_and_state":True,
                "non_singleton_attention_cuts":b.uncertain_attention_cuts,
                "reads":asdict(b.attempt_stats)},
            "timing_note":"Construction time is CPU only and includes both indexes; no speedup inferred"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__ == "__main__":
    main()
