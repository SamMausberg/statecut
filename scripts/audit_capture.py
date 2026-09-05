#!/usr/bin/env python3
"""Offline E24 audit of observed BF16 SDPA tensors. Does not certify the backend."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from statecut.arithmetic import bf16_value, round_bf16_bits, bf16, certify_bf16
from statecut.cache import Entry
from statecut.forest import ForestCache
from statecut.tree_attention import verify_tree_attention, dense_tree_attention


def main():
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture")
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--max-expansions", type=int, default=0)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.heads < 1 or a.block_size < 1 or a.max_expansions < 0:
        p.error("invalid positive sizes / nonnegative budget")
    with np.load(a.capture, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
        arrays = {k:z[k] for k in ("q","k","v","out")}
    if meta.get("schema") != "statecut-sdpa-observation-v1" or not meta.get("full_visible_zero_bias"):
        raise SystemExit("unsupported observation schema")
    if any(x.dtype != np.uint16 or x.ndim != 4 for x in arrays.values()):
        raise SystemExit("invalid tensor encoding")
    q,k,v,out = (arrays[x] for x in ("q","k","v","out"))
    h, hk, n = q.shape[1], k.shape[1], k.shape[2]
    group = meta["q_to_kv_group_size"]
    if (q.shape[0] != 1 or q.shape[2] != 1 or k.shape[0] != 1 or v.shape[0] != 1
            or hk < 1 or group*hk != h or v.shape[1:3] != (hk,n)
            or q.shape[-1] != k.shape[-1] or out.shape != (1,h,1,v.shape[-1])):
        raise SystemExit("shape / head map mismatch")
    scale = F(float.fromhex(meta["scale_hex"]))
    rows = []
    for head in range(min(a.heads, h)):
        kh = head//group
        cache = ForestCache(f"capture:{Path(a.capture).name}:head:{head}", a.block_size)
        for i in range(n):
            cache = cache.append(Entry(tuple(bf16_value(int(b)) for b in k[0,kh,i]),
                                       tuple(bf16_value(int(b)) for b in v[0,kh,i])))
        query = tuple(scale*bf16_value(int(b)) for b in q[0,head,0])
        r = verify_tree_attention(cache, query, certify_bf16, lambda x:tuple(map(bf16,x)),
                                  max_expansions=a.max_expansions, direct_bf16=True)
        dense = tuple(map(bf16,dense_tree_attention(cache,query)))
        bits = [round_bf16_bits(x) for x in r.value]
        observed = [int(b) for b in out[0,head,0]]
        rows.append({"head":head, "kv_head":kh, "entries":n,
                     "e24_equal_dense":r.value == dense,
                     "e24_cut_accepted_without_fallback":r.accepted_from_bounds,
                     "raw_entries":r.stats.raw_entries,"summary_nodes":r.stats.summary_nodes,
                     "observed_backend_bit_matches":sum(x == y for x,y in zip(bits,observed)),
                     "coordinates":len(bits),
                     "negative_zero_note":"E24 canonicalizes zero; observed bits are not canonicalized"})
        if r.value != dense:
            raise AssertionError("E24 implementation regression")
    report = {"claim":"E24 correctness test and observational backend comparison only",
              "backend_replacement_authorized":False,
              "reason":"no proved enclosure of the pinned deployed floating-point operator",
              "rows":rows}
    Path(a.output).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__ == "__main__":
    main()
