#!/usr/bin/env python3
"""Synthetic summary-gate microbenchmark, NOT an LLM speedup benchmark.

Records the dense SDPA primitive too, but does not report their latency ratio
as a speedup: the E24 gate and Torch SDPA have different numerical contracts.
Proposal is the construction's known value 16, not a dense-oracle output.
"""
import argparse
import json
from pathlib import Path
import platform
import statistics
import time


def main():
    import torch
    import torch.nn.functional as nnf
    from statecut.cuda_frontier import CudaFrontier
    p=argparse.ArgumentParser()
    p.add_argument("--library",default="build-gh200/libstatecut_residual_cuda.so")
    p.add_argument("--length",type=int,default=32768)
    p.add_argument("--heads",type=int,default=8)
    p.add_argument("--dim",type=int,default=64)
    p.add_argument("--repeats",type=int,default=100)
    p.add_argument("--out",default="results/gh200_microbenchmark.json")
    a=p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. No GPU timing or certificate claimed.")
    if not a.length>0 or a.length%4 or not 1<=a.dim<=128 or not a.heads>0 or a.repeats<2:
        raise SystemExit("length must be a positive multiple of 4; 1<=dim<=128; heads>0; repeats>=2")
    h,n,d=a.heads,a.length,a.dim
    if n>=2**31:
        raise SystemExit("count exceeds signed wrapper ABI")
    start=time.perf_counter()
    rows=torch.arange(n,device="cuda")
    k=torch.zeros((1,h,n,d),device="cuda",dtype=torch.bfloat16)
    k[...,0]=((rows%2)*2-1).to(torch.bfloat16)/16
    v=(16+((((rows//2)%2)*2-1).to(torch.float32)/4))[None,None,:,None].expand(1,h,n,d).to(torch.bfloat16).contiguous()
    q=torch.zeros((h,d),device="cuda",dtype=torch.float64);q[:,0]=1
    kl=k[0].to(torch.float64).amin(dim=1,keepdim=True).contiguous()
    ku=k[0].to(torch.float64).amax(dim=1,keepdim=True).contiguous()
    vl=v[0].to(torch.float64).amin(dim=1,keepdim=True).contiguous()
    vu=v[0].to(torch.float64).amax(dim=1,keepdim=True).contiguous()
    # Exact by construction: alternating dyadic values, n multiple of four,
    # each sum 16*n is exactly representable in binary64 for the count domain.
    sums=torch.full((h,1,d,2),float(16*n),device="cuda",dtype=torch.float64)
    counts=torch.full((h,1),n,device="cuda",dtype=torch.int32)
    candidate=torch.full((h,d),0x4180,device="cuda",dtype=torch.int16)
    verifier=CudaFrontier(a.library,q,kl,ku,counts,vl,vu,sums,candidate)
    torch.cuda.synchronize()
    construction_ms=1000*(time.perf_counter()-start)
    qb=q.to(torch.bfloat16)[None,:,None,:].contiguous()
    def dense():
        return nnf.scaled_dot_product_attention(qb,k,v,dropout_p=0.0,is_causal=False,scale=1.0)
    for _ in range(10):
        verifier.evaluate();dense()
    torch.cuda.synchronize()
    flags=verifier.evaluate().clone()
    reference=dense()
    torch.cuda.synchronize()
    if not bool((flags==1).all().item()):
        raise AssertionError("synthetic exact-cut gate did not accept on this device")
    parity=bool((reference==torch.tensor(16,dtype=torch.bfloat16,device="cuda")).all().item())
    if not parity:
        raise AssertionError("SDPA observed-cut comparison failed; no backend proof was assumed")
    def timings(fn):
        device_ms=[];wall_ms=[]
        for _ in range(a.repeats):
            start_evt=torch.cuda.Event(enable_timing=True);end_evt=torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize();start=time.perf_counter()
            start_evt.record();fn();end_evt.record();end_evt.synchronize()
            device_ms.append(start_evt.elapsed_time(end_evt));wall_ms.append(1000*(time.perf_counter()-start))
        def describe(xs):
            xs=sorted(xs)
            return {"p50":statistics.median(xs),"p95":xs[min(len(xs)-1,int(.95*len(xs)))],"min":xs[0]}
        return {"device_ms":describe(device_ms),"synchronized_wall_ms":describe(wall_ms)}
    result={"scope":"synthetic local E24 certificate microbenchmark, not end-to-end acceleration",
            "gpu":torch.cuda.get_device_name(),"compute_capability":torch.cuda.get_device_capability(),
            "torch":torch.__version__,"cuda":torch.version.cuda,"machine":platform.machine(),
            "heads":h,"length":n,"dim":d,"construction_ms":construction_ms,
            "summary_nodes_per_head":1,"raw_KV_rows_read_by_gate":0,
            "proposal":"constant 16 from construction; no dense oracle proposal",
            "gate":timings(verifier.evaluate),"torch_sdpa":timings(dense),
            "observed_SDPA_cut_equality":parity,"pretrained":False,
            "end_to_end_speedup":None,"backend_equivalence_proved":False}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
