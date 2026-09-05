#!/usr/bin/env python3
"""Offline summary acceptance on captured pretrained inputs, in the E24 profile.

Two proposals are explicitly distinguished: summary-only uniform-value mean and
the already observed SDPA answer (an oracle diagnostic). Neither path accelerates
the model. Summaries and all observed inputs are read in full during this audit.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from statecut.cuda_frontier import CudaFrontier


def exact_sum_guard(bits, count):
    """A sufficient integer-grid condition for exact binary64 sums of BF16.

    Every input is an integer multiple of 2**emin. If count times a bound on
    the absolute integer coefficient is <=2**53, every partial sum in any
    reduction order is representable exactly in binary64. No empirical epsilon.
    """
    import numpy as np
    magnitudes = bits & 0x7fff
    nonzero = magnitudes[magnitudes != 0]
    if not nonzero.size:
        return True
    exponent = ((nonzero >> 7) & 255).astype(np.int32)
    if bool((exponent == 255).any()):
        return False
    grid = np.maximum(exponent - 134, -133)
    return count * (255 << int(grid.max() - grid.min())) <= 2**53


def main():
    import numpy as np
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--library", default="build-gh200/libstatecut_residual_cuda.so")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[1, 16, 64, 128])
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", default="results/gh200/capture_cuda_audit.json")
    args = parser.parse_args()
    if min(args.block_sizes) < 1 or args.stride < 1:
        parser.error("positive block sizes and stride required")
    torch.set_num_threads(4)
    rows = []
    aggregates = defaultdict(lambda: {"heads": 0, "accepted_heads": 0, "calls": 0, "accepted_calls": 0,
                                     "invalid_weight_heads":0,"nonpositive_mass_heads":0,"valid_mass_heads":0})
    for manifest_path in args.manifests:
        manifest = json.loads(Path(manifest_path).read_text())
        for workload in manifest["workloads"]:
            directory = Path(workload["capture_directory"])
            files = sorted(workload["captures_sha256"])[::args.stride]
            for filename in files:
                import hashlib
                path = directory / filename
                assert hashlib.sha256(path.read_bytes()).hexdigest() == workload["captures_sha256"][filename]
                with np.load(path, allow_pickle=False) as capture:
                    metadata = json.loads(str(capture["metadata"]))
                    assert metadata["schema"] == "statecut-sdpa-observation-v1"
                    assert metadata["full_visible_zero_bias"]
                    data = {name: capture[name] for name in ["q", "k", "v", "out"]}
                scale = float.fromhex(metadata["scale_hex"])
                # Point-valued query scaling is exact for these two d=64 models.
                if scale != 0.125:
                    raise ValueError("this audit supports the exact d=64 scale 1/8; otherwise use outward queries")
                n = data["k"].shape[2]
                if not exact_sum_guard(data["v"], n):
                    raise ValueError("first moment exactness guard failed; use a directed summary builder")
                def decode(name):
                    return torch.from_numpy(data[name].view(np.int16).copy()).view(torch.bfloat16).to("cuda")
                q, k, v, observed = map(decode, ["q", "k", "v", "out"])
                h, group = q.shape[1], metadata["q_to_kv_group_size"]
                q = (q[0,:,0].to(torch.float64) * scale).contiguous()
                k = k[0].repeat_interleave(group, dim=0).to(torch.float64)
                v = v[0].repeat_interleave(group, dim=0).to(torch.float64)
                candidates = {
                    "summary_uniform_mean": v.sum(1).div(n).to(torch.bfloat16).view(torch.int16).contiguous(),
                    "observed_sdpa_oracle": observed[0,:,0].view(torch.int16).contiguous(),
                }
                # Canonical zero is required by the E24 ABI; this also makes
                # clear that candidate acceptance alone does not prove SDPA bits.
                candidates = {name: torch.where(bits == -32768, 0, bits).contiguous()
                              for name, bits in candidates.items()}
                for block in args.block_sizes:
                    spans = list(range(0, n, block))
                    def reduce_blocks(x, operation):
                        full = n // block
                        pieces = []
                        if full:
                            chunks = x[:,:full*block].reshape(h,full,block,x.shape[-1])
                            pieces.append(getattr(chunks, operation)(2))
                        if full * block < n:
                            pieces.append(getattr(x[:,full*block:], operation)(1,keepdim=True))
                        return torch.cat(pieces,1).contiguous()
                    kl, ku = reduce_blocks(k,"amin"), reduce_blocks(k,"amax")
                    vl, vu = reduce_blocks(v,"amin"), reduce_blocks(v,"amax")
                    total = reduce_blocks(v,"sum")
                    sums = torch.stack((total,total), -1).contiguous()
                    counts = torch.tensor([min(block,n-start) for start in spans], dtype=torch.int32,
                                          device="cuda").expand(h,-1).contiguous()
                    for proposal, candidate in candidates.items():
                        verifier = CudaFrontier(args.library,q,kl,ku,counts,vl,vu,sums,candidate)
                        accepted = int(verifier.evaluate().sum().item())
                        valid_weight = torch.isfinite(verifier.den).all(-1).all(-1)
                        # Positive count summaries have invalid masses only when
                        # the weight enclosure rejects its score domain here.
                        invalid_weight = int((~valid_weight).sum().item())
                        nonpositive = int((valid_weight & (verifier.den[:,:,0].sum(-1) <= 0)).sum().item())
                        key = (manifest["model"], workload["context_tokens"], block, proposal)
                        aggregate = aggregates[key]
                        aggregate["heads"] += h
                        aggregate["accepted_heads"] += accepted
                        aggregate["calls"] += 1
                        aggregate["accepted_calls"] += int(accepted == h)
                        aggregate["invalid_weight_heads"] += invalid_weight
                        aggregate["nonpositive_mass_heads"] += nonpositive
                        aggregate["valid_mass_heads"] += h-invalid_weight-nonpositive
                        rows.append({"capture": str(path), "model": manifest["model"],
                                     "context_tokens": workload["context_tokens"], "block_size": block,
                                     "nodes_per_head":len(spans), "proposal":proposal,
                                     "heads":h, "accepted_heads":accepted,
                                     "invalid_weight_heads":invalid_weight,
                                     "nonpositive_mass_heads":nonpositive})
            print(f"Audited {manifest['model']} {workload['prompt']} {workload['context_tokens']}: {len(files)} calls", flush=True)
    report = {"scope":"offline E24 local-gate acceptance on pretrained tensors; not a model replacement",
              "backend_replacement_authorized":False, "performance_claim":None,
              "cuda_bound":"continuous-mass chord (finite-count strengthening currently CPU/Lean only)",
              "weight_enclosure":"monotone E24 rounding of outward exp endpoints; exact negative tail",
              "exact_first_moments":"integer-grid sufficient condition checked for each capture",
              "all_raw_rows_read_during_preprocessing":True, "stride":args.stride,
              "aggregates":[dict(model=k[0],context_tokens=k[1],block_size=k[2],proposal=k[3],**v)
                            for k,v in sorted(aggregates.items())], "rows":rows}
    Path(args.output).write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
