"""Optional Torch/ctypes wrapper for local E24 cut certificates on CUDA.

This module does NOT install an LLM attention replacement. A head flag is not
an end-to-end token certificate and is not a PyTorch numerical equivalence proof.
"""
from __future__ import annotations
import ctypes
from pathlib import Path


class CudaFrontier:
    def __init__(self, library: str | Path, q, kl, ku, counts, vmin, vmax, sums, candidates):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable")
        tensors = (q, kl, ku, counts, vmin, vmax, sums, candidates)
        if any(not x.is_cuda or x.device != q.device or not x.is_contiguous() for x in tensors):
            raise ValueError("all tensors must be contiguous on the same CUDA device")
        if q.ndim!=2 or kl.ndim!=3 or vmin.ndim!=3:
            raise ValueError("expected q[H,D], keys[H,M,D], values[H,M,DV]")
        h, d = q.shape
        hk, m, dk = kl.shape
        hv, mv, dv = vmin.shape
        if (h,d)!=(hk,dk) or (h,m)!=(hv,mv) or not 1<=d<=128 or not 1<=dv<=128 or m<1 or h<1:
            raise ValueError("unsupported dimensions")
        expected = (ku.shape==kl.shape and vmax.shape==vmin.shape and
                    counts.shape==(h,m) and sums.shape==(h,m,dv,2) and candidates.shape==(h,dv))
        if not expected:
            raise ValueError("shape mismatch")
        if any(x.dtype!=torch.float64 for x in (q,kl,ku,vmin,vmax,sums)):
            raise TypeError("bounds must be float64")
        if counts.dtype!=torch.int32 or candidates.dtype!=torch.int16:
            raise TypeError("counts require int32; candidate BF16 bit patterns require int16")
        # Validation is setup work and intentionally synchronizes once.
        if int(counts.min().item())<=0:
            raise ValueError("empty/overflowed nodes are forbidden; do not zero-pad the frontier")
        self.tensors=tensors
        self.shape=(h,m,d,dv)
        self.lib=ctypes.CDLL(str(Path(library).resolve()))
        self.launch=self.lib.statecut_residual_launch
        self.launch.restype=ctypes.c_int
        self.launch.argtypes=([ctypes.c_void_p]*8+[ctypes.c_int]*4+
                              [ctypes.c_void_p]*5)
        self.den=torch.empty((h,m,2),dtype=torch.float64,device=q.device)
        self.lower=torch.empty((h,m,dv,2),dtype=torch.float64,device=q.device)
        self.upper=torch.empty_like(self.lower)
        self.accepted=torch.empty((h,),dtype=torch.int32,device=q.device)

    def evaluate(self):
        """Asynchronous on the current Torch stream; returned flags need normal stream ordering.

        Every input/output allocation is kept alive by this object. This is not
        CUDA-graph-safe lifetime management for an external serving framework.
        """
        import torch
        stream=torch.cuda.current_stream(self.tensors[0].device)
        args=[ctypes.c_void_p(t.data_ptr()) for t in self.tensors]
        args.extend(self.shape)
        args.extend(ctypes.c_void_p(t.data_ptr()) for t in (self.den,self.lower,self.upper,self.accepted))
        args.append(ctypes.c_void_p(stream.cuda_stream))
        with torch.cuda.device(self.tensors[0].device):
            status=self.launch(*args)
        if status:
            raise RuntimeError(f"StateCut CUDA launch failed, error code {status}")
        return self.accepted
