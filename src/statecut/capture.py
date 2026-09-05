"""Capture-only SDPA probe. Never replaces an output or authorizes a fast path.

This temporarily wraps torch.nn.functional.scaled_dot_product_attention.
Use only in an isolated, single-threaded audit process, not in serving. Model
code that imported or fused SDPA elsewhere may bypass it; coverage is counted.
"""
from __future__ import annotations
from contextlib import AbstractContextManager
from pathlib import Path
import json
import math


class CaptureSDPA(AbstractContextManager):
    def __init__(self, directory: str | Path, *, limit: int = 4,
                 max_rows: int = 4096, allow_cpu: bool = False):
        if limit < 1 or max_rows < 1:
            raise ValueError("positive capture limit and max_rows required")
        self.directory = Path(directory)
        self.limit, self.max_rows, self.allow_cpu = limit, max_rows, allow_cpu
        self.calls = 0
        self.files: list[str] = []
        self.skipped: dict[str, int] = {}
        self._original = None

    def _skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0)+1

    def _record(self, args, kwargs, out) -> None:
        import numpy as np
        import torch
        self.calls += 1
        if len(self.files) >= self.limit:
            self._skip("capture-limit")
            return
        def arg(i, name, default=None):
            return args[i] if len(args) > i else kwargs.get(name, default)
        q, k, v = (arg(i, name) for i, name in enumerate(("query", "key", "value")))
        mask = arg(3, "attn_mask")
        dropout = arg(4, "dropout_p", 0.0)
        causal = arg(5, "is_causal", False)
        if not all(isinstance(x, torch.Tensor) and x.ndim == 4 for x in (q,k,v,out)):
            self._skip("unsupported-rank"); return
        if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1 or q.shape[2] != 1:
            self._skip("not-batch-one-decode"); return
        if k.shape[2] > self.max_rows:
            self._skip("max-rows-not-a-truncation"); return
        if dropout != 0 or causal:
            # A top-left causal one-query mask is NOT a full-cache decode mask.
            self._skip("dropout-or-causal-semantics"); return
        if mask is not None:
            valid = (bool(mask.all().item()) if mask.dtype == torch.bool
                     else bool((mask == 0).all().item()))
            if not valid:
                self._skip("not-full-visible-zero-bias"); return
        if not all(x.dtype == torch.bfloat16 for x in (q,k,v,out)):
            self._skip("not-bf16"); return
        if not self.allow_cpu and not all(x.is_cuda for x in (q,k,v,out)):
            self._skip("not-cuda"); return
        if not all(bool(torch.isfinite(x).all().item()) for x in (q,k,v,out)):
            self._skip("nonfinite"); return
        h, hk, hv = q.shape[1], k.shape[1], v.shape[1]
        gqa = bool(kwargs.get("enable_gqa", False))
        if h < 1 or hk < 1 or hk != hv or h % hk or (h != hk and not gqa):
            self._skip("unsupported-head-mapping"); return
        if k.shape[2] != v.shape[2] or q.shape[-1] != k.shape[-1]:
            self._skip("invalid-shape"); return
        scale = kwargs.get("scale")
        # This float is an explicit observation-profile parameter, NOT a claim
        # about the device's internal scale representation or multiplication.
        scale = 1.0/math.sqrt(q.shape[-1]) if scale is None else float(scale)
        if not math.isfinite(scale):
            self._skip("nonfinite-scale"); return
        def bits(x):
            return x.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
        name = f"capture-{len(self.files):04d}.npz"
        path = self.directory/name
        if path.exists():
            raise FileExistsError(path)
        meta = {"schema": "statecut-sdpa-observation-v1", "call_index": self.calls,
                "scale_hex": scale.hex(), "enable_gqa": gqa,
                "q_to_kv_group_size": h//hk, "full_visible_zero_bias": True,
                "returned_output_unchanged": True,
                "profile_bridge": "ABSENT: observation cannot certify replacement"}
        np.savez(path, q=bits(q), k=bits(k), v=bits(v), out=bits(out),
                 metadata=np.array(json.dumps(meta)))
        self.files.append(name)

    def __enter__(self):
        import torch.nn.functional as functional
        if self._original is not None or getattr(functional.scaled_dot_product_attention,
                                                  "_statecut_capture", False):
            raise RuntimeError("nested or concurrent capture probe")
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise FileExistsError("capture directory must be empty")
        self._original = functional.scaled_dot_product_attention
        original = self._original
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            self._record(args, kwargs, result)
            return result
        wrapped._statecut_capture = True
        functional.scaled_dot_product_attention = wrapped
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        import torch.nn.functional as functional
        functional.scaled_dot_product_attention = self._original
        self._original = None
        (self.directory/"coverage.json").write_text(json.dumps({
            "sdpa_calls_seen": self.calls, "captures": self.files, "skipped": self.skipped,
            "claim": "capture only; unchanged reference outputs; no certified backend bridge",
            "coverage_warning": "calls through other aliases or fused backends are not observed"
        }, indent=2)+"\n")
        return False
