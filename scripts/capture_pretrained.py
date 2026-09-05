#!/usr/bin/env python3
"""Run an unmodified numeric SDPA path and save selected decode observations.

Requires optional transformers/torch, a BF16-compatible causal LM and CUDA.
No remote model code is executed. A hub model needs an immutable 40-hex revision.
This is deliberately NOT an accelerator and NOT a benchmark.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from statecut.capture import CaptureSDPA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=4096)
    a = p.parse_args()
    local = Path(a.model).is_dir()
    if a.steps < 1:
        p.error("steps must be positive")
    if not local and not (a.revision and re.fullmatch(r"[a-fA-F0-9]{40}", a.revision)):
        p.error("hub models require --revision with the immutable 40-hex commit")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise SystemExit("NOT RUN: CUDA GPU required")
    text = Path(a.prompt_file).read_text()
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, revision=a.revision, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    if inputs["input_ids"].shape[-1] < 1:
        raise SystemExit("empty token sequence")
    chosen = []
    # Explicit greedy loop, not model.generate processors / sampling policies.
    with CaptureSDPA(a.output, limit=a.limit, max_rows=a.max_rows) as cap, torch.inference_mode():
        result = model(**inputs, use_cache=True)
        mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
        for _ in range(a.steps):
            token = result.logits[:, -1].argmax(dim=-1, keepdim=True)
            chosen.append(int(token.item()))
            mask = torch.cat((mask, torch.ones_like(token)), dim=-1)
            result = model(input_ids=token, attention_mask=mask,
                           past_key_values=result.past_key_values, use_cache=True)
    directory = Path(a.output)
    def sha(path):
        h = hashlib.sha256()
        with open(path,"rb") as f:
            for chunk in iter(lambda:f.read(1<<20), b""):
                h.update(chunk)
        return h.hexdigest()
    weights = {}
    if local:
        for f in sorted(Path(a.model).rglob("*")):
            if f.is_file() and f.suffix in (".safetensors", ".bin", ".json"):
                weights[str(f.relative_to(a.model))] = sha(f)
    manifest = {"model":a.model, "revision":a.revision,
                "resolved_revision":getattr(model.config,"_commit_hash",None),
                "local_artifact_sha256":weights, "prompt_sha256":sha(a.prompt_file),
                "captures_sha256":{f:sha(directory/f) for f in cap.files},
                "greedy_selected_ids":chosen, "attention_backend_requested":"sdpa",
                "torch":torch.__version__, "cuda":torch.version.cuda,
                "transformers":importlib.metadata.version("transformers"),
                "python":sys.version, "platform":platform.platform(),
                "device":torch.cuda.get_device_name(),
                "device_capability":list(torch.cuda.get_device_capability()),
                "contract":"unchanged numeric calls; capture overhead invalidates latency measurements",
                "bridge_to_e24":"NOT PROVED", "coverage":len(cap.files)}
    (directory/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if not cap.files:
        raise SystemExit("NO COVERAGE: no supported decode SDPA call captured; inspect coverage.json")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
