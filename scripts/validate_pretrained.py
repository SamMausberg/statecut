#!/usr/bin/env python3
"""Compare unchanged SDPA execution with the capture harness on pinned models.

Full K/V tensors and last-position logits are compared directly at every step.
This verifies observer transparency; it does not install a StateCut replacement.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from statecut.capture import CaptureSDPA

MODELS = {
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"),
}
PROMPTS = {
    "prose": "A careful scientific experiment states its assumptions, records observations, and tests alternative explanations. ",
    "repeated": "The blue bird sings beside the river. ",
    "code": "def sum_squares(values):\n    total = 0\n    for value in values:\n        total += value * value\n    return total\n",
}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--prompts", choices=PROMPTS, nargs="+", default=["prose", "repeated", "code"])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--output", default="results/gh200")
    args = parser.parse_args()
    if args.steps < 1 or min(args.lengths) < 2 or not torch.cuda.is_available():
        parser.error("positive steps, lengths >=2, and CUDA required")
    torch.set_num_threads(4)
    torch.manual_seed(7183)
    model_id, revision = MODELS[args.model]
    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=args.cache_dir,
                                              trust_remote_code=False, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, cache_dir=args.cache_dir,
                                                trust_remote_code=False, local_files_only=True,
                                                dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    def snapshot(result):
        cache = result.past_key_values
        kv = [(layer.keys.detach().cpu().clone(), layer.values.detach().cpu().clone())
              for layer in cache.layers]
        return {"logits": result.logits[:, -1].detach().cpu().clone(), "kv": kv,
                "lengths": [layer.get_seq_length() for layer in cache.layers]}

    for length in args.lengths:
        for prompt_name in args.prompts:
            phrase = tokenizer(PROMPTS[prompt_name], add_special_tokens=False)["input_ids"]
            token_ids = (phrase * ((length + len(phrase) - 1) // len(phrase)))[:length]
            inputs = torch.tensor([token_ids], device="cuda")
            reference = []
            chosen_runs = []
            capture_dir = output / "captures" / f"{args.model}-{prompt_name}-{length}"
            cap = CaptureSDPA(capture_dir, limit=model.config.num_hidden_layers * args.steps,
                              max_rows=length + args.steps)
            for recording in [False, True]:
                chosen = []
                mask = torch.ones_like(inputs)
                with cap if recording else nullcontext(), torch.inference_mode():
                    result = model(input_ids=inputs, attention_mask=mask, use_cache=True)
                    for step in range(args.steps + 1):
                        state = snapshot(result)
                        if not recording:
                            reference.append(state)
                        else:
                            expected = reference[step]
                            assert state["lengths"] == expected["lengths"]
                            assert torch.equal(state["logits"], expected["logits"]), (prompt_name, length, step, "logits")
                            assert len(state["kv"]) == len(expected["kv"])
                            for layer, (actual, target) in enumerate(zip(state["kv"], expected["kv"])):
                                for kind, a, b in zip(["K", "V"], actual, target):
                                    assert a.dtype == b.dtype and torch.equal(a.view(torch.int16), b.view(torch.int16)), (step, layer, kind)
                        if step == args.steps:
                            break
                        token = result.logits[:, -1].argmax(-1, keepdim=True)
                        chosen.append(int(token.item()))
                        mask = torch.cat((mask, torch.ones_like(token)), -1)
                        result = model(input_ids=token, attention_mask=mask,
                                       past_key_values=result.past_key_values, use_cache=True)
                chosen_runs.append(chosen)
            assert chosen_runs[0] == chosen_runs[1]
            expected_captures = model.config.num_hidden_layers * args.steps
            assert len(cap.files) == expected_captures, (len(cap.files), expected_captures, cap.skipped)
            row = {"prompt": prompt_name, "context_tokens": length, "decode_steps": args.steps,
                   "input_token_ids": token_ids, "selected_token_ids": chosen_runs[0],
                   "input_ids_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
                   "all_layer_kv_bit_equal": True, "last_position_logits_equal": True,
                   "states_compared": len(reference), "layers": model.config.num_hidden_layers,
                   "capture_directory": str(capture_dir), "captures": len(cap.files),
                   "captures_sha256": {name: hashlib.sha256((capture_dir / name).read_bytes()).hexdigest()
                                        for name in cap.files}, "skipped": cap.skipped}
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ("input_token_ids", "captures_sha256")}), flush=True)
    report = {"scope": "pretrained observer transparency; unchanged SDPA, no acceleration",
              "model": model_id, "revision": revision, "resolved_revision": model.config._commit_hash,
              "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
              "numpy": importlib.metadata.version("numpy"), "gpu": torch.cuda.get_device_name(),
              "numeric_profile": "BF16 inputs, Torch SDPA automatic backend dispatch",
              "backend_replacement_authorized": False, "workloads": rows}
    (output / f"pretrained_{args.model}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
