import json
from pathlib import Path
import subprocess
import sys
import importlib.util

import numpy as np
import pytest

from statecut.arithmetic import round_bf16_bits


def write_capture(capture):
    q = np.array([1, 1, 1], dtype=float)
    k = np.array([0, 4096, -100], dtype=float)
    def encode(values):
        return np.array([round_bf16_bits(x) for x in values], dtype=np.uint16).reshape(1, 3, 1, 1)
    np.savez(capture, q=encode(q), k=encode(k), v=encode(q), out=encode(q),
             metadata=np.array(json.dumps({"schema":"statecut-sdpa-observation-v1",
                                           "full_visible_zero_bias":True,
                                           "scale_hex":float(1).hex(),
                                           "q_to_kv_group_size":1})))


def test_capture_audit_retains_supported_heads_and_records_domain_failures(tmp_path):
    capture = tmp_path / "mixed_domains.npz"
    output = tmp_path / "audit.json"
    write_capture(capture)
    script = Path(__file__).resolve().parents[1] / "scripts/audit_capture.py"
    result = subprocess.run([sys.executable, str(script), str(capture), "--heads", "3",
                             "--block-size", "1", "--output", str(output)],
                            capture_output=True, text=True, check=True)
    report = json.loads(output.read_text())
    assert json.loads(result.stdout) == report
    assert report["heads_completed"] == 3 and report["supported_heads"] == 1
    assert report["unsupported_heads"] == 2
    first, large, zero = report["rows"]
    assert first["e24_equal_dense"] is True
    assert first["observed_backend_bit_matches"] == 1
    assert large["e24_equal_dense"] is None and large["error_kind"] == "oracle-resource"
    assert large["exact_score_min"] == large["exact_score_max"] == "4096"
    assert zero["e24_equal_dense"] is None and "denominator zero" in zero["error"]
    assert zero["error_kind"] == "zero-reference-denominator"
    assert report["backend_replacement_authorized"] is False


def test_capture_audit_does_not_hide_an_internal_enclosure_failure(tmp_path, monkeypatch):
    capture = tmp_path / "valid.npz"
    output = tmp_path / "audit.json"
    write_capture(capture)
    script = Path(__file__).resolve().parents[1] / "scripts/audit_capture.py"
    spec = importlib.util.spec_from_file_location("statecut_audit_capture", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def inconsistent(*args, **kwargs):
        raise ArithmeticError("inconsistent sound enclosures: provenance/arithmetic defect")
    monkeypatch.setattr(module, "verify_tree_attention", inconsistent)
    monkeypatch.setattr(sys, "argv", [str(script), str(capture), "--output", str(output)])
    with pytest.raises(ArithmeticError, match="provenance/arithmetic defect"):
        module.main()
    assert not output.exists()
