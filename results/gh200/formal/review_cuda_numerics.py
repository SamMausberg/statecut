#!/usr/bin/env python3
"""Reproduce the independent E24 arithmetic and BF16 summary edge probes."""
import ctypes
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import runpy
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def float_bits(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def bits_float(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def exact_grid(value):
    """An independent rational oracle; no StateCut rounding helper is used."""
    scaled = Fraction(value) * (1 << 24)
    lower, remainder = divmod(scaled.numerator, scaled.denominator)
    if 2 * remainder > scaled.denominator or (
        2 * remainder == scaled.denominator and lower % 2
    ):
        lower += 1
    return Fraction(lower, 1 << 24)


def bf16_fraction(bits):
    sign = -1 if bits & 0x8000 else 1
    magnitude = bits & 0x7fff
    exponent, mantissa = magnitude >> 7, magnitude & 127
    coefficient = mantissa if exponent == 0 else 128 + mantissa
    power = -133 if exponent == 0 else exponent - 134
    return sign * coefficient * Fraction(2) ** power


def main():
    rng = random.Random(624091)
    patterns = set()
    for exponent in range(2047):
        for mantissa in [0, 1, 2, (1 << 51) - 1, 1 << 51, (1 << 52) - 2, (1 << 52) - 1]:
            patterns.add((exponent << 52) | mantissa)
    for _ in range(6000):
        patterns.add(rng.randrange(0x7ff0000000000000))
    for lower in [0, 1, 2, 3, 127, 128, 255, 256, 2**24 - 1, 2**31 - 1,
                  2**51 - 1, 2**52 - 1, 2**53 - 1]:
        mid = float(Fraction(2 * lower + 1, 1 << 25))
        patterns.update(bits_float(x) for x in (
            math.nextafter(mid, 0), mid, math.nextafter(mid, math.inf)))
    accepted = rejected_overflow = 0
    with tempfile.TemporaryDirectory(prefix="statecut-grid-review-") as temporary:
        temporary = Path(temporary)
        source = temporary / "probe.cpp"
        source.write_text(
            '#include "residual.cuh"\n#include <cstring>\n'
            'extern "C" double grid_bits(uint64_t bits) { '
            'double x; std::memcpy(&x,&bits,8); '
            'return statecut::rne_e24_grid(x); }\n')
        binary = temporary / "probe.so"
        subprocess.run([
            "c++", "-std=c++17", "-shared", "-fPIC", "-O2", "-fno-fast-math",
            "-ffp-contract=off", "-I", str(ROOT / "cuda"), str(source), "-o", str(binary)
        ], check=True)
        library = ctypes.CDLL(str(binary))
        library.grid_bits.argtypes = [ctypes.c_uint64]
        library.grid_bits.restype = ctypes.c_double
        for pattern in sorted(patterns):
            value = float_bits(pattern)
            actual = library.grid_bits(pattern)
            if not math.isfinite(value * (1 << 24)):
                assert math.isnan(actual), (pattern, value, actual)
                rejected_overflow += 1
            else:
                assert Fraction(actual) == exact_grid(value), (pattern, value, actual)
                accepted += 1
        invalid_patterns = [bits_float(-1.), bits_float(-math.ldexp(1., -1074)),
                            bits_float(math.inf), bits_float(-math.inf), 0x7ff8000000000001]
        for pattern in invalid_patterns:
            assert math.isnan(library.grid_bits(pattern))
        assert library.grid_bits(0x8000000000000000) == 0.

    sys.path.insert(0, str(ROOT / "src"))
    from statecut.arithmetic import exp_reference

    score_values = {-1000., -100., -65., -64., -25., 0., 64.,
                    math.nextafter(-25., -math.inf), math.nextafter(-25., math.inf)}
    # Decimal locates difficult inputs; exact rational exponential enclosures
    # decide their expected E24 result independently of this approximation.
    with localcontext() as context:
        context.prec = 160
        for integer in [0, 1, 2, 3, 127, 255, 65535, 2**24 - 1, 2**32 - 1,
                        2**52 - 1, 2**80, 2**100]:
            midpoint = float(((Decimal(integer) + Decimal("0.5")) / (1 << 24)).ln())
            score_values.update([math.nextafter(midpoint, -math.inf), midpoint,
                                 math.nextafter(midpoint, math.inf)])
    score_values = sorted(score_values)
    requests = "".join(f"weights {value:.17g} {value:.17g}\n" for value in score_values)
    expected = [exp_reference(Fraction(value)) for value in score_values]
    weight_checks = {}
    for name in ["statecut_host_residual", "statecut_device_residual"]:
        binary = ROOT / "build-gh200" / name
        result = subprocess.run([str(binary)], input=requests, text=True,
                                capture_output=True, check=True)
        lines = result.stdout.splitlines()
        assert len(lines) == len(expected)
        exact_points = 0
        for value, line, true in zip(score_values, lines, expected):
            low, high = map(float, line.split())
            assert Fraction(low) <= true <= Fraction(high), (name, value, line, true)
            exact_points += int(Fraction(low) == true == Fraction(high))
        weight_checks[name] = {"passed": len(lines), "exact_points": exact_points,
                               "binary_sha256": sha256(binary.read_bytes()).hexdigest()}

    exact_sum_guard = runpy.run_path(str(ROOT / "scripts/audit_capture_cuda.py"))["exact_sum_guard"]
    passed_arrays = rejected_arrays = partial_checks = 0
    for span in [0, 1, 8, 16, 32, 40, 41, 42, 44, 46, 50, 80, 120]:
        for count in [1, 2, 3, 7, 16, 31, 64]:
            raw = [((50 + (span if i % 2 else 0)) << 7) |
                   (127 if i % 3 else 1) | (0x8000 if i % 3 else 0)
                   for i in range(count)]
            values = [bf16_fraction(bits) for bits in raw]
            if not exact_sum_guard(np.array(raw, dtype=np.uint16), count):
                rejected_arrays += 1
                continue
            passed_arrays += 1
            for order in [values, list(reversed(values)), sorted(values, key=abs),
                          sorted(values, key=abs, reverse=True)]:
                floating = 0.
                exact = Fraction(0)
                for value in order:
                    floating += float(value)
                    exact += value
                    assert Fraction(floating) == exact, (span, count, value, floating, exact)
                    partial_checks += 1
    assert not exact_sum_guard(np.array([0x3f80, 0x0001, 0xbf80], dtype=np.uint16), 3)
    assert sum(map(float, [Fraction(1), Fraction(1, 1 << 133), Fraction(-1)])) == 0.
    assert not exact_sum_guard(np.array([0x7f80], dtype=np.uint16), 1)
    assert not exact_sum_guard(np.array([0x7fc1], dtype=np.uint16), 1)
    assert exact_sum_guard(np.array([0, 0x8000], dtype=np.uint16), 2)
    scale, smallest_bf16 = math.ldexp(1., -1000), math.ldexp(1., -133)
    assert math.frexp(scale)[0] == 0.5 and smallest_bf16 * scale == 0.
    report = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": 624091,
        "grid_oracle": "independent Fraction divmod, exact half comparison, and integer parity",
        "grid_accepted_cases": accepted,
        "grid_overflow_rejections": rejected_overflow,
        "grid_invalid_input_rejections": len(invalid_patterns),
        "grid_binary64_exponents_covered": 2047,
        "weight_endpoint_checks": weight_checks,
        "sum_guard_accepted_arrays": passed_arrays,
        "sum_guard_rejected_arrays": rejected_arrays,
        "sum_guard_exact_partial_additions_checked": partial_checks,
        "scale_counterexample": {
            "input_hex": smallest_bf16.hex(), "scale_hex": scale.hex(),
            "actual_product_hex": (smallest_bf16 * scale).hex(),
            "exact_product": "2^-1133",
            "impact": "Power-of-two scaling alone is insufficient. The reviewed audit now restricts to exactly 1/8; its BF16 scaling is exact."
        },
        "source_sha256": {path: sha256((ROOT / path).read_bytes()).hexdigest() for path in [
            "cuda/residual.cuh", "cuda/interval.cuh", "scripts/audit_capture_cuda.py",
            "results/gh200/formal/review_cuda_numerics.py"]},
        "proof_reuse": "StateCut.monotone_weight_bounds already proves generic monotone endpoint transport; no new Lean theorem is needed.",
        "scope": "Independent edge tests and mathematical review, not formal proof of CUDA or PyTorch implementation."
    }
    output = ROOT / "results/gh200/formal/cuda_numerical_review.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
