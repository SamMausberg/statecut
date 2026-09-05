"""Device certificates checked against the independent exact rational target.

Set STATECUT_CUDA_LIBRARY to the freshly built shared library to run these.
Tests cover multi-node reduction, odd cells, invalid inputs, and a nondefault
Torch stream; empirical checks do not verify generated instructions formally.
"""
from fractions import Fraction as F
import os
from random import Random

import pytest

torch = pytest.importorskip("torch")
LIBRARY = os.environ.get("STATECUT_CUDA_LIBRARY")
pytestmark = pytest.mark.skipif(not LIBRARY or not torch.cuda.is_available(),
                               reason="requires CUDA and STATECUT_CUDA_LIBRARY")

from statecut.arithmetic import bf16_value, exp_reference, round_bf16_bits
from statecut.cuda_frontier import CudaFrontier
from statecut.residual import bf16_cell


def tensor(data, dtype=torch.float64):
    return torch.tensor(data, dtype=dtype, device="cuda").contiguous()


def make_frontier(query, keys, values, candidate):
    # Each case describes one head and equal-sized disjoint nodes.
    d, dv = len(query), len(values[0][0])
    counts = [[len(node) for node in keys]]
    kl = [[[min(row[j] for row in node) for j in range(d)] for node in keys]]
    ku = [[[max(row[j] for row in node) for j in range(d)] for node in keys]]
    vl = [[[min(row[j] for row in node) for j in range(dv)] for node in values]]
    vu = [[[max(row[j] for row in node) for j in range(dv)] for node in values]]
    sums = [[[[sum(row[j] for row in node)] * 2 for j in range(dv)] for node in values]]
    bits = [[b if b < 32768 else b - 65536 for b in candidate]]
    return CudaFrontier(LIBRARY, tensor([query]), tensor(kl), tensor(ku),
                        tensor(counts, torch.int32), tensor(vl), tensor(vu),
                        tensor(sums), tensor(bits, torch.int16))


@pytest.mark.parametrize("dim,value_dim,nodes", [(1,1,1), (3,7,3), (64,64,5), (128,128,2)])
def test_device_bounds_enclose_rational_rows(dim, value_dim, nodes):
    rng = Random(981 + dim)
    query = [F(rng.randrange(-4, 5), 32) for _ in range(dim)]
    keys = [[[F(rng.randrange(-8, 9), 16) for _ in range(dim)] for _ in range(4)]
            for _ in range(nodes)]
    values = [[[F(rng.randrange(-32, 33), 8) for _ in range(value_dim)] for _ in range(4)]
              for _ in range(nodes)]
    weights = [[exp_reference(sum(q * k for q, k in zip(query, row))) for row in node]
               for node in keys]
    den = sum(sum(node) for node in weights)
    output = [sum(w * row[j] for ws, vs in zip(weights, values) for w, row in zip(ws, vs)) / den
              for j in range(value_dim)]
    correct = [round_bf16_bits(x) for x in output]
    verifier = make_frontier(query, keys, values, correct)
    # Verify input allocation/stream ordering too.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        flags = verifier.evaluate().clone()
    stream.synchronize()
    for n in range(nodes):
        dlo, dhi = map(F, verifier.den[0,n].tolist())
        assert dlo <= sum(weights[n]) <= dhi
        for j in range(value_dim):
            cell = bf16_cell(correct[j])
            for threshold, bounds in [(cell.lo, verifier.lower), (cell.hi, verifier.upper)]:
                exact = sum(w * (row[j] - threshold) for w, row in zip(weights[n], values[n]))
                lo, hi = map(F, bounds[0,n,j].tolist())
                assert lo <= exact <= hi
    assert int(flags.item()) in (0, 1)
    wrong = [0 if b != 0 else 0x3f80 for b in correct]
    wrong_verifier = make_frontier(query, keys, values, wrong)
    assert wrong_verifier.evaluate().item() == 0


@pytest.mark.parametrize("bits", [0,1,2,0x7f,0x80,0x3f80,0x3f81,0x7f7f,0x8001,0xbf81,0xff7f])
def test_constant_output_including_subnormal_and_odd_cells(bits):
    value = bf16_value(bits)
    verifier = make_frontier([F(1)], [[[F(-1,4)], [F(1,4)]]],
                             [[[value], [value]]], [bits])
    assert verifier.evaluate().item() == 1


@pytest.mark.parametrize("bits,expected", [(0x3f80,1),(0x3f81,0),(0x3f82,1)])
def test_ties_to_even_at_exact_boundary(bits, expected):
    # Zero scores give exactly unit weights; a BF16 pair averages to a midpoint.
    a = bf16_value(0x3f80 if bits < 0x3f82 else 0x3f81)
    b = bf16_value(0x3f81 if bits < 0x3f82 else 0x3f82)
    verifier = make_frontier([F(0)], [[[F(0)]], [[F(0)]]], [[[a]], [[b]]], [bits])
    assert verifier.evaluate().item() == expected


@pytest.mark.parametrize("defect", ["nan-query", "reversed-keys", "reversed-values", "bad-moment",
                                   "out-of-domain", "zero-mass", "negative-zero", "infinity", "nan-cell"])
def test_invalid_or_unsupported_frontier_fails_closed(defect):
    verifier = make_frontier([F(1)], [[[F(0)]]], [[[F(1)]]], [0x3f80])
    q, kl, ku, _, vmin, vmax, sums, candidate = verifier.tensors
    if defect == "nan-query": q.fill_(float("nan"))
    elif defect == "reversed-keys": kl.fill_(1)
    elif defect == "reversed-values": vmin.fill_(2)
    elif defect == "bad-moment": sums.fill_(8)
    elif defect == "out-of-domain": kl.fill_(65); ku.fill_(65)
    elif defect == "zero-mass": kl.fill_(-64); ku.fill_(-64)
    elif defect == "negative-zero": candidate.fill_(-32768)
    elif defect == "infinity": candidate.fill_(0x7f80)
    elif defect == "nan-cell": candidate.fill_(0x7fc0)
    assert verifier.evaluate().item() == 0
