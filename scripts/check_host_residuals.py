#!/usr/bin/env python3
"""Cross-check outward C++ bounds against independent exact rational cases."""
from fractions import Fraction as F
from random import Random
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from statecut.arithmetic import exp_reference
from statecut.residual import bf16_cell

ROOT = Path(__file__).resolve().parents[1]

def run():
    rng = Random(90833)
    requests, checkers = [], []
    def fstr(x):
        return format(float(x), ".17g")
    for case in range(2500):
        n = rng.randrange(1, 100)
        vs = [F(rng.randrange(-256, 257), 16) for _ in range(n)]
        total, lo, hi = sum(vs), min(vs), max(vs)
        a = F(rng.randrange(0, 100), 16)
        b = a+F(rng.randrange(0, 100), 16)
        w = [a+(b-a)*F(rng.randrange(0, 17), 16) for _ in vs]
        t = F(rng.randrange(-300, 301), 16)
        requests.append("moment "+str(n)+" "+" ".join(map(fstr, (total,total,lo,hi,a,b,t))))
        exact = sum(x*(y-t) for x,y in zip(w,vs))
        checkers.append(("interval",exact))
    # All finite canonical BF16 cells, not just normal positive values.
    for code in range(65536):
        if code==0x8000 or (code&0x7f80)==0x7f80:
            continue
        c=bf16_cell(code)
        requests.append(f"cell {code}")
        checkers.append(("cell", c))
    for i in range(513):
        z=F(i-256,8)
        requests.append(f"weights {fstr(z)} {fstr(z)}")
        checkers.append(("interval",exp_reference(z)))
    proc=subprocess.run([str(ROOT/"build/statecut_host_residual")],
                        input="\n".join(requests)+"\n",text=True,capture_output=True,check=True)
    lines=proc.stdout.splitlines()
    if len(lines)!=len(checkers):
        raise AssertionError("C++ output count mismatch")
    for line,(kind,expected) in zip(lines,checkers):
        parts=line.split()
        low,high=F(float(parts[0])),F(float(parts[1]))
        if kind=="interval":
            assert low<=expected<=high, (line,expected)
        else:
            assert (low,high,int(parts[2]),int(parts[3]))==(expected.lo,expected.hi,int(expected.closed),1)
    result={"passed":len(lines),"moment_cases":2500,"all_finite_canonical_bf16_cells":65279,
            "e24_weight_cases":513,"device_executed":False,
            "scope":"host cross-check only; not Lean or CUDA implementation verification"}
    (ROOT/"results/host_residuals.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    run()
