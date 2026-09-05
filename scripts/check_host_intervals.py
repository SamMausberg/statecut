#!/usr/bin/env python3
"""Differential test of C++ host emulation against exact fractions and mpmath."""
from fractions import Fraction as F
from pathlib import Path
from random import Random
import json,math,subprocess,sys
import mpmath as mp
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/"src"))
from statecut.arithmetic import Interval
exe=Path(sys.argv[1]) if len(sys.argv)>1 else root/"build"/"statecut_host_interval"
r=Random(3004)
lines=[];expected=[]
for _ in range(500):
    a,b=sorted(r.randint(-1000,1000)/16 for _ in range(2))
    c,d=sorted(r.randint(-1000,1000)/16 for _ in range(2))
    for op in ("add","mul"):
        lines.append(f"{op} {a} {b} {c} {d}")
        x,y=Interval(F(a),F(b)),Interval(F(c),F(d))
        expected.append(x+y if op=="add" else x*y)
    c,d=sorted((r.randint(1,1000)/16,r.randint(1,1000)/16))
    lines.append(f"div {a} {b} {c} {d}")
    expected.append(Interval(F(a),F(b))/Interval(F(c),F(d)))
mp.mp.dps=150
for i in range(-512,513):
    x=i/8
    lines.append(f"exp {x}")
    expected.append(mp.exp(mp.mpf(x)))
proc=subprocess.run([str(exe)],input="\n".join(lines)+"\n",text=True,capture_output=True,check=True)
outputs=proc.stdout.splitlines()
assert len(outputs)==len(expected)
for line,truth in zip(outputs,expected):
    lo,hi=map(float,line.split())
    assert math.isfinite(lo) and math.isfinite(hi) and lo<=hi
    if isinstance(truth,Interval):
        assert F(lo)<=truth.lo and truth.hi<=F(hi),(line,truth)
    else:
        assert mp.mpf(lo)<=truth<=mp.mpf(hi),(line,truth)
result=dict(passed=len(expected),arithmetic_cases=1500,exp_cases=1025,
            device_intrinsics_tested=False,
            note="Host emulation tested; this is not CUDA execution or a formal floating-point proof.")
(root/"results"/"host_interval_tests.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
