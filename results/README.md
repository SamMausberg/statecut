# Evidence provenance

`gh200/` contains the current v0.3.0 measurements from the September 5, 2026
GH200 validation run. `gh200/STATUS.json` and `gh200/SOURCE_MANIFEST.sha256`
summarize and bind the current artifact; `scripts/check_claims.py` checks their
consistency. Formal evidence has its own source and log hashes in
`gh200/formal/summary.json`.

The files directly under `results/` (other than this index) were imported at
commit `73cd266` as v0.2.0 evidence. Their manifest describes that imported
source, not the subsequently modified working tree. `v0.1.0/` is the earlier
archive supplied by the repository. Historical results are not relabeled as
current runs.

Current raw captures are excluded from Git because they duplicate tensors
that can be reproduced from public pinned checkpoints and tracked input token
IDs. Each model manifest includes the SHA-256 of every captured file. The
audit results, input sequences, package versions, command scripts, selected
per-head results, and compilation/test logs are retained.

`capture_cuda_audit_half_quantum.json` records the first CUDA enclosure.
`capture_cuda_audit.json` records the tighter monotone-grid enclosure. These
are local E24 certificate observations, including failures; neither authorizes
a model replacement. The synthetic microbenchmark compares different numeric
contracts and therefore does not report their ratio as a certified speedup.
