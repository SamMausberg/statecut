# StateCut paper

[Read the PDF](main.pdf). Samuel Mausberg is the author; the affiliation is StateCut Project. This private research draft uses the official ICML 2026 preprint style. It has eight pages of main text, followed by references and complete mathematical proofs, reproducibility details, and the compiled Lean inventory.

## Build

From the repository root:

```sh
make -C paper
```

The build needs Python 3, `latexmk`, pdfLaTeX, BibTeX, TikZ/PGFPlots, and the `standalone` class. On the GH200 these are supplied by the installed TeX Live packages. Python rendering uses only the standard library. The official style files are vendored without modification; [template_source.json](template_source.json) records the conference archive URL and SHA-256 hashes for the archive and each vendored file.

The build regenerates tables and the measured ablation CSV from checked-in JSON, then compiles the paper and four standalone vector figures. It does not run experiments. [generated/input_manifest.json](generated/input_manifest.json) identifies every measurement input by path and SHA-256. Missing required evidence, inconsistent success records, or stale formal source hashes stop rendering. Unchanged generated files retain their timestamps so a repeated build does not needlessly regenerate the PDF.

To render a different paired logical-read experiment while retaining the selected GH200 evidence:

```sh
python3 paper/render_results.py --experiment path/to/v2_experiments.json
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Running `make` again restores the default input `results/gh200/algorithm/v2_experiments.json`. Full experiment commands and pinned model revisions are documented in [the repository reproduction guide](../docs/REPRODUCE.md).

## Figures and evidence

The figures are native TikZ/PGFPlots source with embedded vector output and redundant labels or line styles:

| Figure | Source | Standalone PDF | Input |
|---|---|---|---|
| Persistent disjoint frontier | [frontier.tex](figures/frontier.tex) | [frontier.pdf](figures/frontier.pdf) | Algorithm structure |
| Transactional write certification | [transaction.tex](figures/transaction.tex) | [transaction.pdf](figures/transaction.pdf) | State contract |
| Sharp finite-count envelope | [envelope.tex](figures/envelope.tex) | [envelope.pdf](figures/envelope.pdf) | Exact analytic formula; not model measurements |
| Finite-count raw-read ablation | [integer_ablation.tex](figures/integer_ablation.tex) | [integer_ablation.pdf](figures/integer_ablation.pdf) | `results/gh200/integer_moment.json` |

The finite-count bound is sharp for the retained count, sum, range, and independent common weight-box constraints. Its handwritten proof and compiled Lean results do not establish a formal refinement of Python or CUDA. The paper preserves the negative pretrained findings: no complete captured attention call is certified, all 72 exact-reference sampled heads use raw fallback, and 501 of 4,608 observed backend coordinate bits differ from E24. Local gate timings have separate construction and launch scopes and do not establish an end-to-end speedup.

[The literature audit](../docs/LITERATURE_AUDIT.md) records primary sources and the contribution boundary. The chord bound and general adaptive-predicate, majorization, and neural interval methods are credited as existing mathematics. The paper makes no exhaustive priority claim.

## Verification

[build_verification.json](build_verification.json) records the final PDF/source hashes, tool versions, warning checks, page count, and pages visually inspected. The build checks citation resolution and text overflow; rendered-page inspection checks layout and figure labels. The PDF title and author metadata are checked alongside the visible title block. The manuscript has not been submitted or sent to an external format checker.
