PYTHON ?= python3
LATEXMK ?= latexmk

.PHONY: all tables figures clean
all: tables figures
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex

tables:
	$(PYTHON) render_results.py

figures: tables
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=figures -jobname=frontier figures/frontier_standalone.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=figures -jobname=transaction figures/transaction_standalone.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=figures -jobname=envelope figures/envelope_standalone.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=figures -jobname=integer_ablation figures/integer_ablation_standalone.tex

clean:
	$(LATEXMK) -c main.tex
