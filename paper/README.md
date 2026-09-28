# Papers

| Paper | Target | Source | PDF |
|---|---|---|---|
| ViterbiNet Revisited: A Matched-Information Classical Baseline for Learned Trellis Detection | IEEE Communications Letters (3 pages) | `letter/main.tex` | `letter/main.pdf` |
| When Does Learning Help Trellis Detection? Matched Baselines, Structure, and Complexity of Model-Based Deep Viterbi Receivers | IEEE Transactions on Machine Learning in Communications and Networking (7 pages) | `journal/main.tex` | `journal/main.pdf` |

Both use `refs.bib` (every entry checked against a publisher or index record)
and the figures in `figures/`.

## Rebuild

```bash
python3 paper/make_figures.py                 # figures/*.pdf and figures/numbers.tex from Results/metrics
cd paper/letter  && pdflatex main && bibtex main && pdflatex main && pdflatex main
cd ../journal    && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Every SER in the tables comes from `figures/numbers.tex`, which `make_figures.py`
generates from the CSVs. For BPSK it applies two bookkeeping corrections to the
raw sweep output, uniformly to all receivers: the stored SER averages pilot words
in as zero-error (reported SER = stored x 125/120), and zero-error bounds use the
true 14,400 information bits per repetition rather than the CSV's nominal count.

## Before submission
- Replace the author block, affiliations and `[repository URL]`.
- Mamba2 is deliberately excluded (sweep incomplete).
