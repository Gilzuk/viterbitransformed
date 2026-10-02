# Response to the internal review of the Communications Letter

Every change applies to both the letter (`letter/main.tex`) and the journal
paper (`journal/main.tex`) unless stated. Numbers come from the files named;
all can be regenerated.

## 1. Pilot count
**Reviewer:** the abstract says "one pilot word per frame", while the system model has 5 per frame.
**Verified:** correct, the abstract was wrong. `pilots_num=25` means one pilot every 25 words, so 5 per 125-word frame.
**Change:** the abstract now says "a pilot word every 25 words". The journal protocol states "one in every 25 words".

## 2. The adaptation gate uses ground truth
**Reviewer:** "decoded SER <= 0.02" needs the transmitted bits.
**Verified:** correct. `Code/trainer.py` (`online_evaluation`) computes the SER against the transmitted word. When 0 < SER <= 0.02 it also adapts on the raw hard decisions, not the re-encoded word. This is the gate of the reference ViterbiNet implementation. It is applied identically to every adaptive receiver: LS-Viterbi runs through the same loop, and only `online_training` differs.
**Changes:**
- Both papers now describe the gate exactly and call it an oracle.
- I added an implementable gate, `gate_mode='rs'` in `Code/trainer.py`; the default is unchanged, so no existing result moves.
  - A word is accepted when the re-encoded decoded word differs from the hard decisions in at most one RS symbol. That is bounded-distance decoding for two parity symbols.
  - Adaptation always uses the re-encoded word.
- `experiments/gate_check.py` re-runs LS-Viterbi, ViterbiNet K=5, VNet-affine and ViterbiNet K=200 under both gates at 7, 10 and 12 dB. All runs use the same offline weights and the same draws; results are in `Results/metrics/gate_check.csv`.
  - The implementable gate accepts fewer words at 7 dB (33% vs 44% for LS-Viterbi) but labels them better (15% vs 37% of accepted words carry any label error).
  - No receiver's SER changes by more than its marginal 95% CI. (Superseded in round 2: one paired difference is significant; see below.)
  - LS-Viterbi keeps the lowest SER under both gates.
- New journal subsection "An implementable adaptation gate" with Table `tab:gate`, plus one paragraph in the letter.

## 3. The 0.126 dB estimation loss is an approximation
**Reviewer:** (sigma^2/T) I assumes nearly orthogonal columns.
**Verified:**
- I computed s^T (X^T X)^-1 s over 2,000 RS-encoded words (the repo encoder) and all 16 states: 0.0301, against L/T = 0.0294.
- The loss is therefore 0.129 dB exact vs 0.126 dB approximate.
- The median condition number of X^T X / T is 1.39.

**Change:** both papers now state the general covariance, the orthogonality assumption, and this check. The equation is written with ≈.

## 4. "Same Monte Carlo draws" is not supported
**Reviewer:** repetition counts differ (10,000 / 1,000 / 50).
**Verified:** partly correct.
- Receivers run in this container read repetition r < 200 from one shared on-disk cache (`Code/channel/data_cache.py`), so their first repetitions are paired. This covers perfect CSI, LS-Viterbi, ViterbiNet K=5, VNet-affine and Mamba2.
- Their per-repetition SERs correlate positively (median about 0.5).
- ViterbiNet K=200 and the Transformer ran on Colab with independent draws. The noisy-CSI sweeps used a separate cache.

**Changes:**
- Removed "all measured on the same Monte Carlo draws" and "identical Monte Carlo draws".
- The protocol now says which receivers are paired and which are not.
- New `experiments/paired_stats.py` writes `Results/metrics/paired_ls_vs_learned.csv`: paired LS-minus-learned differences with 95% CIs over the shared repetitions.
- New journal Table `tab:paired`.
- The K=5 vs K=200 comparison is now labelled unpaired.

## 5. High-SNR rankings are weakly resolved
**Verified:** correct. On the shared draws:
- LS-Viterbi is significantly better than ViterbiNet K=5 at 1, 6, 7 and 9-12 dB.
- It is significantly better than VNet-affine at 0, 2-7 and 11 dB.
- It is never significantly worse.
- At 13-14 dB the paired differences are within noise. At 13 dB, LS-Viterbi is even slightly (not significantly) worse than VNet-affine: +6.9e-6 ± 1.1e-5.

**Change:** "never worse / beats at every SNR" is now "lowest point-estimate SER", plus the paired statements above. 14 dB ratios are called indicative, with the 95% Poisson interval for 4 events (0.3-2.6 times the estimate). The 14 dB SER numbers were removed from the letter abstract.

## 6. Data-cache bug provenance
**Verified:**
- The defect was fixed in `7011367` (2026-09-04): it replayed one draw across all repetitions.
- `4865013` then emptied `mc_sweep_validation.csv` to its header, so every row now in the file was produced after the fix. This includes the Colab ViterbiNet K=200 and Transformer rows, imported on 2026-09-26.
- As an independent check, the defect's signature was zero variance between repetitions. All 105 rows the papers use (0-14 dB) have a standard deviation of at least 1.9% of the mean.

**Change:** a provenance statement in both papers.

## Wording and blockers
- "smallest learned metric" is now "smallest free per-state affine metric", noting that a tap-parameterized metric (as in LS-Viterbi) is smaller.
- **Author block:** now filled in: Gil Zukerman, School of Electrical Engineering, Tel Aviv University (gilzukerman@mail.tau.ac.il).
- **Code availability:** points to https://github.com/Gilzuk/viterbinet-revisited, a public repository with the curated code, data and paper sources.
- **"Manuscript received" date:** removed, because the editor assigns it.

---

# Round 2: independent verification (on commit 40080746)

An independent check of the round-1 fixes found most of them correct. It also raised a blocker, new problems and overstatements. Each item and the action taken:

## Blockers
- **B1, code URL 404.** The repository `Gilzuk/viterbinet-revisited` was published after the checked commit and now resolves publicly. This document previously named the wrong repository; corrected.
- **B2, journal abstract over 250 words.** Rewritten for the fixes below at 346 words, with every highlight kept as the author asked. Still over the IEEE 150-250 guideline (resolved in round 3).

## Major
- **M1, "best learned receiver at most SNRs".** The journal abstract and Table I now name each comparator and its SNRs: ViterbiNet K=5 at 1, 6, 7 and 9-12 dB; VNet-affine at 0, 2-7 and 11 dB. "Never significantly worse" is limited to the available paired comparisons. K=200 and the Transformer are marked unpaired.
- **M2, gate robustness overinterpreted.** New paired RS-minus-oracle differences: `experiments/paired_stats.py` writes `Results/metrics/gate_paired_diff.csv`, and Table V has a new column.
  - VNet-affine at 7 dB is significant: +7.7e-4 ± 5.6e-4.
  - The other 11 differences are not resolved.
  - The text now claims only that LS-Viterbi keeps the lowest point-estimate SER at 7, 10 and 12 dB. "The two effects cancel" and "conclusions hold" are removed.
- **M3, normal instead of Student-t intervals.** Paired intervals are now pointwise paired Student-t, with pairing capped at the 200 shared repetitions. No verdict changes. Tables and text say "pointwise paired Student-t".
- **M4, universal-superiority wording in the conclusions.** Removed "reaches the bound", "can at best match it" and "matched or outperformed every learned detector". They are replaced by the measured point-estimate and paired results, with 13-14 dB unresolved.
- **M5, incomplete description of prior work.** Both papers now say that the 2020 paper also uses an initial-CSI baseline under block fading, compares learned detectors and covers non-Gaussian channels, and that the follow-ups compare adaptation methods. Only the Gaussian-setting classical baselines are revisited.
- **M6, "reference implementation" attribution.** Changed to "the gate of the code base this study builds on". K=200 is attributed to [raviv2023online], noting that it uses mini-batches of 64 while we use full-word updates. *Not verified here:* the claim that the published protocol gates on receiver-side re-encoding. It is not asserted in the papers.
- **M7, 100-58 network.** Both papers state that the original uses 100-50 and that 58 is this study's choice.
- **M8, Table I attention row.** Now reads "Attention-based architectures [vaswani, dosovitskiy], applied here as branch metrics".
- **M9, causal take-home sentence.** Now "In this experiment, the tested learned receivers show no established advantage over a classical receiver given the same information". Both papers note that meta-learned variants were not tested.
- **M10, Poisson "events".** Now "bit errors", noted to be clustered in one or two repetitions; the Poisson and 3/N bounds are described as treating them as more independent than they are.
- **M11, wrong raw bit and error metadata.**
  - `run_mc_sweep.py` now reports information bits (120 × 120 per repetition). The stopping budget keeps its old unit, so published runs reproduce.
  - `experiments/fix_sweep_bit_metadata.py` rewrote `bits_run` and `errors_observed` in `mc_sweep_validation.csv` (and `bits_run` in the topology and training-budget CSVs).
  - SER columns are untouched. LS-Viterbi at 14 dB now reads 14,400,000 bits and 4 errors.
- **M12, LS-loss check not reproducible; assumptions incomplete.** Added the seeded `experiments/ls_loss_check.py` and its output `Results/metrics/ls_loss_check.json`: 0.03009 vs L/T 0.02941, median condition number 1.394. The papers now state the independent-noise and unchanged-taps assumptions, and call the SNR-shift reading a heuristic.
- **M13, "Transformer worst at every SNR" is false.** Corrected: at 12 and 13 dB the Transformer is slightly below ViterbiNet K=200.

## Minor
- **m1:** 1 dB added to the letter abstract.
- **m2:** "meet the measured detection-plus-adaptation budget" with 1.4% headroom; RS decoding and gating are not timed.
- **m3:** "same pilots and acceptance/labeling rule, applied to each receiver's own decisions" everywhere, including captions and Fig. 2.
- **m4:** "20-50 repetitions".
- **m5:** provenance variance statement limited to the 105 nonzero BPSK points.
- **m6:** "about 3-30%".
- **m7:** the 125/120 pilot correction is now explained in both papers.
- **m8:** pairing capped at 200.
- **m9:** "missing or incomplete per-repetition record".
- **m10:** "Adam updates", and "40 times fewer updates" instead of "40 times cheaper".
- **m11:** "every no-CSI adaptive receiver within 1.5×".

## Result
- Letter: 4 pages, abstract 246 words.
- Journal: 12 pages, abstract 346 words.
- Both compile with no overfull boxes and no unresolved references. `numbers.tex` (every SER in the papers) is unchanged.

---

# Round 3: second verification (on commit 260d8811)

The round-2 numerical fixes reproduce: all 28 paired comparisons, the gate difference, the LS-loss check and the 105-row provenance check. Remaining items and actions:

- **Blocker, journal abstract over 250 words.** Condensed to 247 words (PDF count; 244 in the source) without dropping any highlight: matched LS baseline, comparator-specific significance, 3-30% / 0.13 dB, implementable gate, adaptation budget, affine metric and attention metrics, per-word time, QPSK tying, fast fading, take-home, code/data/design guidance. The letter abstract is 247 words in the PDF. The 346-word version is in git history (commit 260d8811).
- **"In every setting studied ... none had a lower SER".** Now limited to the BPSK SNR sweep. Under fast fading the learned receivers do beat last-word LS, but not LS with exponential forgetting (journal Discussion and Conclusion).
- **Label errors and channel jumps "have little effect".** Removed from both papers. The measurements are now described as compatible with the heuristic, without attributing the remaining gap.
- **"No receiver without CSI tracks within a word".** Now: none of the tested receivers without CSI approaches the per-sample genie, including PSP-LMS, which updates its taps at every trellis step (journal abstract, contributions, Table I, Section on fast fading).
- **Affine "too little capacity to overfit".** Replaced by the observation: its SER kept falling up to K = 100-200, which does not show that it cannot overfit.
- **Gate K=200 initialization.** Both papers and the `gate_check.py` docstring now say that the K=200 gate runs start from the K=5 sweep's offline weights, not the separately trained K=200 sweep's.
- **fast_fading.csv bit metadata.** Repaired by `fix_sweep_bit_metadata.py`: 54 rows, 50 repetitions = 720,000 information bits. SER untouched.
- **Sweep logs in budget units.** `run_mc_sweep.py` now labels the planning quantities as the legacy budget unit and logs true information-bit and bit-error counts.
- **"same pilots and decisions" (design guidance).** Now "same pilots and acceptance rule, applied to its own decisions".
- **LS 11.4 ms.** Both papers now state that it is the measured trellis time plus a separately timed 0.02 ms LS solve.
- **Fig. 18 overlap.** The legend moved to the right panel, the annotation is wrapped onto three lines, and the x-labels are shortened; re-rendered.
- **Release README "RS(15,13)".** Corrected to shortened RS(17,15) over GF(2^8).
- **Pin the release revision.** Suggest tagging the paper repository at submission (e.g. `v1.0-submission`) and citing the tag; not done yet (author decision).

---

# Primary-source check of the cited ViterbiNet paper (arXiv 1905.10750v2)

- **Original widths 100 and 50:** confirmed (p. 16: "a 1x100 layer followed by a 100x50 layer and a 50x16 layer, using intermediate sigmoid and ReLU").
- **Initial-CSI baseline under block fading:** confirmed (p. 23; "Viterbi, initial CSI" in Figs. 12-13).
- **Learned-detector comparison (SBRNN) and non-Gaussian channels (Poisson, alpha-stable):** confirmed (pp. 16-21).
- **Added sentence on the published rule (both papers).** Algorithm 2 (pp. 13-14) retrains when the estimated number of decoding errors is below a threshold (2%, p. 23) and always retrains on the re-encoded word. Our implementable RS gate follows that labeling. The oracle gate of our code base departs from it by using raw hard decisions when the decoded SER is nonzero.
- **Still to check against its source:** "200 Adam updates per word with mini-batches of 64" [Raviv et al., 2023].

# Primary-source check of Meta-ViterbiNet (arXiv 2103.13483v1)

- **"Meta-learning to adapt faster" and "the follow-up work compares adaptation methods":** confirmed. It compares joint, online and online meta-learning training, for ViterbiNet and an LSTM detector (p. 4).
- **No classical Viterbi baseline** appears in its evaluation. This is consistent with our statement that these evaluations compare against Viterbi with perfect/initial/corrupted CSI "or against other learned receivers".
- **Network:** 1x100, 100x50, 50xM^L with sigmoid, ReLU and softmax (p. 4). This confirms the original 100-50 widths a second time.
- **Gate:** re-trains on blocks whose decoding is correct "as determined by error detection", using the re-encoded word (p. 3). This agrees with our description of the published labeling.
- **Simulation setup:** 136-symbol blocks, RS[17,15] with two parity symbols, one pilot block followed by 24 data blocks, L = 4, SNR = 1/sigma^2, COST 2100 taps (p. 5). Our setup follows it.
- **Fix:** both papers now attribute this setup to Meta-ViterbiNet.

# Primary-source check of Raviv et al. 2023 (arXiv 2203.14359)

- **"200 Adam updates per word with mini-batches of 64":** confirmed (p. 20: "All training methods use the Adam optimizer with Isgd = 200 iterations and learning rate 10^-3 ... Both meta-learning and online learning employ a batch size of 64 symbols").
- **Network 1x100, 100x50, 50x|S|^L:** confirmed (p. 19).
- **Gate:** "re-training occurs if the normalized bits difference between the re-encoded word and the hard-decision of the channel word is smaller than the threshold of 0.02" (pp. 20-21), with re-encoded labels (p. 8). This is a receiver-side test close to our RS gate. Both papers now say so, and that the oracle gate of our code base departs from both published rules.
- **Setup:** B = 136, 120 information bits, RS[17,15], L = 4 (p. 21). Both papers now attribute the setup to [raviv2021metaviterbinet, raviv2023online].

All statements attributed to the three cited ViterbiNet papers have now been checked against the papers themselves.
