# Prompt: independent verification of the post-review fixes

Copy everything below the line into another LLM. Attach, or give it access to:
- the letter (`paper/letter/main.tex` and `main.pdf`);
- the journal (`paper/journal/main.tex` and `main.pdf`);
- `paper/REVIEW_RESPONSE.md`;
- the data files named in the prompt.

Repository: https://github.com/Gilzuk/viterbitransformed, branch `claude/transformer-sionna-mlp-comparison-wc67zp`.

---

You are an independent, skeptical reviewer for an IEEE Communications Letter and its companion journal paper (IEEE TMLCN). An internal review raised six problems and a list of submission blockers. The author claims to have fixed all of them. Your job is to check each claimed fix against the manuscripts, the code and the data. Do not trust the author's response document; use it only as a list of claims to test.

## Materials
- **Letter:** `paper/letter/main.tex` / `main.pdf`, "ViterbiNet Revisited: A Matched-Information Classical Baseline for Learned Trellis Detection".
- **Journal:** `paper/journal/main.tex` / `main.pdf`, "When Does Learning Help Trellis Detection? Matched Baselines, Structure, and Complexity of Model-Based Deep Viterbi Receivers".
- **Author's response:** `paper/REVIEW_RESPONSE.md` (claims only).
- **Code and data**, if available:
  - `Code/trainer.py` (`online_evaluation`, `gate_mode`);
  - `Code/channel/data_cache.py`;
  - `experiments/gate_check.py` and `experiments/paired_stats.py`;
  - `paper/make_figures.py`;
  - `Results/metrics/mc_sweep_validation.csv`;
  - `Results/metrics/paired_ls_vs_learned.csv`;
  - `Results/metrics/gate_check.csv`;
  - `Results/metrics/.mc_sweep_checkpoints/*.json`;
  - `paper/figures/numbers.tex`, `paired_table.tex` and `gate_table.tex`.

## The original review findings and the claimed fixes

Check each item and give a verdict: **FIXED**, **PARTIALLY FIXED**, **NOT FIXED**, or **NEW PROBLEM INTRODUCED**. Quote the exact sentence or line you relied on, with the file and section.

### 1. Pilot count
- **Original:** the abstract said "one pilot word per frame", but the system model has 5 pilots per 125-word frame.
- **Claimed fix:**
  - Both papers say "a pilot word every 25 words", which is 5 per frame.
  - The code uses `pilots_num=25`, meaning one pilot every 25 words.
- **Verify:**
  - Every mention of pilots in both papers (abstract, system model, receivers, protocol, figure captions and tables) is consistent with 5 pilots and 120 data words per frame.
  - If the code is available, check that it matches.

### 2. The adaptation gate uses ground truth
- **Original:** "decoded SER ≤ 0.02" needs the transmitted bits, so the LS baseline may use information a real receiver lacks.
- **Claimed fix:**
  - Both papers now describe the gate exactly and call it an oracle, the reference ViterbiNet implementation's rule.
  - The oracle gate is applied identically to every adaptive receiver, LS-Viterbi included.
  - A new implementable gate (`gate_mode='rs'`) accepts a word when the re-encoded decoded word differs from the hard decisions in at most one RS symbol, and always adapts on the re-encoded word.
  - Under both gates, at 7, 10 and 12 dB, no receiver's SER changes by more than its 95% CI, and LS-Viterbi keeps the lowest SER.
  - At 7 dB the RS gate accepts fewer words (about 33% vs 44% for LS-Viterbi) but with fewer label errors (about 15% vs 37%).
- **Verify:**
  - (a) Read `online_evaluation` in `Code/trainer.py`:
    - Does the oracle gate really compare with the transmitted word?
    - Does it adapt on the raw hard decisions when 0 < SER ≤ 0.02?
    - Does the RS gate use only receiver-side information, i.e. never `transmitted_word` for the decision? The label-error bookkeeping may use it; the decision must not.
    - Is the default still `'oracle'`, so that earlier results are unchanged?
  - (b) Recompute, from `gate_check.csv`, the claim that no SER moves beyond its CI, for every receiver and SNR.
  - (c) Check that LS-Viterbi has the lowest SER under both gates at each SNR.
  - (d) Check that the numbers in Table `tab:gate` (journal) and in the letter's gate sentence match the CSV.
  - (e) Is "bounded-distance decoding for two parity symbols" (corrects one symbol) described correctly?

### 3. The 0.126 dB estimation-loss argument
- **Original:** Δ ≈ 10·log10(1+L/T) assumes an estimator covariance of (σ²/T)·I, i.e. nearly orthogonal regressors. The assumption was not stated or checked.
- **Claimed fix:**
  - Both papers give the general covariance σ²(XᵀX)⁻¹ and state the orthogonality assumption.
  - They report a check on 2,000 RS-encoded words: sᵀ(XᵀX)⁻¹s = 0.0301 vs L/T = 0.0294, which gives 0.129 dB exact vs 0.126 dB approximate.
  - The median condition number of XᵀX/T is 1.39.
  - The equation uses ≈.
- **Verify:**
  - The algebra: does 10·log10(1+4/136) equal 0.126 dB, and does 10·log10(1.0301) equal 0.129 dB?
  - The assumption is stated where the formula appears.
  - The abstract's "0.13 dB" is consistent with this.
  - The papers do not over-claim what "consistent with the predicted loss" means for the 3–30% gap to perfect CSI.

### 4. "Same Monte Carlo draws" was not supported
- **Original:** repetition counts differ between receivers (e.g. 10,000 / 1,000 / 50), so the draws cannot all be the same.
- **Claimed fix:**
  - The phrases "all measured on the same Monte Carlo draws" and "identical Monte Carlo draws" were removed.
  - The protocol now says:
    - perfect CSI, LS-Viterbi, ViterbiNet K=5 and VNet-affine share draws for repetitions r < 200 through an on-disk cache;
    - ViterbiNet K=200 and the Transformer ran on Colab with independent draws;
    - the noisy-CSI sweeps used a separate cache.
  - Paired differences with 95% CIs are computed only over shared repetitions (journal Table `tab:paired`).
  - K=5 vs K=200 is labelled unpaired.
- **Verify:**
  - (a) Search both papers for any remaining claim of identical or same draws that is broader than the true pairing.
  - (b) Check `Code/channel/data_cache.py`: does its cache key actually make repetitions r < 200 identical across receivers?
  - (c) Check that `paired_stats.py` pairs only when the checkpoint repetition count equals the CSV n_reps. Is a paired t-interval on per-repetition means appropriate, and is it computed correctly?
  - (d) Check that every entry of Table `tab:paired` matches `paired_ls_vs_learned.csv`.
  - (e) Check that no comparison involving K=200 or the Transformer is presented as paired.

### 5. High-SNR rankings were weakly resolved
- **Original:** claims such as "never worse" or "beats at every SNR" are not supported at 13–14 dB, where only a few errors are observed.
- **Claimed fix:**
  - Claims are now "lowest point-estimate SER" plus paired statements:
    - LS-Viterbi is significantly better than ViterbiNet K=5 at 1, 6, 7 and 9–12 dB;
    - it is significantly better than VNet-affine at 0, 2–7 and 11 dB;
    - it is never significantly worse;
    - 13–14 dB is not resolved, and at 13 dB LS-Viterbi is even slightly (not significantly) worse than VNet-affine.
  - 14 dB ratios are called indicative, with a Poisson interval.
- **Verify:**
  - (a) Recompute these significance statements from `paired_ls_vs_learned.csv` and compare them with the text of the abstract, introduction, results and conclusion of both papers. The letter abstract says "significantly better than the best ViterbiNet configuration at 6, 7, and 9–12 dB". Is that consistent with the CSV, which also shows significance at 1 dB?
  - (b) Flag any remaining sentence that implies superiority at 13–14 dB.
  - (c) Check the "never significantly worse" claim against every row.
  - (d) Check the Poisson interval quoted for 4 events (claimed 0.3–2.6 times the estimate).

### 6. Data-cache bug provenance
- **Original:** the repository's own sweep script mentioned a data-cache bug, which raises doubt about whether the reported rows predate the fix.
- **Claimed fix:**
  - The defect (one draw replayed across all repetitions) was fixed in commit `7011367` (2026-09-04).
  - Commit `4865013` then emptied `mc_sweep_validation.csv` to its header, so every row now in the file was produced after the fix.
  - Independent check: the defect's signature is zero variance between repetitions, and all 105 rows the papers use (0–14 dB) have std/mean ≥ 1.9%.
  - Both papers carry a provenance statement.
- **Verify:**
  - (a) If git history is available, confirm the commit dates and that `4865013` truncated the CSV.
  - (b) Recompute the minimum std/mean over the rows the papers use.
  - (c) Check that the provenance statement in each paper is accurate and not overstated.

### 7. Wording and submission blockers
- **Claimed fixes:**
  - "smallest learned metric" is now "smallest free per-state affine metric", with a note that a tap-parameterized metric such as LS-Viterbi's is smaller.
  - The author block is now Gil Zukerman, School of Electrical Engineering, Tel Aviv University, gilzukerman@mail.tau.ac.il.
  - The "Manuscript received" line was removed (the editor assigns it).
  - The code-availability URL is https://github.com/Gilzuk/viterbinet-revisited.
- **Verify:**
  - No placeholder text remains anywhere, e.g. "Author Name", "Institution", "Month DD", "[repository URL]", "TODO", "XX".
  - With a single author, the text never says "the authors".
  - The repository URL resolves and contains the code and data claimed.
  - The AI-use statement is present.

## Changes made after the review, also to check
- **Letter introduction:** the contributions are now framed against prior evaluations of ViterbiNet and its meta-learned extensions.
- **Journal Table I** ("What this paper adds to prior evaluations"): each row pairs a prior claim or design choice with this paper's finding.
- **Both abstracts** now end with a take-home sentence: on this channel, ViterbiNet's reported gain over classical detection comes from the baselines it was compared against, not from learning.

For these, verify:
- (a) Every statement attributed to prior work (Shlezinger et al. 2020, TWC; Raviv et al. 2021, ICC Workshops; Raviv et al. 2023, TWC) is an accurate description of those papers. Flag anything you cannot confirm from the cited papers themselves.
- (b) No unsupported "first" or "only" claims are made, and the hedge "that we are aware of" is kept.
- (c) The take-home sentence is scoped to the channel studied (linear Gaussian, 4 taps, COST 2100) and does not generalize beyond the evidence.
- (d) Every number in Table I matches the body and the data.
- (e) **Known inconsistency to confirm:**
  - The journal abstract says LS-Viterbi and ViterbiNet K=5 "come close to" the real-time budget.
  - Journal Table I, the journal latency section and the letter abstract say they "meet" it: 11.4 ms and 16.8 ms against a 17 ms word.
  - Report which wording the data supports.
- (f) Abstract lengths: the letter's is 250 words and the journal's 349. Report them against the venues' limits, but do not propose removing content.

## Cross-cutting consistency checks
- Every number that appears in both papers must agree: SERs, ratios, dB values, parameter counts (7,002; 32; 6,960), latencies, "40 times cheaper", and "3–30% of the perfect-CSI SER".
- BPSK bookkeeping: the reported SER is the stored SER × 125/120, and zero-error bounds use 14,400 bits per repetition. Check that this is applied uniformly and stated.
- Results are reported to 14 dB, and SNR plots stop at 15 dB, for both BPSK SNR and QPSK Es/N0. Flag any figure or text that goes beyond this.
- Check that every figure and table is referenced in the text and every label resolves (no "??").

## Output format
1. **Summary table** with columns: Item | Verdict | Evidence (file/section + quote) | Required action.
2. **List of every discrepancy** found. For each: severity (blocker / major / minor), exact location, what the manuscript says, what the evidence says, and a proposed one-line fix.
3. **Final judgment:** is each paper ready to submit? If not, list the blocking items.

Be concrete. Do not restate claims you have not checked. If you could not verify something because a file was missing, say so explicitly instead of assuming it is fixed.
