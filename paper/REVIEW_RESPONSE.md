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
  - No receiver's SER changes by more than its 95% CI.
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
- **Code availability:** now points to https://github.com/Gilzuk/viterbitransformed, which is public.
- **"Manuscript received" date:** removed, because the editor assigns it.
