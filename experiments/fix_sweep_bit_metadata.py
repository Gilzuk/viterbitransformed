"""One-off repair of the bit bookkeeping in the BPSK sweep CSVs, including the fast-fading study
(SER values are not touched).

Before the fix in run_mc_sweep.py, bits_run counted a nominal 16 bits per word (the two RS
parity symbols, 8 * n_symbols) over all 125 words of a repetition: 2,000 bits per repetition.
A repetition actually detects 120 data words x 120 information bits = 14,400 bits, and the
stored ser_mean averages the per-word bit error rate over all 125 words with pilots as 0, so
    bits_run        = n_reps * 14,400
    errors_observed = ser_mean * n_reps * 125 * 120     (= reported SER * bits_run, with SER = ser_mean * 125/120)
Only rows still carrying the old nominal count (bits_run == n_reps * 2,000) are rewritten, so the
script is idempotent.

    python3 experiments/fix_sweep_bit_metadata.py
"""
import csv, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET = os.path.join(ROOT, 'Results', 'metrics')
OLD_PER_REP, INFO_PER_REP, WORD_BITS = 125 * 16, 120 * 120, 125 * 120

for name in ('mc_sweep_validation.csv', 'vnet_topology_study_snr7.csv', 'minibatch_search_snr7.csv', 'fast_fading.csv'):
    path = os.path.join(MET, name)
    rows = list(csv.DictReader(open(path)))
    fields = list(rows[0].keys())
    fixed = 0
    for r in rows:
        n = int(r['n_reps'])
        if int(float(r['bits_run'])) != n * OLD_PER_REP:
            continue
        r['bits_run'] = str(n * INFO_PER_REP)
        if 'errors_observed' in r:
            r['errors_observed'] = repr(float(r['ser_mean']) * n * WORD_BITS)
        fixed += 1
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader(); w.writerows(rows)
    print(f'{name}: {fixed} of {len(rows)} rows rewritten')
