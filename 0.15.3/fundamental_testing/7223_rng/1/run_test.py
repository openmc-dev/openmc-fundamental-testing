# %%
import ctypes
from collections import Counter
from functools import lru_cache
from math import comb, factorial
from pathlib import Path

import numpy as np
from scipy import stats
from openmc.lib import _dll

ALPHA = 0.05
N_SAMPLES = 1_000_000
SEED = 123

_prn = _dll._ZN6openmc3prnEPm
_prn.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
_prn.restype = ctypes.c_double

# %%
class PCGRNG:
    def __init__(self, seed):
        self._seed = ctypes.c_uint64(seed)

    def next_float(self):
        return _prn(ctypes.byref(self._seed))

# %%
rng = PCGRNG(seed=SEED)
samples = np.fromiter((rng.next_float() for _ in range(N_SAMPLES)), dtype=float, count=N_SAMPLES)

# %%
N_BINS_FREQ = 100
bin_counts, _ = np.histogram(samples, bins=N_BINS_FREQ, range=(0, 1))
expected_count = N_SAMPLES / N_BINS_FREQ
chi2_freq = np.sum((bin_counts - expected_count)**2 / expected_count)
pval_freq = stats.chi2.sf(chi2_freq, N_BINS_FREQ - 1)
freq_pass = pval_freq > ALPHA

# %%
N_BINS_SERIAL = 10
hist_2d, _, _ = np.histogram2d(
    samples[:-1], samples[1:], bins=N_BINS_SERIAL, range=[[0, 1], [0, 1]],
)
expected_2d = (N_SAMPLES - 1) / N_BINS_SERIAL**2
chi2_serial = np.sum((hist_2d - expected_2d)**2 / expected_2d)
pval_serial = stats.chi2.sf(chi2_serial, N_BINS_SERIAL**2 - 1)
serial_pass = pval_serial > ALPHA

# %%
binary = samples >= np.median(samples)
n_runs = 1 + np.count_nonzero(binary[1:] != binary[:-1])
n_above = float(binary.sum())
n_below = float(len(binary) - n_above)
n_total = n_above + n_below
expected_runs = 2 * n_above * n_below / n_total + 1
var_runs = (2 * n_above * n_below / n_total) * (
    (2 * n_above * n_below - n_total) / (n_total * (n_total - 1))
)
z_runs = (n_runs - expected_runs) / np.sqrt(var_runs)
pval_runs = 2 * stats.norm.sf(abs(z_runs))
runs_pass = pval_runs > ALPHA

# %%
GAP_A, GAP_B = 0.0, 0.5
MAX_GAP = 10


def count_gaps(sequence, lower, upper):
    gaps = []
    current_gap = 0
    for in_range in (sequence >= lower) & (sequence < upper):
        if in_range:
            gaps.append(current_gap)
            current_gap = 0
        else:
            current_gap += 1
    return gaps[1:]


gaps = count_gaps(samples, GAP_A, GAP_B)
gap_counts = Counter(min(gap, MAX_GAP) for gap in gaps)
prob_in_range = GAP_B - GAP_A
expected_gaps = np.array([
    len(gaps) * (prob_in_range * (1 - prob_in_range)**gap
                 if gap < MAX_GAP else (1 - prob_in_range)**MAX_GAP)
    for gap in range(MAX_GAP + 1)
])
observed_gaps = np.array([gap_counts.get(gap, 0) for gap in range(MAX_GAP + 1)])
valid_gaps = expected_gaps > 5
chi2_gap = np.sum((observed_gaps[valid_gaps] - expected_gaps[valid_gaps])**2 / expected_gaps[valid_gaps])
pval_gap = stats.chi2.sf(chi2_gap, valid_gaps.sum() - 1)
gap_pass = pval_gap > ALPHA

# %%
D_TUPLE = 5
N_CATEGORIES = 10
digits = np.minimum((samples * N_CATEGORIES).astype(int), N_CATEGORIES - 1)
tuples = digits[:len(digits) // D_TUPLE * D_TUPLE].reshape(-1, D_TUPLE)
distinct_counts = np.sum(np.diff(np.sort(tuples, axis=1), axis=1) != 0, axis=1) + 1


@lru_cache(maxsize=None)
def stirling2(size, groups):
    if size == groups == 0:
        return 1
    if size == 0 or groups == 0:
        return 0
    return groups * stirling2(size - 1, groups) + stirling2(size - 1, groups - 1)


poker_counts = Counter(distinct_counts)
observed_poker = np.array([poker_counts.get(groups, 0) for groups in range(1, D_TUPLE + 1)])
expected_poker = np.array([
    len(tuples) * stirling2(D_TUPLE, groups) * comb(N_CATEGORIES, groups)
    * factorial(groups) / N_CATEGORIES**D_TUPLE
    for groups in range(1, D_TUPLE + 1)
])
valid_poker = expected_poker > 5
chi2_poker = np.sum((observed_poker[valid_poker] - expected_poker[valid_poker])**2 / expected_poker[valid_poker])
pval_poker = stats.chi2.sf(chi2_poker, valid_poker.sum() - 1)
poker_pass = pval_poker > ALPHA

# %%
MAX_LAG = 10
autocorr_results = []
for lag in range(1, MAX_LAG + 1):
    correlation = np.corrcoef(samples[:-lag], samples[lag:])[0, 1]
    pvalue = 2 * stats.norm.sf(abs(correlation * np.sqrt(len(samples) - lag)))
    autocorr_results.append((lag, pvalue))
autocorr_pass = all(pvalue > ALPHA / MAX_LAG for _, pvalue in autocorr_results)

# %%
tests = [
    ('Frequency', pval_freq, freq_pass),
    ('Serial', pval_serial, serial_pass),
    ('Runs', pval_runs, runs_pass),
    ('Gap', pval_gap, gap_pass),
    ('Poker', pval_poker, poker_pass),
    ('Autocorrelation', min(result[-1] for result in autocorr_results), autocorr_pass),
]
overall_pass = all(passed for _, _, passed in tests)
report_lines = [f'{name:<20} p={pvalue:.6g} {"PASS" if passed else "FAIL"}'
                for name, pvalue, passed in tests]
report_lines.append(f'OVERALL: {"PASS" if overall_pass else "FAIL"}')
Path('knuth_report.txt').write_text('\n'.join(report_lines) + '\n')
Path('results.txt').write_text('PASS\n' if overall_pass else 'FAIL\n')


