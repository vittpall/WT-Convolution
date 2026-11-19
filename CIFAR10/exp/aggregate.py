#!/usr/bin/env python3
"""Summarize results/*.json: mean +- std of final-epoch test acc, Welch t-test Haar vs HT."""
import glob, json, os, collections
import numpy as np
from scipy import stats
R = collections.defaultdict(list)
for f in glob.glob(os.path.join(os.path.dirname(__file__), 'results', '*.json')):
    d = json.load(open(f)); R[d['config']].append((d['seed'], d['final_acc'], d['best_acc_ref'], d['params']))
print(f"{'config':14s} {'n':>2s} {'params':>8s} {'final acc':>14s} {'best(ref)':>14s}")
for c in sorted(R):
    a = np.array([r[1] for r in R[c]]); b = np.array([r[2] for r in R[c]])
    print(f"{c:14s} {len(a):2d} {R[c][0][3]:8,d} {a.mean():7.2f}±{a.std(ddof=1) if len(a)>1 else 0:.2f} {b.mean():8.2f}±{b.std(ddof=1) if len(b)>1 else 0:.2f}")
print('\nWelch t-test on final acc (Haar - HT):')
pairs = [(f'haar_p{p}', f'ht_p{p}') for p in (1, 2, 3)] + [('haar_p3_res', 'ht_p3_res'), ('haar_p3', 'htpan_p3')] \
      + [(f'haar_p3_L{L}', 'ht_p3') for L in (1, 2, 3)] \
      + [('lhaar_p3', 'htpan_p3'), ('lhaar_p3_res', 'htpan_p3'), ('lhaarpos_p3', 'htpan_p3'), ('lhaarpos_p3_res', 'htpan_p3'),
         ('lhaar_p3', 'haar_p3'), ('lhaar_p3_res', 'haar_p3_res'), ('lhaarpos_p3', 'haar_p3'), ('lhaarpos_p3_res', 'haar_p3_res')]
for h, t in pairs:
    if len(R[h]) > 1 and len(R[t]) > 1:
        a = [r[1] for r in R[h]]; b = [r[1] for r in R[t]]
        tt = stats.ttest_ind(a, b, equal_var=False)
        print(f"  {h:12s} vs {t:9s}  diff {np.mean(a)-np.mean(b):+.2f}  p={tt.pvalue:.3g}")
