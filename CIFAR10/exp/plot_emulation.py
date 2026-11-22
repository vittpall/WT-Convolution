#!/usr/bin/env python3
"""Accuracy vs shot budget (gate noise on) for the DSP 2027 paper figure."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'emulation.json')))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'emulation_shots.pdf')

shots = [1e4, 1e5, 1e6, 1e7, 1e8]
lines = [  # config, label, color, style
    ('ht_p3', 'HT, global (10q)', '#7f7f7f', 's--'),
    ('haar_p3', 'Haar, full (10q)', '#000000', 'o--'),
    ('haar_p3_L3', 'Haar $L{=}3$ (6q, 28 CX)', '#d62728', 'o-'),
    ('ht_p3_L3', 'block-HT $L{=}3$ (6q, 0 CX)', '#d62728', 's:'),
    ('haar_p3_L2', 'Haar $L{=}2$ (4q, 2 CX)', '#1f77b4', 'o-'),
    ('ht_p3_L2', 'block-HT $L{=}2$ (4q, 0 CX)', '#1f77b4', 's:'),
    ('haar_p3_L1', 'Haar $L{=}1$ (2q, 0 CX)', '#2ca02c', 'o-'),
]
plt.rcParams.update({'font.size': 7, 'font.family': 'serif'})
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for cfg, lab, c, st in lines:
    m = [np.mean([v['acc'] for k, v in R.items() if k.startswith(cfg + '|') and isinstance(v, dict)
                  and v.get('noise') == 'gates' and v.get('shots') == s]) for s in shots]
    ax.plot(shots, m, st, color=c, label=lab, ms=3, lw=1)
ax.set_xscale('log'); ax.set_ylim(0, 95)
ax.set_xlabel('measurement shots per feature map per transform')
ax.set_ylabel('CIFAR-10 test accuracy (%)')
ax.grid(alpha=.3, lw=.5); ax.legend(fontsize=5.5, loc='upper center', bbox_to_anchor=(0.45, -0.25), ncol=2, frameon=False)
fig.tight_layout(pad=0.2); fig.savefig(out); print('saved', out)
