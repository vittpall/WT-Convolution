#!/usr/bin/env python3
"""Accuracy vs shot budget for emulated-quantum inference (results/emulation.json)."""
import json, os, collections, math
import numpy as np
R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'emulation.json')))
acc = collections.defaultdict(list); clean = collections.defaultdict(list); circ = {}
for k, v in R.items():
    if k.endswith('|circuits'):
        circ[k.split('|')[0]] = v; continue
    acc[(v['config'], v['noise'], v['shots'])].append(v['acc'])
    if v['noise'] == 'shots' and math.isinf(v['shots']): clean[v['config']].append(v['clean'])
cfgs = sorted({c for c, _, _ in acc}); shots = sorted({s for _, _, s in acc})
for noise in ('shots', 'gates'):
    print(f'\n=== {noise}: test accuracy (mean±std over seeds) vs shots per feature map per transform ===')
    print(f"{'config':12s} {'clean':>7s} " + ' '.join(f'{s:>11g}' for s in shots))
    for c in cfgs:
        cells = [f"{np.mean(acc[(c, noise, s)]):5.2f}±{np.std(acc[(c, noise, s)]):.2f}" if acc[(c, noise, s)] else '-' for s in shots]
        print(f"{c:12s} {np.mean(clean[c]):7.2f} " + ' '.join(f'{x:>11s}' for x in cells))
print('\ncircuits (per transform; blocks run in parallel or sequentially, shots split evenly):')
for c, info in circ.items():
    print(f'  {c:12s}', '; '.join(f"{i['map']}px: {i['qubits']}q x{i['blocks']}, {i['ncx']} CX, {i['channel']}" for i in info[::3]))
