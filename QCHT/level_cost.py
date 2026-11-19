import sys, numpy as np
sys.path.insert(0, '../CIFAR10/exp')
from qiskit import transpile
from qiskit.quantum_info import Operator
from haar_circuit import haar_circuit
from layers import haar_matrix as hm_levels
print('correctness vs exp/layers.haar_matrix(n, levels): max err')
for m in (3, 4):
    print(' ', m, [f'L{L}:{np.abs(Operator(haar_circuit(m, levels=L)).data - hm_levels(2**m, L)).max():.1e}' for L in (1, 2, 3)])
print('\n1D cost (cx / depth), basis {cx,rz,sx,x}, all-to-all')
print(f"{'N':>4s} " + ' '.join(f'{"L="+str(L):>12s}' for L in (1, 2, 3, 'full')))
for m in range(2, 7):
    row = []
    for L in (1, 2, 3, None):
        t = transpile(haar_circuit(m, levels=L), basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=2, seed_transpiler=0)
        row.append(f"{t.count_ops().get('cx', 0):5d}/{t.depth():<5d}")
    print(f'{2**m:4d} ' + ' '.join(f'{r:>12s}' for r in row))
