"""Check the emulator's global-depolarizing + mitigation model against qiskit-Aer density-matrix
simulation with per-gate depolarizing noise (e1 on 1q gates, e2 on CX). Metric: relative L2 error of
the recovered (signed) coefficients vs exact, infinite shots, CIFAR patches with the W^T 1 offset."""
import numpy as np, sys
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from haar_circuit import haar_circuit, haar_matrix
from sign_recovery import cifar_patches
e1, e2 = 2.5e-4, 3e-3
nm = NoiseModel()
nm.add_all_qubit_quantum_error(depolarizing_error(e1, 1), ['sx', 'x', 'rz'])
nm.add_all_qubit_quantum_error(depolarizing_error(e2, 2), ['cx'])
sim = AerSimulator(method='density_matrix', noise_model=nm)
print(f"{'circuit':24s} {'qubits':>6s} {'CX':>4s} {'lambda':>7s} | rel.err: {'unmitigated':>11s} {'global-mitig':>12s}")
for name, kind, m in (('Haar 4x4 block (L=2)', 'haar', 2), ('Haar 8x8 block (L=3)', 'haar', 3),
                      ('HT 8x8 (global)', 'ht', 3), ('Haar 32x32 full', 'haar', 5), ('HT 32x32 (global)', 'ht', 5)):
    P = 2 ** m; WW = np.kron(haar_matrix(P), haar_matrix(P)) if kind == 'haar' else None
    if kind == 'ht':
        from scipy.linalg import hadamard; Hm = hadamard(P) / np.sqrt(P); WW = np.kron(Hm, Hm)
    base = QuantumCircuit(2 * m)
    if kind == 'haar':
        haar_circuit(m, 0, base, 2 * m); haar_circuit(m, m, base, 2 * m)
    else:
        base.h(range(2 * m))
    tb = transpile(base, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=2, seed_transpiler=0)
    ops = tb.count_ops(); n1 = sum(v for k, v in ops.items() if k != 'cx'); ncx = ops.get('cx', 0)
    lam = 1 - (1 - e1) ** n1 * (1 - e2) ** ncx; N = P * P
    # optimization_level=2 elides plain SWAPs by relabeling qubits: undo that output permutation
    from qiskit.quantum_info import Operator
    Praw = Operator(tb).data @ Operator.from_circuit(tb).data.conj().T
    perm = np.abs(Praw).argmax(0)            # p_virtual[j] = p_raw[perm[j]]
    assert np.allclose(np.abs(Praw)[perm, np.arange(len(perm))], 1)
    eu, em = [], []
    for x in cifar_patches(m, 8 if m < 5 else 2, seed=3):
        xv = x.flatten(); Y = WW @ xv; c = np.linalg.norm(xv); xp = xv + c * (WW.T @ np.ones(N)); nrm = np.linalg.norm(xp)
        qc = QuantumCircuit(2 * m); v = xp / nrm; qc.set_density_matrix(np.outer(v, v)); qc.compose(tb, inplace=True); qc.save_probabilities()
        pr = sim.run(qc).result().data()['probabilities']
        pr = np.asarray(pr)[perm]
        # qiskit little-endian: statevector index already matches row-major flatten used by WW (verified in haar_circuit.py)
        rec_u = np.sqrt(np.clip(pr, 0, None)) * nrm - c
        rec_m = np.sqrt(np.clip((pr - lam / N) / (1 - lam), 0, None)) * nrm - c
        eu.append(np.linalg.norm(rec_u - Y) / np.linalg.norm(Y)); em.append(np.linalg.norm(rec_m - Y) / np.linalg.norm(Y))
    print(f'{name:24s} {2*m:6d} {ncx:4d} {lam:7.4f} | {"":9s}{np.mean(eu):11.4f} {np.mean(em):12.4f}', flush=True)
