"""Exact noisy measurement channel of the 2D transform circuits (<= 6 qubits).

T[j, a, b] such that the output probability of outcome j for input pure state psi is
    p_j = sum_ab T[j,a,b] psi_a psi_b      (real amplitudes)
computed from the Aer superoperator of the TRANSPILED circuit (basis cx/rz/sx/x, opt level 2)
under per-gate depolarizing noise (e1 on 1q gates, e2 on CX). The qubit relabeling that the
transpiler uses to elide SWAPs is undone, so j indexes coefficients in the order of the
classical matrix (row-major 2D Haar / Hadamard)."""
import os
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from haar_circuit import haar_circuit

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'channel_cache')


def transform_circuit(kind, mh, mw, inverse=False):
    qc = QuantumCircuit(mh + mw)
    if kind == 'haar':
        haar_circuit(mw, 0, qc, mh + mw)    # columns on low qubits
        haar_circuit(mh, mw, qc, mh + mw)   # rows on high qubits
    else:
        qc.h(range(mh + mw))
    if inverse:
        qc = qc.inverse()
    return transpile(qc, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=2, seed_transpiler=0)


def noisy_T(kind, mh, mw, e1=2.5e-4, e2=3e-3, inverse=False):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"{kind}{'_inv' if inverse else ''}_{mh}x{mw}_e1{e1:g}_e2{e2:g}.npy")
    if os.path.exists(f):
        return np.load(f)
    tb = transform_circuit(kind, mh, mw, inverse)
    N = 2 ** (mh + mw)
    Praw = Operator(tb).data @ Operator.from_circuit(tb).data.conj().T
    perm = np.abs(Praw).argmax(0)
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(e1, 1), ['sx', 'x', 'rz'])
    nm.add_all_qubit_quantum_error(depolarizing_error(e2, 2), ['cx'])
    qc = tb.copy(); qc.save_superop()
    S = AerSimulator(method='superop', noise_model=nm).run(qc).result().data()['superop']
    S = np.asarray(S.data if hasattr(S, 'data') else S)
    diag = np.arange(N) * (N + 1)                          # vec index of E(rho)_jj (column stacking)
    T = S[diag].reshape(N, N, N).transpose(0, 2, 1).real    # vec(rho)[a + N b] = rho_ab -> T[j,a,b]
    T = T[perm]
    np.save(f, T)
    return T


if __name__ == '__main__':
    from haar_circuit import haar_matrix
    from scipy.linalg import hadamard
    rng = np.random.default_rng(0)
    for kind, m in (('haar', 1), ('haar', 2), ('haar', 3), ('ht', 3)):
        P = 2 ** m
        W1 = haar_matrix(P) if kind == 'haar' else hadamard(P) / np.sqrt(P)
        W = np.kron(W1, W1)
        T0 = noisy_T(kind, m, m, e1=0.0, e2=0.0)
        T = noisy_T(kind, m, m)
        Ti0 = noisy_T(kind, m, m, 0.0, 0.0, inverse=True)
        psi = rng.standard_normal(P * P); psi /= np.linalg.norm(psi)
        p0 = np.einsum('jab,a,b->j', T0, psi, psi)
        p = np.einsum('jab,a,b->j', T, psi, psi)
        print(f'{kind} {P}x{P} ({2*m} qubits): noiseless T vs |W psi|^2 max err {np.abs(p0 - (W @ psi) ** 2).max():.1e};'
              f' inverse ok {np.abs(np.einsum("jab,a,b->j", Ti0, psi, psi) - (W.T @ psi) ** 2).max():.1e};'
              f' noisy TVD from ideal {0.5*np.abs(p - p0).sum():.4f}; sum p = {p.sum():.6f}')
