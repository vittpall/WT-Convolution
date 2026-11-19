"""n-qubit Haar wavelet transform circuit (qiskit, little-endian).

Recursive structure: one Haar level on N=2^k amplitudes is
    B_k = (cyclic shift of the k qubits moving the lsb to the msb) . (H on lsb)
which sends the pair sums to the first half and pair differences to the second.
The next level acts only on the first half, i.e. on the lower k-1 qubits
*conditioned on the msb being |0>*. Hence

    Haar_m = prod_{l=0}^{m-1} [ controlled-on-|0>^l ( shift_{m-l} . H ) ]

with l controls (the top l qubits). Coefficient order = [DC, coarse details, ..., finest
details], identical to `exp/layers.py::haar_matrix`. 2D = the same circuit on a
column register and a row register (separable): 2m qubits for a 2^m x 2^m patch.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, SwapGate


def haar_circuit(m, qubits_offset=0, qc=None, n_total=None, levels=None):
    n_total = n_total or (m + qubits_offset)
    if qc is None:
        qc = QuantumCircuit(n_total)
    q = lambda i: qubits_offset + i
    for l in range(m if levels is None else min(levels, m)):
        k = m - l                                   # active register: qubits 0..k-1
        ctrls = [q(j) for j in range(k, m)]         # top l qubits, must be |0>
        if l == 0:
            qc.h(q(0))
        else:
            qc.append(HGate().control(l, ctrl_state=0), ctrls + [q(0)])
        for j in range(k - 1):                      # lsb -> msb of the active register
            if l == 0:
                qc.swap(q(j), q(j + 1))
            else:
                qc.append(SwapGate().control(l, ctrl_state=0), ctrls + [q(j), q(j + 1)])
    return qc


def haar_circuit_2d(m):
    qc = QuantumCircuit(2 * m)
    haar_circuit(m, 0, qc)      # columns (low bits)
    haar_circuit(m, m, qc)      # rows (high bits)
    return qc


def haar_matrix(n):
    """Orthonormal Haar, rows [DC, coarse -> fine details] (same as exp/layers.py)."""
    m = int(np.log2(n)); W = np.eye(n); length = n
    for _ in range(m):
        half = length // 2
        B = np.eye(n); B[:length, :length] = 0
        for k in range(half):
            B[k, 2 * k] = B[k, 2 * k + 1] = B[half + k, 2 * k] = 1 / np.sqrt(2)
            B[half + k, 2 * k + 1] = -1 / np.sqrt(2)
        W = B @ W; length = half
    return W


def haar_circuit_baseline_ht(m):
    qc = QuantumCircuit(m)
    for i in range(m):
        qc.h(i)
    return qc


if __name__ == '__main__':
    from qiskit.quantum_info import Operator
    from qiskit import transpile
    print('--- 1D correctness: max |circuit unitary - classical Haar matrix|')
    for m in range(1, 7):
        U = Operator(haar_circuit(m)).data
        print(f'  m={m} (N={2**m:3d}): {np.abs(U - haar_matrix(2**m)).max():.2e}')
    print('--- 2D correctness (kron W,W):')
    for m in (2, 3):
        U = Operator(haar_circuit_2d(m)).data
        W = haar_matrix(2 ** m)
        print(f'  {2**m}x{2**m} patch, {2*m} qubits: {np.abs(U - np.kron(W, W)).max():.2e}')
    print('--- 1D transpiled cost, basis {cx, rz, sx, x}, all-to-all coupling')
    print(f"  {'m':>2s} {'N':>4s} | {'Haar cx':>8s} {'depth':>6s} {'1q':>6s} | {'HT gates':>8s} {'HT depth':>8s}")
    for m in range(1, 8):
        t = transpile(haar_circuit(m), basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=2, seed_transpiler=0)
        ops = t.count_ops()
        h = transpile(haar_circuit_baseline_ht(m), basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=2)
        cx = ops.get('cx', 0); one = sum(v for k, v in ops.items() if k != 'cx')
        print(f'  {m:2d} {2**m:4d} | {cx:8d} {t.depth():6d} {one:6d} | {sum(h.count_ops().values()):8d} {h.depth():8d}')
