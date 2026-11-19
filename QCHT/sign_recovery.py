"""Sign recovery for amplitude-measured Haar coefficients (finite shots).

Measurement gives p_j = |amp_j|^2, i.e. only |coefficient|. Fix: encode x' = x + c*g with
g = W^T 1  (W = 2D Haar, orthogonal). Then W x' = W x + c*1, and c >= max_j |(Wx)_j|
makes EVERY coefficient non-negative, so sqrt(p_j) is the signed value (minus c).
c = ||x||_2 is a valid, classically known bound since |(Wx)_j| <= ||Wx||_2 = ||x||_2.
(Adding a plain constant offset does NOT work: detail rows sum to zero.)
No extra qubits; costs one classical subtraction and some shot-noise amplification.
"""
import numpy as np, pickle, sys
from qiskit.quantum_info import Statevector
from haar_circuit import haar_circuit_2d, haar_matrix

def cifar_patches(m, n, seed=0):
    d = pickle.load(open('../CIFAR10/data/cifar-10-batches-py/test_batch', 'rb'), encoding='bytes')
    imgs = d[b'data'].reshape(-1, 3, 32, 32).astype(float).mean(1) / 255.0
    rng = np.random.default_rng(seed); P = 2 ** m; out = []
    for _ in range(n):
        im = imgs[rng.integers(len(imgs))]; r, c = rng.integers(0, 32 - P + 1, 2)
        out.append(im[r:r + P, c:c + P] - 0.5)       # centred pixels -> signed coefficients
    return np.array(out)

def run(m=2, n_patches=100, shots_list=(10**3, 10**4, 10**5, 10**6), seed=0):
    P = 2 ** m; W = haar_matrix(P); WW = np.kron(W, W)
    circ = haar_circuit_2d(m)
    rng = np.random.default_rng(seed + 1)
    g = WW.T @ np.ones(P * P)
    print(f'{P}x{P} patch, {2*m} qubits, {n_patches} CIFAR patches')
    print(f"{'shots':>8s} | {'naive |.| rel.err vs signed':>27s} {'sign-wrong %':>13s} | {'recovered rel.err':>18s} {'sign-wrong %':>13s}")
    X = cifar_patches(m, n_patches, seed)
    for shots in shots_list:
        e_n, e_r, s_n, s_r = [], [], [], []
        for x in X:
            xv = x.flatten(); Y = WW @ xv; nx = np.linalg.norm(xv)
            def measure(vec):
                v = vec / np.linalg.norm(vec)
                p = np.abs(Statevector(v).evolve(circ).data) ** 2
                p = p / p.sum()
                return rng.multinomial(shots, p) / shots
            # naive: magnitudes only
            a = np.sqrt(measure(xv)) * nx
            e_n.append(np.linalg.norm(a - Y) / np.linalg.norm(Y)); s_n.append(np.mean(np.sign(a) != np.sign(Y)))
            # recovered
            c = nx; xp = xv + c * g
            a2 = np.sqrt(measure(xp)) * np.linalg.norm(xp) - c
            e_r.append(np.linalg.norm(a2 - Y) / np.linalg.norm(Y)); s_r.append(np.mean(np.sign(a2) != np.sign(Y)))
        print(f'{shots:8d} | {np.mean(e_n):27.3f} {100*np.mean(s_n):13.1f} | {np.mean(e_r):18.3f} {100*np.mean(s_r):13.1f}')

if __name__ == '__main__':
    run(2); print(); run(3, n_patches=60)
