#!/usr/bin/env python3
"""Run trained CIFAR-10 models with every transform (forward AND inverse) replaced by an
emulated quantum circuit + measurement, and report test accuracy.

Per (image, channel) feature map, per transform application:
  * the map is split into the transform's independent blocks (an L-level Haar is block-diagonal
    on 2^L x 2^L patches; HT and full-depth Haar are one global block),
  * each block is amplitude-encoded on log2(bh)+log2(bw) qubits with the W^T 1 offset
    (QCHT/sign_recovery.py): every output amplitude is >= 0, so signs are recovered exactly,
  * outputs are measured with S_b = S_total / n_blocks shots (Binomial sampling per outcome),
  * optional gate noise: global depolarizing  p -> (1-lam) p + lam/N, with
    lam = 1 - (1-e1)^n1q (1-e2)^nCX from the TRANSPILED circuit, then standard depolarizing
    mitigation  p_hat -> (p_hat - lam/N)/(1-lam)  (lam assumed known/calibrated).
S_total is the budget per feature map per transform, so Haar-L2 and HT spend the same shots.
"""
import argparse, json, math, os, sys
import numpy as np
import torch
from scipy.linalg import hadamard
import torchvision.datasets as dsets
from torchvision import transforms
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import layers as Lyr  # noqa: E402
from models import build  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--configs', nargs='+', default=['haar_p3', 'haar_p3_L1', 'haar_p3_L2', 'haar_p3_L3', 'ht_p3', 'htpan_p3'])
p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 777, 1234, 9999])
p.add_argument('--shots', nargs='+', type=float, default=[1e4, 1e5, 1e6, 1e7, 1e8, math.inf])
p.add_argument('--e1', type=float, default=2.5e-4, help='1-qubit gate error')
p.add_argument('--e2', type=float, default=3e-3, help='CX error')
p.add_argument('--n-test', type=int, default=10000)
p.add_argument('--out', default=os.path.join(HERE, 'results', 'emulation.json'))
args = p.parse_args()
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- gate counts of the 2D circuits (transpiled) ----------------
sys.path.insert(0, os.path.join(HERE, '..', '..', 'QCHT'))
_counts = {}


def circuit_counts(kind, mh, mw):
    key = (kind, mh, mw)
    if key not in _counts:
        from qiskit import QuantumCircuit, transpile
        from haar_circuit import haar_circuit
        qc = QuantumCircuit(mh + mw)
        if kind == 'haar':
            haar_circuit(mh, 0, qc, mh + mw); haar_circuit(mw, mh, qc, mh + mw)
        else:
            qc.h(range(mh + mw))
        t = transpile(qc, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=2, seed_transpiler=0)  # SWAPs elided by relabeling
        ops = t.count_ops()
        _counts[key] = (sum(v for k, v in ops.items() if k != 'cx'), ops.get('cx', 0))
    return _counts[key]


# ---------------- measurement emulation ----------------
from noisy_channel import noisy_T  # noqa: E402  (QCHT on sys.path)


class Q:
    shots = math.inf
    noisy = False
    gen = None


def axis_map(M, b, kind, outputs_are_coeffs):
    """For each output index r of the 1D matrix M: (block id, index inside the block in the
    circuit's canonical order). Haar coefficients are matched row-by-row against haar_matrix(b)."""
    n = M.shape[0]
    if b >= n:
        return np.zeros(n, int), np.arange(n)
    if not outputs_are_coeffs:
        return np.arange(n) // b, np.arange(n) % b
    Hb = Lyr.haar_matrix(b)
    blk, loc = np.zeros(n, int), np.zeros(n, int)
    for r in range(n):
        k = int(np.nonzero(np.abs(M[r]) > 1e-9)[0].min()) // b
        row = M[r, k * b:(k + 1) * b]
        hit = [i for i in range(b) if np.allclose(row, Hb[i])]
        assert len(hit) == 1, (r, hit)
        blk[r], loc[r] = k, hit[0]
    return blk, loc


def gather_index(bh, lh, bw, lw, b_h, b_w):
    """idx[kh, kw, i] = flat (H*W) position of the output with canonical in-block index i = ih*b_w + iw."""
    Kh, Kw = bh.max() + 1, bw.max() + 1
    idx = np.zeros((Kh, Kw, b_h * b_w), int)
    for r in range(len(bh)):
        for c in range(len(bw)):
            idx[bh[r], bw[c], lh[r] * b_w + lw[c]] = r * len(bw) + c
    return torch.tensor(idx.reshape(Kh * Kw, b_h * b_w), device=dev)


def qmeasure(z, idx, lam, Tp):
    """z: exact orthonormal-transform output (B,C,H,W). idx: (K, Nb) block gather index.
    Tp: exact noisy channel in output space (Nb,Nb,Nb) or None (=> global depolarizing model)."""
    if math.isinf(Q.shots) and not Q.noisy:
        return z
    B, C, H, W = z.shape
    K, Nb = idx.shape
    flat = z.reshape(B * C, H * W).double()
    y = flat[:, idx]                                  # (BC, K, Nb)
    c = y.norm(dim=-1, keepdim=True)                  # W^T 1 offset  =>  all amplitudes >= 0
    yp = y + c
    norm2 = (yp * yp).sum(-1, keepdim=True).clamp_min(1e-300)
    u = yp / norm2.sqrt()
    if Q.noisy and Tp is not None:
        uf = u.reshape(-1, Nb); pr = torch.empty_like(uf)
        for s in range(0, uf.shape[0], 8192):
            v = uf[s:s + 8192]
            pr[s:s + 8192] = torch.einsum('jcd,vc,vd->vj', Tp, v, v)
        pr = pr.reshape(u.shape).clamp_min(0)
    elif Q.noisy:
        pr = (1 - lam) * u * u + lam / Nb
    else:
        pr = u * u
    if not math.isinf(Q.shots):
        Sb = max(1, int(round(Q.shots / K)))
        cnt = torch.binomial(torch.full(pr.shape, float(Sb), device=pr.device), pr.float().clamp(0, 1), generator=Q.gen)
        pr = cnt.double() / Sb
    if Q.noisy:                                       # standard global-depolarizing mitigation
        pr = ((pr - lam / Nb) / (1 - lam)).clamp_min(0)
    out = flat.clone()
    out[:, idx] = pr.sqrt() * norm2.sqrt() - c
    return out.reshape(B, C, H, W).float()


def emulated_forward(self, x):
    h, w = x.shape[-2:]
    f0 = x
    if self.hp > h or self.wp > w:
        f0 = torch.nn.functional.pad(x, (0, self.wp - w, 0, self.hp - h))
    st = self._q
    X = st['Qfh'] @ f0 @ st['Qfw'].T
    X = qmeasure(X, st['idx_f'], st['lam_f'], st['T_f']) * st['sf']
    Y = sum(self.ST[i](self.conv[i](X * self.v[i])) for i in range(self.pods))
    y = st['Qgh'] @ Y @ st['Qgw'].T                 # inverse matrix G = sg * orthonormal Qg
    y = qmeasure(y, st['idx_g'], st['lam_g'], st['T_g']) * st['sg']
    y = y[..., :h, :w]
    return y + x if self.residual else y


def prepare(model, cfg_name):
    from models import CONFIGS
    cfg = CONFIGS[cfg_name]
    kind, levels = cfg['kind'], cfg.get('levels')
    tk = 'haar' if kind == 'haar' else 'ht'
    info = []
    for mod in model.modules():
        if not isinstance(mod, Lyr.TransformConv2D):
            continue
        st, maps = {}, {}
        for ax, n in (('h', mod.hp), ('w', mod.wp)):
            F, G = getattr(mod, 'F' + ax).cpu().numpy(), getattr(mod, 'G' + ax).cpu().numpy()
            # every row has the same norm: 1 if orthonormal; sqrt(n) (F) and 1/sqrt(n) (G) for ht_pan
            sf, sg = float(np.linalg.norm(F[0])), float(np.linalg.norm(G[0]))
            b = n if (kind not in ('haar', 'ht') or levels is None) else min(2 ** levels, n)
            st['Qf' + ax] = torch.tensor(F / sf, dtype=torch.float32, device=dev)
            st['Qg' + ax] = torch.tensor(G / sg, dtype=torch.float32, device=dev)
            maps['f' + ax] = axis_map(F / sf, b, kind, kind == 'haar')
            maps['g' + ax] = axis_map(G / sg, b, kind, False)
            st['sf' + ax], st['sg' + ax], st['b' + ax] = sf, sg, b
        st['sf'], st['sg'] = st['sfh'] * st['sfw'], st['sgh'] * st['sgw']
        bh, bw = st['bh'], st['bw']; mh, mw = int(math.log2(bh)), int(math.log2(bw))
        n1, ncx = circuit_counts(tk, mh, mw)
        for d, inv in (('f', False), ('g', True)):
            st['idx_' + d] = gather_index(*maps[d + 'h'], *maps[d + 'w'], bh, bw)
            st['lam_' + d] = 1 - (1 - args.e1) ** n1 * (1 - args.e2) ** ncx
            if mh + mw <= 6:
                T = noisy_T(tk, mh, mw, args.e1, args.e2, inverse=inv)
                U = np.kron(Lyr.haar_matrix(bh), Lyr.haar_matrix(bw)) if tk == 'haar' else \
                    np.kron(hadamard(bh), hadamard(bw)) / math.sqrt(bh * bw)
                U = U.T if inv else U                  # circuit unitary in canonical order
                Tp = np.einsum('jab,ca,db->jcd', T, U, U)   # psi = U^T y'
                st['T_' + d] = torch.tensor(Tp, dtype=torch.float64, device=dev)
            else:
                st['T_' + d] = None
        info.append(dict(map=mod.hp, block=bh, qubits=mh + mw, blocks=(mod.hp // bh) * (mod.wp // bw), n1q=n1,
                         ncx=ncx, lam=st['lam_f'], channel='aer-exact' if mh + mw <= 6 else 'global-depol (optimistic)'))
        mod._q = st
        mod.forward = emulated_forward.__get__(mod)
    return info


norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
test = dsets.CIFAR10(os.path.join(HERE, '..', 'data'), False, transforms.Compose([transforms.ToTensor(), norm]))
test = torch.utils.data.Subset(test, range(args.n_test))
ld = DataLoader(test, 500, shuffle=False, num_workers=4)


def acc(model):
    model.eval(); c = n = 0
    with torch.no_grad():
        for x, y in ld:
            o = model(x.to(dev)); c += (o.argmax(1).cpu() == y).sum().item(); n += len(y)
    return 100 * c / n


results = json.load(open(args.out)) if os.path.exists(args.out) else {}
for cfg in args.configs:
    for seed in args.seeds:
        model = build(cfg).to(dev)
        model.load_state_dict(torch.load(os.path.join(HERE, 'results', f'{cfg}_s{seed}.pth'), map_location=dev))
        clean = acc(model)
        circ = prepare(model, cfg)
        for noisy in (False, True):
            for S in args.shots:
                key = f'{cfg}|{seed}|{"gates" if noisy else "shots"}|{S:g}'
                if key in results:
                    continue
                Q.shots, Q.noisy = S, noisy
                Q.gen = torch.Generator(device=dev).manual_seed(seed)
                a = acc(model)
                results[key] = dict(config=cfg, seed=seed, noise='gates' if noisy else 'shots', shots=S,
                                    acc=a, clean=clean)
                print(f'{key:40s} acc {a:6.2f}  (clean {clean:.2f})', flush=True)
                json.dump(results, open(args.out, 'w'))
        results[f'{cfg}|circuits'] = circ
        json.dump(results, open(args.out, 'w'))
print('saved', args.out)
