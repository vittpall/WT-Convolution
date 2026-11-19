"""Unified transform-domain conv layer for controlled HT vs Haar comparisons.

Both transforms are applied as separable matrix products  Y = T_h X T_w^T
with precomputed matrices, so the only difference between variants is the
matrix itself (and, for 'ht_pan', the scaling convention of Pan et al.).

kind:
  'haar'    orthonormal multi-level Haar (levels=None -> full decomposition)
  'ht'      orthonormal Walsh-Hadamard  (H / sqrt(n))
  'ht_pan'  Pan et al. CIFAR code: forward H (unnormalized), inverse H / n
"""
import numpy as np
import torch
from scipy.linalg import hadamard


def find_min_power(x, p=2):
    y = 1
    while y < x:
        y *= p
    return y


def haar_matrix(n, levels=None):
    """Orthonormal Haar analysis matrix (n x n) with `levels` decomposition levels.

    Row order: [approx at coarsest level, details coarse -> fine]. Any fixed
    ordering is equivalent for this layer (per-coefficient params are elementwise).
    """
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    L = m if levels is None else min(levels, m)
    W = np.eye(n)
    length = n
    for _ in range(L):
        half = length // 2
        B = np.eye(n)
        B[:length, :length] = 0.0
        for k in range(half):
            B[k, 2 * k] = B[k, 2 * k + 1] = 1 / np.sqrt(2)
            B[half + k, 2 * k] = 1 / np.sqrt(2)
            B[half + k, 2 * k + 1] = -1 / np.sqrt(2)
        W = B @ W
        length = half
    return W


def transform_pair(kind, n, levels=None):
    """Return (forward, inverse) matrices F, G such that G @ (F @ x) == x."""
    if kind == 'haar':
        F = haar_matrix(n, levels)
        G = F.T
    elif kind == 'ht':
        b = n if levels is None else min(2 ** levels, n)   # block Hadamard on 2^L-blocks (control for block Haar)
        F = np.kron(np.eye(n // b), hadamard(b) / np.sqrt(b))
        G = F.T
    elif kind == 'ht_pan':
        F = hadamard(n).astype(float)
        G = hadamard(n) / n
    else:
        raise ValueError(kind)
    return torch.tensor(F, dtype=torch.float32), torch.tensor(G, dtype=torch.float32)


class LearnableHaar(torch.nn.Module):
    """Orthogonal wavelet whose butterfly angle is trainable (N1).

    Each level applies rows  approx = [cos t, sin t],  detail = [sin t, -cos t]
    on adjacent pairs. t = pi/4 is exactly the orthonormal Haar butterfly. Any t
    keeps the level orthogonal, so the inverse is the transpose (exact).
    mode 'level': one angle per level;  'pos': one angle per butterfly.
    """

    def __init__(self, n, mode='level'):
        super().__init__()
        self.n = n
        self.L = int(np.log2(n))
        sizes = [n >> (l + 1) for l in range(self.L)]  # butterflies per level
        self.theta = torch.nn.ParameterList([
            torch.nn.Parameter(torch.full((1 if mode == 'level' else k,), np.pi / 4))
            for k in sizes])

    def forward(self):
        n = self.n
        W = torch.eye(n, device=self.theta[0].device)
        length = n
        for t in self.theta:
            half = length // 2
            c, s = torch.cos(t).expand(half), torch.sin(t).expand(half)
            k = torch.arange(half, device=t.device)
            B = torch.eye(n, device=t.device)
            B[:length, :length] = 0.0
            B[k, 2 * k] = c
            B[k, 2 * k + 1] = s
            B[half + k, 2 * k] = s
            B[half + k, 2 * k + 1] = -c
            W = B @ W
            length = half
        return W


class SoftThresholding(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.T = torch.nn.Parameter(torch.rand(shape) / 10)

    def forward(self, x):
        return torch.sign(x) * torch.relu(torch.abs(x) - torch.abs(self.T))


class TransformConv2D(torch.nn.Module):
    """P-path transform-domain perceptron (scaling -> 1x1 conv -> soft-threshold)."""

    def __init__(self, height, width, in_channels, out_channels, pods=1,
                 residual=True, kind='haar', levels=None):
        super().__init__()
        self.height, self.width = height, width
        self.hp, self.wp = find_min_power(height), find_min_power(width)
        self.pods, self.residual = pods, residual
        self.learn = kind in ('lhaar', 'lhaar_pos')
        if self.learn:
            mode = 'pos' if kind == 'lhaar_pos' else 'level'
            self.lh, self.lw = LearnableHaar(self.hp, mode), LearnableHaar(self.wp, mode)
        else:
            Fh, Gh = transform_pair(kind, self.hp, levels)
            Fw, Gw = transform_pair(kind, self.wp, levels)
            self.register_buffer('Fh', Fh)
            self.register_buffer('Gh', Gh)
            self.register_buffer('Fw', Fw)
            self.register_buffer('Gw', Gw)
        self.conv = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels, out_channels, 1, bias=False) for _ in range(pods)])
        self.v = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.rand(self.hp, self.wp)) for _ in range(pods)])
        self.ST = torch.nn.ModuleList([SoftThresholding((self.hp, self.wp)) for _ in range(pods)])

    def forward(self, x):
        h, w = x.shape[-2:]
        f0 = x
        if self.hp > h or self.wp > w:
            f0 = torch.nn.functional.pad(x, (0, self.wp - w, 0, self.hp - h))
        if self.learn:
            Fh, Fw = self.lh(), self.lw()
            Gh, Gw = Fh.T, Fw.T
        else:
            Fh, Fw, Gh, Gw = self.Fh, self.Fw, self.Gh, self.Gw
        X = Fh @ f0 @ Fw.T
        Y = sum(self.ST[i](self.conv[i](X * self.v[i])) for i in range(self.pods))
        y = (Gh @ Y @ Gw.T)[..., :h, :w]
        return y + x if self.residual else y
