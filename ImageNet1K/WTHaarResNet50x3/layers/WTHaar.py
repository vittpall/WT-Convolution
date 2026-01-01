#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HWT-MA-Net: Haar Wavelet Transform + Multiplication-Avoiding Operators
FINAL WORKING VERSION
"""

import numpy as np
import torch
#from scipy.linalg import hadamard
from .common import SoftThresholding, find_min_power

def haar_transform_1d(u, axis=-1, inverse=False):
    if axis != -1:
        u = torch.transpose(u, -1, axis)

    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    x = u.clone()
    norm = 1.0 / np.sqrt(2)

    #equivalent to applying the filter bank + downsampling m times
    if not inverse:
        # forward
        for _ in range(m):
            # ... take all the rows, ::2 start from 0 go till the end and move by step of 2
            even = x[..., ::2]
            odd  = x[..., 1::2]

            x[..., ::2] = norm * (even + odd)
            x[..., 1::2] = norm * (even - odd)
    else:
        # inverse (reverse order)
        for _ in range(m):
            a = x[..., ::2]
            d = x[..., 1::2]

            x[..., ::2] = norm * (a + d)
            x[..., 1::2] = norm * (a - d)

    if axis != -1:
        x = torch.transpose(x, -1, axis)

    return x


def haar_transform_2d(x, inverse=False):
    """2D separable Haar Wavelet Transform."""
    x_haar = haar_transform_1d(x, axis=-1, inverse=inverse)
    x_haar = haar_transform_1d(x_haar, axis=-2, inverse=inverse)
    return x_haar

class HWTConv2D(torch.nn.Module):
    """2D Haar Wavelet Conv layer - HWT-MA-Net"""
    def __init__(self, height, width, in_channels, out_channels, pods=1, residual=True):
        super().__init__()
        self.height = height
        self.width = width
        self.height_pad = find_min_power(self.height)
        self.width_pad = find_min_power(self.width)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pods = pods
        
        # 1x1 Convolution, Channel mixing per pod
        self.conv = torch.nn.ModuleList([torch.nn.Conv2d(in_channels, out_channels, 1, bias=False) for _ in range(self.pods)])
        
        # Scaling parameters per pod
        self.v = torch.nn.ParameterList([
            torch.nn.Parameter(torch.rand(1, 1, self.height_pad, self.width_pad))
            for _ in range(pods)
        ])
        
        # Soft-thresholding per pod
        self.ST = torch.nn.ModuleList([
            SoftThresholding((self.height_pad, self.width_pad)) for _ in range(pods)
        ])
        self.residual = residual
        
    def forward(self, x):
        B, C_in, height, width = x.shape
        if height != self.height or width != self.width:
            raise Exception(f'({height}, {width})!=({self.height}, {self.width})')
        
        # Pad to power of 2
        pad_h = self.height_pad - height
        pad_w = self.width_pad - width
        if pad_h > 0 or pad_w > 0:
            f0 = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        else:
            f0 = x
        
        # 2D Haar Transform
        f1 = haar_transform_2d(f0)
        
        outputs = []
        for i in range(self.pods):
            f3 = f1 * self.v[i]  # Scaling
            f4 = self.conv[i](f3)  # 1x1 Conv
            f5 = self.ST[i](f4)  # Soft-threshold
            outputs.append(f5)
        
        # Sum paths
        f6 = torch.stack(outputs, dim=0).sum(dim=0)
        
        # Inverse Haar
        f7 = haar_transform_2d(f6, inverse=True)
        
        # Crop
        y = f7[:, :, :height, :width]
        
        if self.residual:
            y = y + x
        return y


if __name__ == '__main__':
    torch.manual_seed(42)
    x2 = torch.rand((2, 64, 32, 32))
    model2 = HWTConv2D(32, 32, 64, 64, pods=3)
    y2 = model2(x2)
    print(f"2D Haar-MF: {y2.shape}")
