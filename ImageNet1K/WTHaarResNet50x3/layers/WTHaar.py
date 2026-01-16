#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from .common import SoftThresholding, find_min_power

#Previous incorrect implementation, give the same results because wavalet haar transform is symmetric
"""
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
            even = x[..., ::2].clone()
            odd  = x[..., 1::2].clone()

            x[..., ::2] = (norm * (even + odd)) 
            x[..., 1::2] = (norm * (even - odd))
            print(x)
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
"""

def haar_transform_1d(u, axis=-1, inverse=False):

    if axis != -1:
        u = torch.transpose(u, -1, axis)
    
    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    
    x = u.clone()
    #to simultate hadamard transform normalization coefficent
    norm = 1.0 / np.sqrt(2)
    
    if not inverse:
        # Forward transform
        length = n
        for _ in range(m):
            # Process the current approximation coefficients
            # Split into even and odd indices
            half = length // 2
            
            # Create temporary storage
            temp = x[..., :length].clone()
            
            # Compute approximation (low-pass): (even + odd) / sqrt(2)
            x[..., :half] = norm * (temp[..., ::2] + temp[..., 1::2])
            
            # Compute detail (high-pass): (even - odd) / sqrt(2)
            x[..., half:length] = norm * (temp[..., ::2] - temp[..., 1::2])
            
            # Next level operates only on the approximation coefficients
            length = half
    else:
        # Inverse transform
        # Reconstruct from coarsest to finest level
        length = 2
        for _ in range(m):
            half = length // 2
            
            # Get approximation and detail coefficients
            temp = x[..., :length].clone()
            approx = temp[..., :half]
            detail = temp[..., half:length]
            
            # Reconstruct even indices: (approx + detail) / sqrt(2)
            x[..., :length:2] = norm * (approx + detail)
            
            # Reconstruct odd indices: (approx - detail) / sqrt(2)
            x[..., 1:length:2] = norm * (approx - detail)
            
            # Next level processes twice as many coefficients
            length *= 2
    
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
    x1 = torch.rand((4,4))
    y1 = haar_transform_1d(x1)
    print(f"input: {x1}")
    print(f"1D Haar: {y1}")
    x2 = torch.rand((2, 64, 32, 32))
    model2 = HWTConv2D(32, 32, 64, 64, pods=3)
    #y2 = model2(x2)
    #print(f"2D Haar-MF: {y2.shape}")
