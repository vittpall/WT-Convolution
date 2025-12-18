#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 12:14:21 2022

@author: Zephyr
"""

import numpy as np
import torch
from .common import find_min_power, SoftThresholding
from pytorch_wavelets import SWTForward, SWTInverse

class WaveletConv2D(torch.nn.Module):
    def __init__(self, height, width, in_channels, out_channels,
                 wave='db2', levels=2, pods=1, residual=True):
        super().__init__()

        self.height = height
        self.width = width
        self.levels = levels
        self.pods = pods
        self.residual = residual

        # Wavelet transforms
        self.wt = SWTForward(J=levels, wave=wave)
        self.iwt = SWTInverse(wave=wave)

        # Channel mixing (same as Hadamard)
        self.conv = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels, out_channels, 1, bias=False)
            for _ in range(pods)
        ])

        # Soft-thresholding
        self.ST = torch.nn.ModuleList([
            SoftThresholding((height, width))
            for _ in range(pods)
        ])

        # One gain per (level, orientation)
        self.v = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(levels, 3))
            for _ in range(pods)
        ])
       
    def forward(self, x):
        f0 = x

        # Wavelet transform
        yl, yh = self.wt(f0)
        # yl: [B, C, H, W]
        # yh[j]: [B, C, 3, H, W]

        outputs = []

        for i in range(self.pods):

            # ---- Diagonal operator (wavelet domain) ----
            yh_mod = []
            for j in range(self.levels):
                sub = yh[j]
                gain = self.v[i][j].view(1, 1, 3, 1, 1)
                yh_mod.append(sub * gain)

            # ---- Channel mixing in wavelet domain ----
            yl_i = self.conv[i](yl)

            yh_mix = []
            for j in range(self.levels):
                bands = []
                for o in range(3):
                    bands.append(self.conv[i](yh_mod[j][:, :, o]))
                yh_mix.append(torch.stack(bands, dim=2))

            # ---- Thresholding (still in wavelet domain) ----
            yl_i = self.ST[i](yl_i)

            yh_thr = []
            for j in range(self.levels):
                bands = []
                for o in range(3):
                    bands.append(self.ST[i](yh_mix[j][:, :, o]))
                yh_thr.append(torch.stack(bands, dim=2))

            # ---- Inverse wavelet transform ----
            f = self.iwt((yl_i, yh_thr))
            outputs.append(f)

        y = torch.stack(outputs, dim=-1).sum(dim=-1)

        if self.residual:
            y = y + x

        return y

