#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 12:14:21 2022

@author: Zephyr
"""

import numpy as np
import torch
from .common import find_min_power, SoftThresholding

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from pytorch_wavelets import DWTForward, DWTInverse



class WTConv2D(torch.nn.Module):
    def __init__(self, height, width, in_channels, out_channels,
                 wave='db2', levels=2, pods=1, residual=True):
        super().__init__()

        self.height = height
        self.width = width
        self.levels = levels
        self.pods = pods
        self.residual = residual

        # Wavelet transforms
        self.wt = DWTForward(J=levels, wave=wave)
        self.iwt = DWTInverse(wave=wave)

        # Channel mixing (same as Hadamard)
        self.conv = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels, out_channels, 1, bias=False)
            for _ in range(pods)
        ])

        # Soft-thresholding
        self.ST = torch.nn.ModuleList([
            SoftThresholding()
            for _ in range(pods)
        ])

        # One gain per (level, orientation)
        self.v = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(levels, 3))
            for _ in range(pods)
        ])

    def match_size(self, x, ref):
        H, W = ref.shape[-2:]
        x = x[..., :H, :W]
        if x.shape[-2] < H or x.shape[-1] < W:
            x = F.pad(x, (0, W - x.shape[-1], 0, H - x.shape[-2]))
        return x

       
    def forward(self, x):
        f0 = x

        # Wavelet transform
        yl, yh = self.wt(f0)
        # yl: [B, C, H, W]
        # yh[j]: [B, C, 3, H, W]

        outputs = []

        for i in range(self.pods):

            yh_mod = []
            for j in range(self.levels):
                sub = yh[j]
                gain = self.v[i][j].view(1, 1, 3, 1, 1)
                yh_mod.append(sub * gain)

            yl_i = self.conv[i](yl)

            yh_mix = []
            for j in range(self.levels):
                bands = []
                for o in range(3):
                    bands.append(self.conv[i](yh_mod[j][:, :, o]))
                yh_mix.append(torch.stack(bands, dim=2))

            yl_i = self.ST[i](yl_i)

            yh_thr = []
            for j in range(self.levels):
                bands = []
                for o in range(3):
                    bands.append(self.ST[i](yh_mix[j][:, :, o]))
                yh_thr.append(torch.stack(bands, dim=2))
            
            f = self.iwt((yl_i, yh_thr))
            f = self.match_size(f, x)
            
            outputs.append(f)

        y = torch.stack(outputs, dim=-1).sum(dim=-1)

        if self.residual:
            y = y + x

        return y

