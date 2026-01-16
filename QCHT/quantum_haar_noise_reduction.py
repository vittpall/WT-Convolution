#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Hadamard Gate Implementation
Exact match for the CORRECT Haar Wavelet Transform
"""
import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def quantum_hadamard_pair(even, odd):
    """
    Apply quantum Hadamard transform to a single (even, odd) pair.
    Returns (lo, hi) where:
        lo = (even + odd) / sqrt(2)
        hi = (even - odd) / sqrt(2)
    """
    even_val = float(even)
    odd_val = float(odd)
    
    if np.allclose([even_val, odd_val], 0):
        return 0.0, 0.0
    
    pair = np.array([even_val, odd_val])
    norm_factor = np.linalg.norm(pair)
    normalized_pair = pair / norm_factor
    
    qc = QuantumCircuit(1)
    qc.initialize(normalized_pair, 0)
    qc.h(0)
    
    sv = Statevector(qc)
    amplitudes = sv.data
    
    lo = amplitudes[0].real * norm_factor * np.sqrt(2)
    hi = amplitudes[1].real * norm_factor * np.sqrt(2)
    
    norm = 1.0 / np.sqrt(2)
    lo = lo * norm
    hi = hi * norm
    
    return lo, hi


def quantum_haar_transform_1d(u, axis=-1, inverse=False):
    """
    Quantum implementation using Hadamard gates.
    Exactly matches the CORRECT hierarchical Haar transform.
    """
    if axis != -1:
        u = torch.transpose(u, -1, axis)
    
    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    
    x = u.clone()
    
    if not inverse:
        length = n
        for level in range(m):
            half = length // 2
            temp = x[..., :length].clone()
            
            shape_prefix = temp.shape[:-1]
            even_data = temp[..., ::2]
            odd_data = temp[..., 1::2]
            
            even_flat = even_data.reshape(-1)
            odd_flat = odd_data.reshape(-1)
            
            lo_results = torch.zeros_like(even_flat)
            hi_results = torch.zeros_like(odd_flat)
            
            for i in range(len(even_flat)):
                lo, hi = quantum_hadamard_pair(even_flat[i], odd_flat[i])
                lo_results[i] = lo
                hi_results[i] = hi
            
            lo_results = lo_results.reshape(shape_prefix + (half,))
            hi_results = hi_results.reshape(shape_prefix + (half,))
            
            x[..., :half] = lo_results
            x[..., half:length] = hi_results
            
            length = half
    else:
        length = 2
        for level in range(m):
            half = length // 2
            temp = x[..., :length].clone()
            approx = temp[..., :half]
            detail = temp[..., half:length]
            
            shape_prefix = approx.shape[:-1]
            approx_flat = approx.reshape(-1)
            detail_flat = detail.reshape(-1)
            
            even_results = torch.zeros_like(approx_flat)
            odd_results = torch.zeros_like(detail_flat)
            
            for i in range(len(approx_flat)):
                even_val, odd_val = quantum_hadamard_pair(approx_flat[i], detail_flat[i])
                even_results[i] = even_val
                odd_results[i] = odd_val
            
            even_results = even_results.reshape(shape_prefix + (half,))
            odd_results = odd_results.reshape(shape_prefix + (half,))
            
            x[..., :length:2] = even_results
            x[..., 1:length:2] = odd_results
            
            length *= 2
    
    if axis != -1:
        x = torch.transpose(x, -1, axis)
    
    return x


def classical_haar_transform_1d(u, axis=-1, inverse=False):
    """Correct classical Haar transform for comparison"""
    if axis != -1:
        u = torch.transpose(u, -1, axis)

    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    x = u.clone()
    norm = 1.0 / np.sqrt(2)

    if not inverse:
        length = n
        for _ in range(m):
            half = length // 2
            temp = x[..., :length].clone()
            x[..., :half] = norm * (temp[..., ::2] + temp[..., 1::2])
            x[..., half:length] = norm * (temp[..., ::2] - temp[..., 1::2])
            length = half
    else:
        length = 2
        for _ in range(m):
            half = length // 2
            temp = x[..., :length].clone()
            approx = temp[..., :half]
            detail = temp[..., half:length]
            x[..., :length:2] = norm * (approx + detail)
            x[..., 1:length:2] = norm * (approx - detail)
            length *= 2

    if axis != -1:
        x = torch.transpose(x, -1, axis)

    return x


if __name__ == "__main__":
    # Single forward comparison
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    
    y_classical = classical_haar_transform_1d(x.clone())
    y_quantum = quantum_haar_transform_1d(x.clone())
    
    print("Input:     ", x)
    print("Classical: ", y_classical)
    print("Quantum:   ", y_quantum)
    print("Max diff:  ", torch.abs(y_classical - y_quantum).max().item())