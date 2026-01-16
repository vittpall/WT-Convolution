import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from typing import Tuple

class QuantumWaveletTransform:
    """
    Implements Haar Wavelet Transform using Hadamard gates on quantum circuits.
    
    The Hadamard gate transforms basis states to create superpositions:
    H|0⟩ = (|0⟩ + |1⟩)/√2  (average)
    H|1⟩ = (|0⟩ - |1⟩)/√2  (difference)
    
    This is analogous to the classical Haar transform:
    lo = (even + odd) / 2
    hi = (even - odd) / 2
    """
    
    @staticmethod
    def encode_image_to_amplitudes(image: np.ndarray) -> np.ndarray:
        """
        Encode image data as quantum state amplitudes.
        Flattens and normalizes the image.
        """
        flat = image.flatten().astype(float)
        # Normalize to unit vector (required for quantum states)
        norm = np.linalg.norm(flat)
        if norm == 0:
            norm = 1
        return flat / norm
    
    @staticmethod
    def apply_1d_wavelet_qft(data: np.ndarray, levels: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply 1D quantum wavelet transform using Hadamard gates.
        
        Args:
            data: 1D array of data (must be power of 2 length)
            levels: Number of decomposition levels
            
        Returns:
            lo: Low-frequency (approximation) coefficients
            hi: High-frequency (detail) coefficients
        """
        n = len(data)
        if n & (n - 1) != 0:
            raise ValueError(f"Data length must be power of 2, got {n}")
        
        n_qubits = int(np.log2(n))
        
        # Create quantum circuit
        qr = QuantumRegister(n_qubits, 'q')
        qc = QuantumCircuit(qr)
        
        # Initialize with data as amplitudes
        normalized_data = data / np.linalg.norm(data) if np.linalg.norm(data) > 0 else data
        qc.initialize(normalized_data, qr)
        
        # Apply Hadamard gates in pyramid structure for wavelet transform
        # Each level applies H to half the qubits
        for level in range(min(levels, n_qubits)):
            # Apply Hadamard to the most significant qubit at this level
            qc.h(n_qubits - 1 - level)
        
        # Get final statevector
        sv = Statevector(qc)
        amplitudes = sv.data
        
        # Split into low and high frequency components
        # After Hadamard, first half = approximation, second half = detail
        mid = n // 2
        lo = amplitudes[:mid] * np.sqrt(2)  # Scale back
        hi = amplitudes[mid:] * np.sqrt(2)
        
        return np.abs(lo), np.abs(hi)
    
    @staticmethod
    def quantum_wavelet_2d(image: np.ndarray, levels: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply 2D quantum wavelet transform using separable 1D transforms.
        
        Returns:
            LL: Approximation (low-low)
            LH: Horizontal detail (low-high)
            HL: Vertical detail (high-low)
            HH: Diagonal detail (high-high)
        """
        rows, cols = image.shape
        
        # Row-wise transform
        row_results = []
        for i in range(rows):
            row = image[i, :]
            if np.linalg.norm(row) > 0:
                lo, hi = QuantumWaveletTransform.apply_1d_wavelet_qft(row, levels)
                row_results.append((lo, hi))
            else:
                row_results.append((row[:cols//2], row[cols//2:]))
        
        # Separate low and high frequency rows
        L_rows = np.array([r[0] for r in row_results])
        H_rows = np.array([r[1] for r in row_results])
        
        # Column-wise transform on low frequencies
        LL_cols = []
        LH_cols = []
        for j in range(cols // 2):
            col = L_rows[:, j]
            if np.linalg.norm(col) > 0:
                lo, hi = QuantumWaveletTransform.apply_1d_wavelet_qft(col, levels)
                LL_cols.append(lo)
                LH_cols.append(hi)
            else:
                LL_cols.append(col[:rows//2])
                LH_cols.append(col[rows//2:])
        
        # Column-wise transform on high frequencies
        HL_cols = []
        HH_cols = []
        for j in range(cols // 2):
            col = H_rows[:, j]
            if np.linalg.norm(col) > 0:
                lo, hi = QuantumWaveletTransform.apply_1d_wavelet_qft(col, levels)
                HL_cols.append(lo)
                HH_cols.append(hi)
            else:
                HL_cols.append(col[:rows//2])
                HH_cols.append(col[rows//2:])
        
        LL = np.array(LL_cols).T
        LH = np.array(LH_cols).T
        HL = np.array(HL_cols).T
        HH = np.array(HH_cols).T
        
        return LL, LH, HL, HH


# Demo and comparison
def demo_quantum_wavelet():
    """Compare classical and quantum wavelet transforms"""
    
    # Create test image (power of 2 dimensions for quantum)
    x = np.random.randint(0, 20, size=(4, 4))
    print("Input Image:")
    print(x)
    print()
    
    # Classical approach (from your original code)
    def classical_haar_1d(data):
        evens, odds = data[::2], data[1::2]
        lo = evens + odds
        hi = evens - odds
        return lo, hi
    
    # Apply classical to first row
    print("Classical Haar Transform (first row):")
    row = x[0, :]
    lo_c, hi_c = classical_haar_1d(row)
    print(f"Low:  {lo_c}")
    print(f"High: {hi_c}")
    print()
    
    # Quantum approach
    print("Quantum Hadamard Transform (first row):")
    qwt = QuantumWaveletTransform()
    lo_q, hi_q = qwt.apply_1d_wavelet_qft(row.astype(float), levels=1)
    
    # Scale quantum results to match classical (quantum gives normalized)
    scale = np.linalg.norm(row)
    lo_q_scaled = lo_q * scale * np.sqrt(2)
    hi_q_scaled = hi_q * scale * np.sqrt(2)
    
    print(f"Low:  {lo_q_scaled}")
    print(f"High: {hi_q_scaled}")
    print()
    
    # 2D Transform
    print("2D Quantum Wavelet Transform:")
    LL, LH, HL, HH = qwt.quantum_wavelet_2d(x.astype(float), levels=1)
    print(f"LL (Approximation):\n{LL}")
    print(f"\nLH (Horizontal Detail):\n{LH}")
    print(f"\nHL (Vertical Detail):\n{HL}")
    print(f"\nHH (Diagonal Detail):\n{HH}")
    
    # Visualize circuit for small example
    print("\n" + "="*50)
    print("Example Quantum Circuit for 4-element transform:")
    print("="*50)
    n = 4
    qr = QuantumRegister(2, 'q')
    qc = QuantumCircuit(qr)
    qc.initialize([1, 2, 3, 4] / np.linalg.norm([1, 2, 3, 4]), qr)
    qc.h(1)  # Apply Hadamard to most significant qubit
    print(qc)

if __name__ == "__main__":
    demo_quantum_wavelet()