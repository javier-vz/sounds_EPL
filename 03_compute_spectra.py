#!/usr/bin/env python3
"""
03_compute_spectra.py
=====================
Computes Laplacian spectra and heat traces for all language graphs.

For each language:
  - Computes normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
  - Extracts eigenvalues (full spectrum if n ≤ 1200, else k=800 smallest)
  - Computes heat trace: Z(t) = Σ exp(-t·λᵢ) over logarithmic time grid

Outputs:
  - outputs/spectra/{iso}_eigvals.npy: Eigenvalue array
  - outputs/spectra/{iso}_heattrace.npz: Heat trace time grid and values
"""

import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.sparse import linalg as sp_linalg

# ============================================================================
# CONFIGURATION
# ============================================================================

GRAPHS_PKL = "outputs/pickles/graphs.pkl"
OUTPUT_DIR = "outputs/spectra"
FULL_SPECTRUM_THRESHOLD = 1200  # Use dense solver if n ≤ this
SPARSE_K = 800  # Number of smallest eigenvalues for sparse solver

# Heat trace parameters
HEAT_T_MIN = 1e-3
HEAT_T_MAX = 1e1
HEAT_N_POINTS = 300

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_normalized_laplacian_spectrum(G, full_threshold=1200, sparse_k=800):
    """
    Compute eigenvalues of normalized Laplacian.
    
    L = I - D^(-1/2) A D^(-1/2)
    
    Args:
        G: NetworkX graph
        full_threshold: Use dense solver if n ≤ this
        sparse_k: Number of smallest eigenvalues for sparse solver
    
    Returns:
        np.array of eigenvalues in ascending order
    """
    n = G.number_of_nodes()
    
    if n == 0:
        return np.array([])
    
    # Get normalized Laplacian matrix (sparse)
    L = nx.normalized_laplacian_matrix(G).astype(float)
    
    # Choose solver based on graph size
    if n <= full_threshold:
        # Dense: get all eigenvalues
        L_dense = L.toarray()
        eigvals = np.linalg.eigvalsh(L_dense)
    else:
        # Sparse: get k smallest eigenvalues
        k = min(sparse_k, n - 2)  # scipy requires k < n-1
        eigvals, _ = sp_linalg.eigsh(L, k=k, which='SM')
    
    # Sort in ascending order
    eigvals = np.sort(eigvals)
    
    # Clip to [0, 2] (numerical stability)
    eigvals = np.clip(eigvals, 0, 2)
    
    return eigvals

def compute_heat_trace(eigvals, t_min=1e-3, t_max=1e1, n_points=300):
    """
    Compute heat trace Z(t) = Σ exp(-t·λᵢ).
    
    Args:
        eigvals: Eigenvalue array
        t_min, t_max: Time range
        n_points: Number of time points (logarithmic grid)
    
    Returns:
        t_grid: Time points
        Z_vals: Heat trace values
    """
    t_grid = np.logspace(np.log10(t_min), np.log10(t_max), n_points)
    Z_vals = np.array([np.sum(np.exp(-t * eigvals)) for t in t_grid])
    return t_grid, Z_vals

# ============================================================================
# MAIN PIPELINE
# ============================================================================

print(f"{'='*70}")
print("LAPLACIAN SPECTRUM COMPUTATION")
print(f"{'='*70}\n")

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load graphs
print(f"Loading graphs from {GRAPHS_PKL}...")
with open(GRAPHS_PKL, 'rb') as f:
    graphs = pickle.load(f)

print(f"Loaded {len(graphs)} graphs\n")

# Process each graph
for iso_code, G in graphs.items():
    n = G.number_of_nodes()
    print(f"Processing {iso_code:8s} | Nodes: {n:4d} ... ", end='', flush=True)
    
    # Compute spectrum
    eigvals = compute_normalized_laplacian_spectrum(
        G, 
        full_threshold=FULL_SPECTRUM_THRESHOLD,
        sparse_k=SPARSE_K
    )
    
    # Save eigenvalues
    eigvals_path = Path(OUTPUT_DIR) / f"{iso_code}_eigvals.npy"
    np.save(eigvals_path, eigvals)
    
    # Compute heat trace
    t_grid, Z_vals = compute_heat_trace(
        eigvals,
        t_min=HEAT_T_MIN,
        t_max=HEAT_T_MAX,
        n_points=HEAT_N_POINTS
    )
    
    # Save heat trace
    heattrace_path = Path(OUTPUT_DIR) / f"{iso_code}_heattrace.npz"
    np.savez(heattrace_path, t=t_grid, Z=Z_vals)
    
    print(f"✓ Spectrum: {len(eigvals)} eigenvalues | λ_max: {eigvals[-1]:.3f}")

print(f"\n{'='*70}")
print(f"COMPLETED: Spectra computed for {len(graphs)} languages")
print(f"{'='*70}\n")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("SPECTRUM STATISTICS:")
print(f"  Full spectrum threshold: {FULL_SPECTRUM_THRESHOLD} nodes")
print(f"  Sparse k: {SPARSE_K} eigenvalues")
print(f"  Heat trace: {HEAT_N_POINTS} points, t ∈ [{HEAT_T_MIN}, {HEAT_T_MAX}]")
