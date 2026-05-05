#!/usr/bin/env python3
"""
utils.py
========
Shared utility functions used across multiple pipeline scripts.
"""

import numpy as np
import librosa
from pathlib import Path

# ============================================================================
# FREQUENCY MAPPING FUNCTIONS
# ============================================================================

def hz_to_mel(f):
    """Convert Hz to Mel scale."""
    return 2595 * np.log10(1 + f / 700)

def mel_to_hz(m):
    """Convert Mel to Hz."""
    return 700 * (10 ** (m / 2595) - 1)

def hz_to_bark(f):
    """Convert Hz to Bark scale (Zwicker & Terhardt 1980)."""
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)

def bark_to_hz(z):
    """
    Convert Bark to Hz (approximate inverse).
    Note: This is a simplified inverse, not exact.
    """
    return 600 * np.sinh(z / 6)

def hz_to_erb(f):
    """Convert Hz to ERB (Equivalent Rectangular Bandwidth)."""
    return 21.4 * np.log10(1 + 0.00437 * f)

def erb_to_hz(e):
    """Convert ERB to Hz."""
    return (10 ** (e / 21.4) - 1) / 0.00437

# ============================================================================
# AUDIO UTILITIES
# ============================================================================

def normalize_audio(audio, target_peak=1.0):
    """
    Normalize audio to target peak amplitude.
    
    Args:
        audio: Audio array
        target_peak: Target peak amplitude (default: 1.0)
    
    Returns:
        Normalized audio array
    """
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio * (target_peak / peak)
    return audio

def apply_fade(audio, sample_rate, fade_ms=50):
    """
    Apply linear fade-in and fade-out.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate in Hz
        fade_ms: Fade duration in milliseconds
    
    Returns:
        Audio with fade applied
    """
    fade_samples = int(fade_ms * sample_rate / 1000)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out
    
    return audio

def extract_mfcc_stats(audio_path, n_mfcc=13, n_fft=2048, hop_length=512, n_mels=128):
    """
    Extract MFCC features with temporal statistics.
    
    Args:
        audio_path: Path to audio file
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of Mel bands
    
    Returns:
        Feature vector (2 * n_mfcc dimensions: mean + std)
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Temporal statistics
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    # Concatenate
    features = np.concatenate([mfcc_mean, mfcc_std])
    
    return features

# ============================================================================
# SPECTRAL UTILITIES
# ============================================================================

def compute_histogram_features(eigenvalues, bins=64, range_val=(0, 2)):
    """
    Convert eigenvalue distribution to histogram features.
    
    Args:
        eigenvalues: Array of eigenvalues
        bins: Number of histogram bins
        range_val: Value range for binning
    
    Returns:
        hist: Probability distribution (normalized)
        bin_centers: Center value of each bin
    """
    hist, bin_edges = np.histogram(eigenvalues, bins=bins, range=range_val, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize to probabilities
    hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
    
    return hist, bin_centers

# ============================================================================
# FILE I/O UTILITIES
# ============================================================================

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)

def load_language_metadata(csv_path, clean_langs=None):
    """
    Load and filter language metadata.
    
    Args:
        csv_path: Path to CSV file
        clean_langs: Optional list of ISO codes to filter
    
    Returns:
        DataFrame with language metadata
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    if clean_langs is not None:
        df = df[df['iso639P3code'].isin(clean_langs)].copy()
    
    df = df.sort_values('iso639P3code').reset_index(drop=True)
    
    return df

# ============================================================================
# PROGRESS REPORTING
# ============================================================================

class ProgressReporter:
    """Simple progress reporter for long-running operations."""
    
    def __init__(self, total, prefix="Progress"):
        self.total = total
        self.current = 0
        self.prefix = prefix
    
    def update(self, message=""):
        self.current += 1
        pct = 100 * self.current / self.total
        print(f"\r{self.prefix}: [{self.current}/{self.total}] {pct:.1f}% {message}", 
              end='', flush=True)
    
    def finish(self):
        print()  # Newline

# ============================================================================
# VALIDATION
# ============================================================================

def validate_audio_file(path, expected_sr=44100, expected_duration=10.0, tolerance=0.1):
    """
    Validate audio file properties.
    
    Args:
        path: Path to audio file
        expected_sr: Expected sample rate
        expected_duration: Expected duration in seconds
        tolerance: Tolerance for duration check
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        y, sr = librosa.load(path, sr=None)
        duration = len(y) / sr
        
        if sr != expected_sr:
            print(f"Warning: Sample rate mismatch ({sr} vs {expected_sr})")
            return False
        
        if abs(duration - expected_duration) > tolerance:
            print(f"Warning: Duration mismatch ({duration:.2f}s vs {expected_duration}s)")
            return False
        
        return True
    
    except Exception as e:
        print(f"Error validating {path}: {e}")
        return False

# ============================================================================
# STATISTICS
# ============================================================================

def inter_intra_distances(distance_matrix, labels):
    """
    Compute inter-family and intra-family distance statistics.
    
    Args:
        distance_matrix: NxN distance matrix
        labels: Family labels
    
    Returns:
        dict with 'inter_mean', 'intra_mean', 'ratio'
    """
    n = len(labels)
    
    inter = []
    intra = []
    
    for i in range(n):
        for j in range(i+1, n):
            if labels[i] == labels[j]:
                intra.append(distance_matrix[i, j])
            else:
                inter.append(distance_matrix[i, j])
    
    inter_mean = np.mean(inter) if inter else 0
    intra_mean = np.mean(intra) if intra else 1
    ratio = inter_mean / intra_mean if intra_mean > 0 else 0
    
    return {
        'inter_mean': inter_mean,
        'intra_mean': intra_mean,
        'ratio': ratio,
        'inter_std': np.std(inter) if inter else 0,
        'intra_std': np.std(intra) if intra else 0
    }

# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Utility functions loaded successfully.")
    print("\nAvailable functions:")
    print("  - Frequency mappings: hz_to_mel, mel_to_hz, hz_to_bark, etc.")
    print("  - Audio: normalize_audio, apply_fade, extract_mfcc_stats")
    print("  - Spectral: compute_histogram_features")
    print("  - I/O: ensure_dir, load_language_metadata")
    print("  - Stats: inter_intra_distances")
