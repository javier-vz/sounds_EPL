#!/usr/bin/env python3
"""
04_synthesize_sounds.py
=======================
Synthesizes audio signals from Laplacian spectra using additive synthesis
and heat-trace modulation.

For each language and each sonification strategy:
  - Maps eigenvalue histogram to frequency bins
  - Generates additive synthesis signal: x(t) = Σ aⱼ sin(2πfⱼt)
  - Exports to WAV file (44100 Hz, 10s, mono)

Strategies:
  V1-V10: Histogram-based with different mappings
  HT: Heat-trace amplitude modulation

Outputs: outputs/audio/{variant}/{iso}.wav
"""

import os
import pickle
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy import signal

# ============================================================================
# CONFIGURATION
# ============================================================================

SPECTRA_DIR = "outputs/spectra"
LANGS_PKL = "outputs/pickles/clean_languages.pkl"
OUTPUT_BASE = "outputs/audio"

# Audio parameters
SAMPLE_RATE = 44100
DURATION = 10.0  # seconds
FADE_MS = 50  # fade in/out duration

# Synthesis parameters
EPSILON = 1e-3  # avoid division by zero

# ============================================================================
# SONIFICATION STRATEGIES
# ============================================================================

STRATEGIES = {
    # Baseline: 64 bins, linear mapping, 110-3520 Hz
    'V1_linear': {
        'bins': 64,
        'mapping': 'linear',
        'f_min': 110,
        'f_max': 3520,
        'description': 'Linear (baseline)'
    },
    
    # V2: Logarithmic mapping
    'V2_log': {
        'bins': 64,
        'mapping': 'log',
        'f_min': 110,
        'f_max': 3520,
        'description': 'Logarithmic'
    },
    
    # V3: Square-root mapping
    'V3_sqrt': {
        'bins': 64,
        'mapping': 'sqrt',
        'f_min': 110,
        'f_max': 3520,
        'description': 'Square-root'
    },
    
    # V4: Narrow range
    'V4_narrow': {
        'bins': 64,
        'mapping': 'linear',
        'f_min': 220,
        'f_max': 880,
        'description': 'Linear (narrow)'
    },
    
    # V5: Wide range
    'V5_wide': {
        'bins': 64,
        'mapping': 'linear',
        'f_min': 55,
        'f_max': 7040,
        'description': 'Linear (wide)'
    },
    
    # V6: Dense (128 bins)
    'V6_dense': {
        'bins': 128,
        'mapping': 'linear',
        'f_min': 110,
        'f_max': 3520,
        'description': 'Linear (dense)'
    },
    
    # V7: Sparse (32 bins) ⭐ BEST
    'V7_sparse': {
        'bins': 32,
        'mapping': 'linear',
        'f_min': 110,
        'f_max': 3520,
        'description': 'Linear (sparse)'
    },
    
    # V8: Mel scale
    'V8_mel': {
        'bins': 64,
        'mapping': 'mel',
        'f_min': 110,
        'f_max': 3520,
        'description': 'Mel scale'
    },
    
    # V9: Bark scale
    'V9_bark': {
        'bins': 64,
        'mapping': 'bark',
        'f_min': 110,
        'f_max': 3520,
        'description': 'Bark scale'
    },
    
    # V10: ERB scale
    'V10_erb': {
        'bins': 64,
        'mapping': 'erb',
        'f_min': 110,
        'f_max': 3520,
        'description': 'ERB scale'
    },
}

# ============================================================================
# HELPER FUNCTIONS: FREQUENCY MAPPINGS
# ============================================================================

def linear_map(bin_centers, f_min, f_max):
    """Linear frequency mapping."""
    # Normalize to [0, 1]
    normalized = (bin_centers - 0) / 2.0
    # Map to [f_min, f_max]
    return f_min + normalized * (f_max - f_min)

def log_map(bin_centers, f_min, f_max):
    """Logarithmic frequency mapping."""
    normalized = (bin_centers - 0) / 2.0
    # Logarithmic scale
    log_f_min = np.log(f_min)
    log_f_max = np.log(f_max)
    return np.exp(log_f_min + normalized * (log_f_max - log_f_min))

def sqrt_map(bin_centers, f_min, f_max):
    """Square-root frequency mapping."""
    normalized = (bin_centers - 0) / 2.0
    # Square-root scale
    sqrt_f_min = np.sqrt(f_min)
    sqrt_f_max = np.sqrt(f_max)
    return (sqrt_f_min + normalized * (sqrt_f_max - sqrt_f_min)) ** 2

def hz_to_mel(f):
    """Convert Hz to Mel scale."""
    return 2595 * np.log10(1 + f / 700)

def mel_to_hz(m):
    """Convert Mel to Hz."""
    return 700 * (10 ** (m / 2595) - 1)

def mel_map(bin_centers, f_min, f_max):
    """Mel scale frequency mapping."""
    normalized = (bin_centers - 0) / 2.0
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_vals = mel_min + normalized * (mel_max - mel_min)
    return mel_to_hz(mel_vals)

def hz_to_bark(f):
    """Convert Hz to Bark scale."""
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)

def bark_to_hz(z):
    """Convert Bark to Hz (approximate inverse)."""
    # Simplified inverse
    return 600 * np.sinh(z / 6)

def bark_map(bin_centers, f_min, f_max):
    """Bark scale frequency mapping."""
    normalized = (bin_centers - 0) / 2.0
    bark_min = hz_to_bark(f_min)
    bark_max = hz_to_bark(f_max)
    bark_vals = bark_min + normalized * (bark_max - bark_min)
    return bark_to_hz(bark_vals)

def hz_to_erb(f):
    """Convert Hz to ERB (Equivalent Rectangular Bandwidth)."""
    return 21.4 * np.log10(1 + 0.00437 * f)

def erb_to_hz(e):
    """Convert ERB to Hz."""
    return (10 ** (e / 21.4) - 1) / 0.00437

def erb_map(bin_centers, f_min, f_max):
    """ERB scale frequency mapping."""
    normalized = (bin_centers - 0) / 2.0
    erb_min = hz_to_erb(f_min)
    erb_max = hz_to_erb(f_max)
    erb_vals = erb_min + normalized * (erb_max - erb_min)
    return erb_to_hz(erb_vals)

# ============================================================================
# SYNTHESIS FUNCTIONS
# ============================================================================

def histogram_to_audio(eigvals, bins, mapping, f_min, f_max):
    """
    Convert eigenvalue histogram to audio via additive synthesis.
    
    Args:
        eigvals: Eigenvalue array
        bins: Number of histogram bins
        mapping: Frequency mapping function name
        f_min, f_max: Frequency range (Hz)
    
    Returns:
        audio: Synthesized audio array (normalized to [-1, 1])
    """
    # Create histogram
    hist, bin_edges = np.histogram(eigvals, bins=bins, range=(0, 2), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize histogram to probabilities
    p_j = hist / np.sum(hist) if np.sum(hist) > 0 else hist
    
    # Map bin centers to frequencies
    mapping_functions = {
        'linear': linear_map,
        'log': log_map,
        'sqrt': sqrt_map,
        'mel': mel_map,
        'bark': bark_map,
        'erb': erb_map,
    }
    
    freq_func = mapping_functions[mapping]
    f_j = freq_func(bin_centers, f_min, f_max)
    
    # Compute amplitudes with de-emphasis: a_j = p_j / sqrt(c_j + ε)
    a_j = p_j / np.sqrt(bin_centers + EPSILON)
    
    # Normalize amplitudes
    if np.max(a_j) > 0:
        a_j = a_j / np.max(a_j)
    
    # Generate time array
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    
    # Additive synthesis: x(t) = Σ a_j sin(2πf_j t)
    audio = np.zeros_like(t)
    for amp, freq in zip(a_j, f_j):
        if amp > 0 and freq > 0:
            audio += amp * np.sin(2 * np.pi * freq * t)
    
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    return audio

def heat_trace_to_audio(t_grid, Z_vals, carrier_freq=440):
    """
    Convert heat trace to audio via amplitude modulation.
    
    Args:
        t_grid: Heat trace time points
        Z_vals: Heat trace values Z(t)
        carrier_freq: Carrier frequency (Hz)
    
    Returns:
        audio: Synthesized audio array
    """
    # Normalize heat trace to [0, 1]
    Z_norm = (Z_vals - Z_vals[-1]) / (Z_vals[0] - Z_vals[-1])
    
    # Interpolate to audio sample rate
    t_audio = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    
    # Map heat trace time to audio time
    t_heat_normalized = (t_grid - t_grid[0]) / (t_grid[-1] - t_grid[0])
    t_heat_audio = t_heat_normalized * DURATION
    
    # Interpolate envelope
    envelope = np.interp(t_audio, t_heat_audio, Z_norm)
    
    # Generate carrier
    carrier = np.sin(2 * np.pi * carrier_freq * t_audio)
    
    # Amplitude modulation
    audio = envelope * carrier
    
    return audio

def apply_fade(audio, fade_samples):
    """Apply linear fade-in and fade-out."""
    fade = np.linspace(0, 1, fade_samples)
    audio[:fade_samples] *= fade
    audio[-fade_samples:] *= fade[::-1]
    return audio

# ============================================================================
# MAIN PIPELINE
# ============================================================================

print(f"{'='*70}")
print("AUDIO SYNTHESIS FROM LAPLACIAN SPECTRA")
print(f"{'='*70}\n")

# Load language list
with open(LANGS_PKL, 'rb') as f:
    languages = pickle.load(f)

print(f"Languages to process: {len(languages)}")
print(f"Strategies: {len(STRATEGIES)}")
print(f"Sample rate: {SAMPLE_RATE} Hz")
print(f"Duration: {DURATION} s\n")

# Calculate fade samples
fade_samples = int(FADE_MS * SAMPLE_RATE / 1000)

# Process each strategy
for variant_name, params in STRATEGIES.items():
    print(f"\n{'─'*70}")
    print(f"STRATEGY: {variant_name} — {params['description']}")
    print(f"{'─'*70}")
    
    # Create output directory
    output_dir = Path(OUTPUT_BASE) / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each language
    for iso_code in languages:
        # Load eigenvalues
        eigvals_path = Path(SPECTRA_DIR) / f"{iso_code}_eigvals.npy"
        eigvals = np.load(eigvals_path)
        
        # Synthesize audio
        audio = histogram_to_audio(
            eigvals,
            bins=params['bins'],
            mapping=params['mapping'],
            f_min=params['f_min'],
            f_max=params['f_max']
        )
        
        # Apply fade
        audio = apply_fade(audio, fade_samples)
        
        # Save
        output_path = output_dir / f"{iso_code}.wav"
        sf.write(output_path, audio, SAMPLE_RATE, subtype='PCM_16')
        
        print(f"  ✓ {iso_code:8s} → {output_path}")

# ============================================================================
# HEAT TRACE VARIANT
# ============================================================================

print(f"\n{'─'*70}")
print(f"STRATEGY: HT — Heat trace")
print(f"{'─'*70}")

output_dir = Path(OUTPUT_BASE) / "HT_heattrace"
output_dir.mkdir(parents=True, exist_ok=True)

for iso_code in languages:
    # Load heat trace
    ht_path = Path(SPECTRA_DIR) / f"{iso_code}_heattrace.npz"
    ht_data = np.load(ht_path)
    t_grid = ht_data['t']
    Z_vals = ht_data['Z']
    
    # Synthesize
    audio = heat_trace_to_audio(t_grid, Z_vals, carrier_freq=440)
    
    # Apply fade
    audio = apply_fade(audio, fade_samples)
    
    # Save
    output_path = output_dir / f"{iso_code}.wav"
    sf.write(output_path, audio, SAMPLE_RATE, subtype='PCM_16')
    
    print(f"  ✓ {iso_code:8s} → {output_path}")

print(f"\n{'='*70}")
print(f"COMPLETED: Audio files generated for {len(STRATEGIES) + 1} strategies")
print(f"{'='*70}\n")
