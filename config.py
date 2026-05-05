#!/usr/bin/env python3
"""
config.py
=========
Centralized configuration for the EPL sonification pipeline.

All parameters are defined here to ensure consistency across scripts.
Modify these values to adapt the pipeline to different datasets or settings.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Input data
DATA_DIR = Path("data")
UDHR_DIR = DATA_DIR / "udhr"
LANGUOID_CSV = DATA_DIR / "languoid.csv"
GEO_CSV = DATA_DIR / "languages_and_dialects_geo.csv"

# Output directories
OUTPUT_BASE = Path("outputs")
PICKLES_DIR = OUTPUT_BASE / "pickles"
SPECTRA_DIR = OUTPUT_BASE / "spectra"
AUDIO_DIR = OUTPUT_BASE / "audio"
ANALYSIS_DIR = OUTPUT_BASE / "analysis"
FIGURES_DIR = OUTPUT_BASE / "figures"

# Create output directories
for dir_path in [PICKLES_DIR, SPECTRA_DIR, AUDIO_DIR, ANALYSIS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET PARAMETERS
# ============================================================================

# Language selection
MIN_FAMILY_SIZE = 5  # Minimum number of languages per family
MACROAREAS = ['North America', 'South America']

# Expected families (for validation)
EXPECTED_FAMILIES = {
    'Quechuan': 13,
    'Arawakan': 8,
    'Mayan': 8,
    'Otomanguean': 8,
    'Panoan': 7
}

# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

# Graph-of-Words parameters
WINDOW_SIZE = 2  # Co-occurrence radius (1 = bigrams, 2 = radius 2)
MIN_TOKEN_LENGTH = 2  # Minimum word length after preprocessing

# ============================================================================
# SPECTRAL ANALYSIS
# ============================================================================

# Laplacian spectrum
FULL_SPECTRUM_THRESHOLD = 1200  # Use dense solver if n ≤ this
SPARSE_K = 800  # Number of smallest eigenvalues for sparse solver

# Heat trace
HEAT_T_MIN = 1e-3
HEAT_T_MAX = 1e1
HEAT_N_POINTS = 300  # Number of logarithmic time points

# ============================================================================
# SONIFICATION PARAMETERS
# ============================================================================

# Audio synthesis
SAMPLE_RATE = 44100  # Hz
DURATION = 10.0  # seconds
FADE_MS = 50  # Fade in/out duration (ms)
EPSILON = 1e-3  # Avoid division by zero

# Frequency ranges (Hz)
FREQ_RANGES = {
    'baseline': (110, 3520),  # 5 octaves
    'narrow': (220, 880),     # 2 octaves
    'wide': (55, 7040)        # 7 octaves
}

# Bin counts
BIN_COUNTS = {
    'sparse': 32,
    'baseline': 64,
    'dense': 128
}

# ============================================================================
# MFCC FEATURE EXTRACTION
# ============================================================================

N_MFCC = 13  # Number of MFCC coefficients
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Hop length for STFT
N_MELS = 128  # Number of Mel bands

# ============================================================================
# CLASSIFICATION & EVALUATION
# ============================================================================

N_FOLDS = 5  # Cross-validation folds
RANDOM_STATE = 42  # For reproducibility
N_ESTIMATORS = 100  # Random Forest trees

# ============================================================================
# VISUALIZATION
# ============================================================================

DPI = 300  # Figure resolution
FIGSIZE_SINGLE = (8, 6)
FIGSIZE_DOUBLE = (12, 5)
FIGSIZE_LARGE = (14, 10)

# Color palette (colorblind-friendly Set2)
FAMILY_COLORS = {
    'Quechuan': '#1f77b4',     # Blue
    'Arawakan': '#ff7f0e',     # Orange
    'Mayan': '#2ca02c',        # Green
    'Otomanguean': '#d62728',  # Red
    'Panoan': '#9467bd'        # Purple
}

# ============================================================================
# SONIFICATION VARIANTS
# ============================================================================

SONIFICATION_STRATEGIES = {
    'V1_linear': {
        'bins': 64,
        'mapping': 'linear',
        'f_min': 110,
        'f_max': 3520,
        'description': 'Linear (baseline)'
    },
    'V2_log': {
        'bins': 64,
        'mapping': 'log',
        'f_min': 110,
        'f_max': 3520,
        'description': 'Logarithmic'
    },
    'V3_sqrt': {
        'bins': 64,
        'mapping': 'sqrt',
        'f_min': 110,
        'f_max': 3520,
        'description': 'Square-root'
    },
    'V4_narrow': {
        'bins': 64,
        'mapping': 'linear',
        'f_min': 220,
        'f_max': 880,
        'description': 'Linear (narrow)'
    },
    'V5_wide': {
        'bins': 64,
        'mapping': 'linear',
        'f_min': 55,
        'f_max': 7040,
        'description': 'Linear (wide)'
    },
    'V6_dense': {
        'bins': 128,
        'mapping': 'linear',
        'f_min': 110,
        'f_max': 3520,
        'description': 'Linear (dense)'
    },
    'V7_sparse': {
        'bins': 32,
        'mapping': 'linear',
        'f_min': 110,
        'f_max': 3520,
        'description': 'Linear (sparse)'
    },
    'V8_mel': {
        'bins': 64,
        'mapping': 'mel',
        'f_min': 110,
        'f_max': 3520,
        'description': 'Mel scale'
    },
    'V9_bark': {
        'bins': 64,
        'mapping': 'bark',
        'f_min': 110,
        'f_max': 3520,
        'description': 'Bark scale'
    },
    'V10_erb': {
        'bins': 64,
        'mapping': 'erb',
        'f_min': 110,
        'f_max': 3520,
        'description': 'ERB scale'
    },
}

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Run basic configuration validation."""
    errors = []
    
    # Check paths
    if not UDHR_DIR.exists():
        errors.append(f"UDHR directory not found: {UDHR_DIR}")
    
    # Check parameters
    if WINDOW_SIZE < 1:
        errors.append(f"WINDOW_SIZE must be >= 1, got {WINDOW_SIZE}")
    
    if N_FOLDS < 2:
        errors.append(f"N_FOLDS must be >= 2, got {N_FOLDS}")
    
    if SAMPLE_RATE not in [22050, 44100, 48000]:
        errors.append(f"Warning: Unusual SAMPLE_RATE: {SAMPLE_RATE}")
    
    if errors:
        print("Configuration errors found:")
        for err in errors:
            print(f"  - {err}")
        return False
    
    print("✓ Configuration validated successfully")
    return True

if __name__ == "__main__":
    validate_config()
