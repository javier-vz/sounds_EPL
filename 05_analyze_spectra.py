#!/usr/bin/env python3
"""
05_analyze_spectra.py
=====================
Analysis for the EPL paper subset only (44 languages, 5 families)

Outputs:
  - distance_matrices/: distance matrices for each variant
  - mfcc_features/: MFCC features per variant
  - classification_results.csv
  - classification_results.pkl
  - classification_results_with_f1.csv
"""

import pickle
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import MDS
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

# ============================================================================
# CONFIGURATION
# ============================================================================

AUDIO_BASE = Path("outputs/audio")
SPECTRA_DIR = Path("outputs/spectra")
AMERICAS_CSV = "americas_families.csv"
LANGS_PKL = "outputs/pickles/paper_languages_44.pkl"
OUTPUT_DIR = Path("outputs/analysis")

# MFCC parameters fixed by manuscript
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

# Classification parameters
N_FOLDS = 5
RANDOM_STATE = 42
N_ESTIMATORS = 100

PAPER_FAMILIES = {
    "quec1387": "Quechuan",
    "araw1281": "Arawakan",
    "maya1287": "Mayan",
    "otom1299": "Otomanguean",
    "pano1259": "Panoan",
}

EXPECTED_COUNTS = {
    "quec1387": 13,
    "araw1281": 8,
    "maya1287": 8,
    "otom1299": 8,
    "pano1259": 7,
}

EXPECTED_TOTAL = 44

VARIANT_LABELS = {
    "V1": "Linear (baseline)",
    "V2": "Logarithmic",
    "V3": "Square-root",
    "V4": "Narrow range",
    "V5": "Wide range",
    "V6": "Dense (128 bins)",
    "V7": "Sparse (32 bins)",
    "V8": "Mel scale",
    "V9": "Bark scale",
    "V10": "ERB scale",
    "HT": "Heat trace",
}


# ============================================================================
# HELPERS
# ============================================================================

def extract_mfcc_features(audio_path):
    """
    Returns 26-dim feature vector = mean(13 MFCC) + std(13 MFCC)
    """
    y, sr = librosa.load(audio_path, sr=None)

    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])


def compute_wasserstein_distance_matrix(spectra_dict):
    iso_codes = sorted(spectra_dict.keys())
    n = len(iso_codes)
    dist_matrix = np.zeros((n, n))

    for i, iso1 in enumerate(iso_codes):
        for j in range(i + 1, n):
            iso2 = iso_codes[j]
            dist = wasserstein_distance(spectra_dict[iso1], spectra_dict[iso2])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix, iso_codes


def evaluate_clustering(distance_matrix, labels):
    sil = silhouette_score(distance_matrix, labels, metric="precomputed")

    mds = MDS(
        n_components=10,
        dissimilarity="precomputed",
        random_state=RANDOM_STATE,
    )
    X_embedded = mds.fit_transform(distance_matrix)

    ch = calinski_harabasz_score(X_embedded, labels)
    db = davies_bouldin_score(X_embedded, labels)

    return {
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
    }


def inter_intra_ratio(distance_matrix, labels):
    inter_dists = []
    intra_dists = []

    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                intra_dists.append(distance_matrix[i, j])
            else:
                inter_dists.append(distance_matrix[i, j])

    mean_inter = float(np.mean(inter_dists)) if inter_dists else 0.0
    mean_intra = float(np.mean(intra_dists)) if intra_dists else 1.0
    ratio = mean_inter / mean_intra if mean_intra > 0 else 0.0

    return ratio, mean_inter, mean_intra


def evaluate_classifier(X, y):
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = cross_validate(
        rf,
        X,
        y,
        cv=cv,
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
        return_train_score=False,
    )

    return {
        "accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "accuracy_std": float(np.std(scores["test_accuracy"])),
        "macro_f1_mean": float(np.mean(scores["test_macro_f1"])),
        "macro_f1_std": float(np.std(scores["test_macro_f1"])),
    }


def canonical_variant_sort_key(name):
    if name == "Spectral_baseline":
        return (0, 0)
    if name.startswith("V"):
        try:
            return (1, int(name[1:]))
        except ValueError:
            return (1, 999)
    if name == "HT":
        return (2, 0)
    return (3, name)


# ============================================================================
# MAIN
# ============================================================================

print("=" * 70)
print("STATISTICAL ANALYSIS OF SONIFIED SPECTRA (PAPER SUBSET)")
print("=" * 70)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "distance_matrices").mkdir(exist_ok=True)
(OUTPUT_DIR / "mfcc_features").mkdir(exist_ok=True)

# Load metadata
langs_df = pd.read_csv(AMERICAS_CSV)
with open(LANGS_PKL, "rb") as f:
    paper_langs = pickle.load(f)

langs_df = langs_df[langs_df["iso_code"].isin(paper_langs)].copy()
langs_df = langs_df[langs_df["family"].isin(PAPER_FAMILIES)].copy()
langs_df = langs_df.sort_values("iso_code").reset_index(drop=True)

observed_counts = langs_df["family"].value_counts().to_dict()
if observed_counts != EXPECTED_COUNTS or len(langs_df) != EXPECTED_TOTAL:
    raise ValueError(
        f"Subset incorrecto para el paper. Observado: {observed_counts}, "
        f"total={len(langs_df)}"
    )

# Family labels in fixed order
family_order = ["quec1387", "araw1281", "maya1287", "otom1299", "pano1259"]
family_to_id = {fam: i for i, fam in enumerate(family_order)}
langs_df["family_id"] = langs_df["family"].map(family_to_id)
langs_df["family_name"] = langs_df["family"].map(PAPER_FAMILIES)

print("\nSubset del paper confirmado:")
for fam in family_order:
    print(f"  {PAPER_FAMILIES[fam]:12s}: {EXPECTED_COUNTS[fam]}")
print(f"  Total: {len(langs_df)}")

# ============================================================================
# SPECTRAL BASELINE
# ============================================================================

print("\n" + "─" * 70)
print("SPECTRAL BASELINE (Raw Eigenvalues)")
print("─" * 70)

spectra = {}
for iso in langs_df["iso_code"]:
    eigvals_path = SPECTRA_DIR / f"{iso}_eigvals.npy"
    if not eigvals_path.exists():
        raise FileNotFoundError(f"Falta espectro: {eigvals_path}")
    spectra[iso] = np.load(eigvals_path)

dist_matrix, iso_codes = compute_wasserstein_distance_matrix(spectra)
np.save(OUTPUT_DIR / "distance_matrices" / "spectral_baseline.npy", dist_matrix)

labels = langs_df.set_index("iso_code").loc[iso_codes, "family_id"].values

metrics = evaluate_clustering(dist_matrix, labels)
ratio, mean_inter, mean_intra = inter_intra_ratio(dist_matrix, labels)

# Histogram features for baseline: 64 bins over [0, 2]
n_bins = 64
X_spectral = np.zeros((len(iso_codes), n_bins))
for i, iso in enumerate(iso_codes):
    hist, _ = np.histogram(
        spectra[iso],
        bins=n_bins,
        range=(0, 2),
        density=True,
    )
    X_spectral[i] = hist

clf = evaluate_classifier(X_spectral, labels)

print(f"Silhouette: {metrics['silhouette']:+.3f}")
print(f"Inter/Intra ratio: {ratio:.2f}")
print(f"Accuracy: {clf['accuracy_mean']*100:.1f}% ± {clf['accuracy_std']*100:.1f}%")
print(f"Macro-F1: {clf['macro_f1_mean']:.3f} ± {clf['macro_f1_std']:.3f}")
print(f"Chance level: {100/len(family_to_id):.1f}%")

all_results = [{
    "variant": "Spectral_baseline",
    "display_name": "Spectral direct",
    "silhouette": metrics["silhouette"],
    "ratio": ratio,
    "mean_inter": mean_inter,
    "mean_intra": mean_intra,
    "accuracy_mean": clf["accuracy_mean"],
    "accuracy_std": clf["accuracy_std"],
    "macro_f1_mean": clf["macro_f1_mean"],
    "macro_f1_std": clf["macro_f1_std"],
}]

# ============================================================================
# SONIFICATION VARIANTS
# ============================================================================

variant_dirs = [d for d in AUDIO_BASE.iterdir() if d.is_dir()]
variant_names = sorted([d.name for d in variant_dirs], key=canonical_variant_sort_key)

print("\n" + "─" * 70)
print(f"SONIFICATION VARIANTS ({len(variant_names)})")
print("─" * 70)

for variant_name in variant_names:
    print(f"\nProcessing {variant_name}...")

    variant_dir = AUDIO_BASE / variant_name
    mfcc_features = {}

    missing_audio = []
    for iso in iso_codes:
        audio_path = variant_dir / f"{iso}.wav"
        if not audio_path.exists():
            missing_audio.append(iso)
            continue
        mfcc_features[iso] = extract_mfcc_features(audio_path)

    if missing_audio:
        raise FileNotFoundError(
            f"Faltan audios en {variant_name}: {missing_audio[:10]}"
            + (" ..." if len(missing_audio) > 10 else "")
        )

    # Save MFCC
    with open(OUTPUT_DIR / "mfcc_features" / f"{variant_name}.pkl", "wb") as f:
        pickle.dump(mfcc_features, f)

    # Feature matrix in same order as iso_codes
    X_mfcc = np.array([mfcc_features[iso] for iso in iso_codes])

    # Euclidean distance in MFCC space
    dist_matrix_mfcc = squareform(pdist(X_mfcc, metric="euclidean"))
    np.save(OUTPUT_DIR / "distance_matrices" / f"{variant_name}.npy", dist_matrix_mfcc)

    metrics = evaluate_clustering(dist_matrix_mfcc, labels)
    ratio, mean_inter, mean_intra = inter_intra_ratio(dist_matrix_mfcc, labels)
    clf = evaluate_classifier(X_mfcc, labels)

    print(f"  Silhouette: {metrics['silhouette']:+.3f}")
    print(f"  Ratio: {ratio:.2f}")
    print(f"  Accuracy: {clf['accuracy_mean']*100:.1f}% ± {clf['accuracy_std']*100:.1f}%")
    print(f"  Macro-F1: {clf['macro_f1_mean']:.3f} ± {clf['macro_f1_std']:.3f}")

    all_results.append({
        "variant": variant_name,
        "display_name": VARIANT_LABELS.get(variant_name, variant_name),
        "silhouette": metrics["silhouette"],
        "ratio": ratio,
        "mean_inter": mean_inter,
        "mean_intra": mean_intra,
        "accuracy_mean": clf["accuracy_mean"],
        "accuracy_std": clf["accuracy_std"],
        "macro_f1_mean": clf["macro_f1_mean"],
        "macro_f1_std": clf["macro_f1_std"],
    })

# ============================================================================
# SAVE
# ============================================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results_df = pd.DataFrame(all_results)

# CSV técnico
results_df.to_csv(OUTPUT_DIR / "classification_results_with_f1.csv", index=False)

# CSV simple
results_df[
    [
        "variant",
        "display_name",
        "silhouette",
        "ratio",
        "accuracy_mean",
        "accuracy_std",
    ]
].to_csv(OUTPUT_DIR / "classification_results.csv", index=False)

with open(OUTPUT_DIR / "classification_results.pkl", "wb") as f:
    pickle.dump(all_results, f)

# Tabla ordenada como en el paper: baseline primero, luego acústicos por silhouette
baseline_df = results_df[results_df["variant"] == "Spectral_baseline"].copy()
acoustic_df = results_df[results_df["variant"] != "Spectral_baseline"].copy()
acoustic_df = acoustic_df.sort_values("silhouette", ascending=False)

paper_table_df = pd.concat([baseline_df, acoustic_df], ignore_index=True)
paper_table_df.to_csv(OUTPUT_DIR / "table4_paper_order.csv", index=False)

print("\nResumen (orden paper):")
print(
    paper_table_df[
        [
            "variant",
            "display_name",
            "silhouette",
            "ratio",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)