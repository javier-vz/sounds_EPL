#!/usr/bin/env python3
"""
07_reviewer_requested_analyses.py
==================================
Additional analyses requested by the EPL reviewer,
restricted to the exact paper subset (44 languages, 5 families).

Outputs:
  - outputs/analysis/graph_size_by_family.csv
  - outputs/analysis/graph_size_by_family_paper.csv
  - outputs/analysis/classification_results_with_f1.csv
  - outputs/analysis/table4_with_f1_paper_order.csv
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

# ============================================================================
# CONFIGURATION
# ============================================================================

AMERICAS_CSV = "americas_families.csv"
LANGS_PKL = "outputs/pickles/paper_languages_44.pkl"
GRAPH_STATS_CSV = "outputs/pickles/graph_stats.csv"
SPECTRA_DIR = Path("outputs/spectra")
MFCC_DIR = Path("outputs/analysis/mfcc_features")
ANALYSIS_DIR = Path("outputs/analysis")
BASE_RESULTS_CSV = ANALYSIS_DIR / "classification_results.csv"

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

FAMILY_ORDER = ["quec1387", "araw1281", "maya1287", "otom1299", "pano1259"]

VARIANT_LABELS = {
    "Spectral_baseline": "Spectral direct",
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


# ============================================================================
# LOAD PAPER SUBSET
# ============================================================================

print("=" * 70)
print("REVIEWER-REQUESTED ANALYSES (PAPER SUBSET ONLY)")
print("=" * 70)
print()

langs_df = pd.read_csv(AMERICAS_CSV)

with open(LANGS_PKL, "rb") as f:
    paper_langs = pickle.load(f)

langs_df = langs_df[langs_df["iso_code"].isin(paper_langs)].copy()
langs_df = langs_df[langs_df["family"].isin(PAPER_FAMILIES)].copy()
langs_df = langs_df.sort_values("iso_code").reset_index(drop=True)

observed_counts = langs_df["family"].value_counts().to_dict()
if observed_counts != EXPECTED_COUNTS or len(langs_df) != EXPECTED_TOTAL:
    raise ValueError(
        f"Subset incorrecto. Observado: {observed_counts}, total={len(langs_df)}"
    )

family_to_id = {fam: i for i, fam in enumerate(FAMILY_ORDER)}
langs_df["family_id"] = langs_df["family"].map(family_to_id)
langs_df["family_name"] = langs_df["family"].map(PAPER_FAMILIES)

iso_codes = sorted(langs_df["iso_code"].tolist())
labels = langs_df.set_index("iso_code").loc[iso_codes, "family_id"].values

print("Subset confirmado:")
for fam in FAMILY_ORDER:
    print(f"  {PAPER_FAMILIES[fam]:12s}: {EXPECTED_COUNTS[fam]}")
print(f"  Total: {len(iso_codes)}")
print()


# ============================================================================
# 1. GRAPH SIZE STATISTICS BY FAMILY
# ============================================================================

print("=" * 70)
print("REQUEST 1: GRAPH SIZE STATISTICS BY FAMILY")
print("=" * 70)
print()

graph_stats = pd.read_csv(GRAPH_STATS_CSV)
graph_stats = graph_stats[graph_stats["iso_code"].isin(iso_codes)].copy()
graph_stats = graph_stats[graph_stats["family"].isin(PAPER_FAMILIES)].copy()
graph_stats["family_name"] = graph_stats["family"].map(PAPER_FAMILIES)

if len(graph_stats) != EXPECTED_TOTAL:
    raise ValueError(
        f"graph_stats.csv no coincide con el subset del paper: {len(graph_stats)} filas"
    )

family_stats = (
    graph_stats
    .groupby(["family", "family_name"], sort=False)
    .agg(
        n_languages=("iso_code", "count"),
        nodes_mean=("nodes", "mean"),
        nodes_std=("nodes", "std"),
        edges_mean=("edges", "mean"),
        edges_std=("edges", "std"),
        nodes_min=("nodes", "min"),
        nodes_max=("nodes", "max"),
        edges_min=("edges", "min"),
        edges_max=("edges", "max"),
    )
    .reset_index()
)

family_stats["family_order"] = family_stats["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)})
family_stats = family_stats.sort_values("family_order").drop(columns="family_order")

paper_table = pd.DataFrame({
    "Family": family_stats["family_name"],
    "Languages": family_stats["n_languages"].astype(int),
    "Nodes (mean ± std)": [
        f"{row.nodes_mean:.0f} ± {row.nodes_std:.0f}"
        for row in family_stats.itertuples()
    ],
    "Edges (mean ± std)": [
        f"{row.edges_mean:.0f} ± {row.edges_std:.0f}"
        for row in family_stats.itertuples()
    ],
})

print("Paper-ready graph size table:")
print(paper_table.to_string(index=False))
print()

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

family_stats.to_csv(ANALYSIS_DIR / "graph_size_by_family.csv", index=False)
paper_table.to_csv(ANALYSIS_DIR / "graph_size_by_family_paper.csv", index=False)

print(f"✓ Saved: {ANALYSIS_DIR / 'graph_size_by_family.csv'}")
print(f"✓ Saved: {ANALYSIS_DIR / 'graph_size_by_family_paper.csv'}")
print()

print("LaTeX table for Methods:")
print(r"""
\begin{table}[h]
\centering
\caption{Average graph size per linguistic family. Values show mean $\pm$ standard deviation.}
\label{tab:graph_size}
\begin{tabular}{lccc}
\toprule
Family & Languages & Nodes & Edges \\
\midrule
""".strip())

for _, row in paper_table.iterrows():
    print(
        f"{row['Family']:12s} & {int(row['Languages']):2d} & "
        f"{row['Nodes (mean ± std)']:12s} & {row['Edges (mean ± std)']:12s} \\\\"
    )

print(r"""\bottomrule
\end{tabular}
\end{table}
""")


# ============================================================================
# 2. MACRO-F1 FOR BASELINE + ALL VARIANTS
# ============================================================================

print()
print("=" * 70)
print("REQUEST 2: MACRO-F1 SCORES")
print("=" * 70)
print()

all_results = []

# 2.1 Spectral baseline
print("Processing spectral baseline...")

spectra = {}
for iso in iso_codes:
    eigvals_path = SPECTRA_DIR / f"{iso}_eigvals.npy"
    if not eigvals_path.exists():
        raise FileNotFoundError(f"Missing spectrum file: {eigvals_path}")
    spectra[iso] = np.load(eigvals_path)

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

baseline_eval = evaluate_classifier(X_spectral, labels)

print(
    f"  Acc: {baseline_eval['accuracy_mean']*100:.1f}% ± {baseline_eval['accuracy_std']*100:.1f}% | "
    f"F1: {baseline_eval['macro_f1_mean']:.3f} ± {baseline_eval['macro_f1_std']:.3f}"
)

all_results.append({
    "variant": "Spectral_baseline",
    "display_name": VARIANT_LABELS["Spectral_baseline"],
    "accuracy_mean": baseline_eval["accuracy_mean"],
    "accuracy_std": baseline_eval["accuracy_std"],
    "macro_f1_mean": baseline_eval["macro_f1_mean"],
    "macro_f1_std": baseline_eval["macro_f1_std"],
})

# 2.2 Acoustic variants from saved MFCC features
print("\nProcessing sonification variants...")

mfcc_files = sorted(
    list(MFCC_DIR.glob("*.pkl")),
    key=lambda p: canonical_variant_sort_key(p.stem),
)

if not mfcc_files:
    raise FileNotFoundError(f"No MFCC feature files found in {MFCC_DIR}")

for mfcc_file in mfcc_files:
    variant_name = mfcc_file.stem

    with open(mfcc_file, "rb") as f:
        mfcc_features = pickle.load(f)

    missing = [iso for iso in iso_codes if iso not in mfcc_features]
    if missing:
        raise ValueError(
            f"Variant {variant_name} is missing MFCC features for: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )

    X_mfcc = np.array([mfcc_features[iso] for iso in iso_codes])
    variant_eval = evaluate_classifier(X_mfcc, labels)

    print(
        f"  {variant_name:>4s} | "
        f"Acc: {variant_eval['accuracy_mean']*100:5.1f}% ± {variant_eval['accuracy_std']*100:4.1f}% | "
        f"F1: {variant_eval['macro_f1_mean']:.3f} ± {variant_eval['macro_f1_std']:.3f}"
    )

    all_results.append({
        "variant": variant_name,
        "display_name": VARIANT_LABELS.get(variant_name, variant_name),
        "accuracy_mean": variant_eval["accuracy_mean"],
        "accuracy_std": variant_eval["accuracy_std"],
        "macro_f1_mean": variant_eval["macro_f1_mean"],
        "macro_f1_std": variant_eval["macro_f1_std"],
    })

results_df = pd.DataFrame(all_results)

# Merge with silhouette/ratio already computed in step 05
if not BASE_RESULTS_CSV.exists():
    raise FileNotFoundError(
        f"Run 05_analyze_spectra.py first. Missing file: {BASE_RESULTS_CSV}"
    )

base_results = pd.read_csv(BASE_RESULTS_CSV)

# Accept either the old or the corrected 05 output shape
merge_cols = ["variant", "silhouette", "ratio"]
optional_cols = [c for c in ["display_name"] if c in base_results.columns]

results_df = results_df.merge(
    base_results[merge_cols + optional_cols],
    on="variant",
    how="left",
    suffixes=("", "_from05"),
)

if "display_name_from05" in results_df.columns:
    results_df["display_name"] = results_df["display_name_from05"].fillna(results_df["display_name"])
    results_df = results_df.drop(columns=["display_name_from05"])

# Baseline first, then acoustic variants sorted by silhouette descending
baseline_df = results_df[results_df["variant"] == "Spectral_baseline"].copy()
acoustic_df = results_df[results_df["variant"] != "Spectral_baseline"].copy()
acoustic_df = acoustic_df.sort_values("silhouette", ascending=False)

paper_results = pd.concat([baseline_df, acoustic_df], ignore_index=True)

paper_results = paper_results[
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
]

paper_results.to_csv(ANALYSIS_DIR / "classification_results_with_f1.csv", index=False)
paper_results.to_csv(ANALYSIS_DIR / "table4_with_f1_paper_order.csv", index=False)

print()
print(f"✓ Saved: {ANALYSIS_DIR / 'classification_results_with_f1.csv'}")
print(f"✓ Saved: {ANALYSIS_DIR / 'table4_with_f1_paper_order.csv'}")
print()

print("Updated Table 4:")
print(paper_results.to_string(index=False))

print()
print("LaTeX Table 4:")
print(r"""
\begin{table*}[t]
\centering
\caption{Discrimination metrics for all sonification strategies and the spectral baseline. Sil: Silhouette score; $R$: inter/intra distance ratio; Acc: Random Forest accuracy (5-fold CV); F1: Macro-F1 score (5-fold CV). Best acoustic result per metric in bold.}
\label{tab:results}
\begin{tabular}{llcccc}
\toprule
 & Strategy & Sil & $R$ & Acc (\%) & F1 \\
\midrule
""".strip())

for _, row in paper_results.iterrows():
    label = "Baseline" if row["variant"] == "Spectral_baseline" else row["variant"]
    sil = f"+{row['silhouette']:.3f}" if row["silhouette"] >= 0 else f"{row['silhouette']:.3f}"
    acc = f"{row['accuracy_mean']*100:.1f} $\\pm$ {row['accuracy_std']*100:.1f}"
    f1 = f"{row['macro_f1_mean']:.3f} $\\pm$ {row['macro_f1_std']:.3f}"

    print(
        f"{label:8s} & {row['display_name']:20s} & "
        f"${sil}$ & {row['ratio']:.2f} & ${acc}$ & ${f1}$ \\\\"
    )
    if row["variant"] == "Spectral_baseline":
        print(r"\midrule")

print(r"""\bottomrule
\end{tabular}
\end{table*}
""")

print("=" * 70)
print("DONE")
print("=" * 70)