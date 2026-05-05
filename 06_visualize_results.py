#!/usr/bin/env python3
"""
06_visualize_results.py
=======================
Generates publication-quality figures:
  - MDS projection with family colors
  - Distance heatmaps sorted by family
  - Comparison plots across variants
  - Summary 4-panel figure

Outputs: figures/*.png at 300 dpi
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import MDS, TSNE
from matplotlib.patches import Rectangle

# ============================================================================
# CONFIGURATION
# ============================================================================

AMERICAS_CSV = "americas_families.csv"
LANGS_PKL = "outputs/pickles/clean_languages.pkl"
ANALYSIS_DIR = "outputs/analysis"
OUTPUT_DIR = "outputs/figures"

DPI = 300
FIGSIZE_SINGLE = (8, 6)
FIGSIZE_DOUBLE = (12, 5)

# Color palette (colorblind-friendly)
#FAMILY_COLORS = {
#    'Quechuan': '#1f77b4',     # Blue
#    'Arawakan': '#ff7f0e',     # Orange
#    'Mayan': '#2ca02c',        # Green
#    'Otomanguean': '#d62728',  # Red
#    'Panoan': '#9467bd'        # Purple
#}

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"{'='*70}")
print("VISUALIZATION GENERATION")
print(f"{'='*70}\n")

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load metadata
langs_df = pd.read_csv(AMERICAS_CSV)
with open(LANGS_PKL, 'rb') as f:
    clean_langs = pickle.load(f)

langs_df = langs_df[langs_df['iso_code'].isin(clean_langs)].copy()
langs_df = langs_df.sort_values('iso_code').reset_index(drop=True)

# Family mapping
family_to_id = {f: i for i, f in enumerate(sorted(langs_df['family'].unique()))}
langs_df['family_id'] = langs_df['family'].map(family_to_id)

iso_codes = sorted(langs_df['iso_code'].tolist())
labels = langs_df.set_index('iso_code').loc[iso_codes, 'family'].values
labels_numeric = langs_df.set_index('iso_code').loc[iso_codes, 'family_id'].values

print(f"Languages: {len(iso_codes)}")
print(f"Families: {len(family_to_id)}\n")

# Top 5 families by number of languages (for colored scatter/hist panels)
top_families = langs_df['family'].value_counts().head(5).index.tolist()
palette = sns.color_palette("tab10", n_colors=len(top_families)).as_hex()
FAMILY_COLORS = dict(zip(top_families, palette))

# ============================================================================
# FIGURE 1: MDS PROJECTION
# ============================================================================

print("Generating Figure 1: MDS projection...")

# Load spectral baseline distance matrix
dist_matrix = np.load(Path(ANALYSIS_DIR) / "distance_matrices" / "spectral_baseline.npy")

# MDS projection
mds = MDS(
    n_components=2,
    dissimilarity='precomputed',
    random_state=42,
    n_init=4
)
coords = mds.fit_transform(dist_matrix)

# Plot
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

for family in FAMILY_COLORS.keys():
    mask = labels == family
    ax.scatter(
        coords[mask, 0],
        coords[mask, 1],
        c=FAMILY_COLORS[family],
        label=family,
        s=80,
        alpha=0.7,
        edgecolors='white',
        linewidths=0.5
    )

ax.set_xlabel('MDS 1', fontsize=12)
ax.set_ylabel('MDS 2', fontsize=12)
ax.set_title(f'MDS Projection of {len(iso_codes)} American Languages',
             fontsize=13, pad=10)
ax.legend(loc='best', frameon=True, fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(OUTPUT_DIR) / "mds_clustering.png", dpi=DPI, bbox_inches='tight')
plt.close()

print("  ✓ Saved: mds_clustering.png")

# ============================================================================
# FIGURE 2: DISTANCE HEATMAP
# ============================================================================

print("Generating Figure 2: Distance heatmap...")

# Sort by family
sort_idx = np.argsort(labels_numeric)
sorted_dist = dist_matrix[sort_idx][:, sort_idx]
sorted_labels = labels[sort_idx]

# Find family boundaries
family_boundaries = []
current_family = sorted_labels[0]
for i, fam in enumerate(sorted_labels):
    if fam != current_family:
        family_boundaries.append(i)
        current_family = fam

# Plot
fig, ax = plt.subplots(figsize=(10, 9))

im = ax.imshow(sorted_dist, cmap='viridis', aspect='auto')

# Add family boundary lines
for boundary in family_boundaries:
    ax.axhline(boundary - 0.5, color='red', linewidth=2)
    ax.axvline(boundary - 0.5, color='red', linewidth=2)

ax.set_title('Wasserstein Distance Matrix (sorted by family)', fontsize=13, pad=10)
ax.set_xlabel('Language index', fontsize=11)
ax.set_ylabel('Language index', fontsize=11)

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Wasserstein distance', fontsize=11)

plt.tight_layout()
plt.savefig(Path(OUTPUT_DIR) / "distance_heatmap.png", dpi=DPI, bbox_inches='tight')
plt.close()

print("  ✓ Saved: distance_heatmap.png")

# ============================================================================
# FIGURE 3: t-SNE PROJECTION
# ============================================================================

print("Generating Figure 3: t-SNE projection...")

# t-SNE
# t-SNE
tsne = TSNE(
    n_components=2,
    metric='precomputed',
    init='random',
    random_state=42,
    perplexity=15
)
coords_tsne = tsne.fit_transform(dist_matrix)

# Plot
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

for family in FAMILY_COLORS.keys():
    mask = labels == family
    ax.scatter(
        coords_tsne[mask, 0],
        coords_tsne[mask, 1],
        c=FAMILY_COLORS[family],
        label=family,
        s=80,
        alpha=0.7,
        edgecolors='white',
        linewidths=0.5
    )

ax.set_xlabel('t-SNE 1', fontsize=12)
ax.set_ylabel('t-SNE 2', fontsize=12)
ax.set_title(f't-SNE Projection of {len(iso_codes)} American Languages', fontsize=13, pad=10)
ax.legend(loc='best', frameon=True, fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(OUTPUT_DIR) / "tsne_clustering.png", dpi=DPI, bbox_inches='tight')
plt.close()

print("  ✓ Saved: tsne_clustering.png")

# ============================================================================
# FIGURE 4: SUMMARY 4-PANEL
# ============================================================================

print("Generating Figure 4: Summary 4-panel...")

# Load results
results_df = pd.read_csv(Path(ANALYSIS_DIR) / "classification_results.csv")

# Create figure with 4 subplots
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Panel A: Silhouette scores
ax1 = fig.add_subplot(gs[0, 0])
variants = results_df['variant'].tolist()
silhouettes = results_df['silhouette'].tolist()

colors = ['#1f77b4' if v == 'Spectral_baseline' else '#ff7f0e' for v in variants]
bars = ax1.barh(range(len(variants)), silhouettes, color=colors, alpha=0.7)

ax1.set_yticks(range(len(variants)))
ax1.set_yticklabels(variants, fontsize=9)
ax1.set_xlabel('Silhouette Score', fontsize=11)
ax1.set_title('A) Clustering Quality', fontsize=12, fontweight='bold')
ax1.axvline(0, color='black', linewidth=0.8)
ax1.grid(axis='x', alpha=0.3)

# Panel B: Classification accuracy
ax2 = fig.add_subplot(gs[0, 1])
accuracies = results_df['accuracy_mean'].tolist()
errors = results_df['accuracy_std'].tolist()

bars = ax2.barh(range(len(variants)), 
                np.array(accuracies) * 100,
                xerr=np.array(errors) * 100,
                color=colors, alpha=0.7, capsize=3)

ax2.set_yticks(range(len(variants)))
ax2.set_yticklabels(variants, fontsize=9)
ax2.set_xlabel('Accuracy (%)', fontsize=11)
ax2.set_title('B) Classification Performance', fontsize=12, fontweight='bold')
chance = 100 / len(family_to_id)
ax2.axvline(chance, color='red', linestyle='--', linewidth=1,
            label=f'Chance ({chance:.1f}%)')

ax2.grid(axis='x', alpha=0.3)
ax2.legend(fontsize=9)

# Panel C: MDS (reuse from above)
ax3 = fig.add_subplot(gs[1, 0])
for family in FAMILY_COLORS.keys():
    mask = labels == family
    ax3.scatter(
        coords[mask, 0],
        coords[mask, 1],
        c=FAMILY_COLORS[family],
        label=family,
        s=60,
        alpha=0.7,
        edgecolors='white',
        linewidths=0.5
    )

ax3.set_xlabel('MDS 1', fontsize=10)
ax3.set_ylabel('MDS 2', fontsize=10)
ax3.set_title('C) MDS Projection (Spectral)', fontsize=12, fontweight='bold')
ax3.legend(loc='best', frameon=True, fontsize=8)
ax3.grid(alpha=0.3)

# Panel D: Metrics table
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

# Create table data
table_data = []
for _, row in results_df.head(6).iterrows():  # Top 6 variants
    table_data.append([
        row['variant'][:20],  # Truncate name
        f"{row['silhouette']:+.3f}",
        f"{row['ratio']:.2f}",
        f"{row['accuracy_mean']*100:.1f}%"
    ])

table = ax4.table(
    cellText=table_data,
    colLabels=['Variant', 'Sil', 'Ratio', 'Acc'],
    cellLoc='left',
    loc='center',
    bbox=[0, 0.2, 1, 0.7]
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header
for i in range(4):
    table[(0, i)].set_facecolor('#1f77b4')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax4.set_title('D) Top Variants Summary', fontsize=12, fontweight='bold', pad=20)

plt.savefig(Path(OUTPUT_DIR) / "summary_4panel.png", dpi=DPI, bbox_inches='tight')
plt.close()

print("  ✓ Saved: summary_4panel.png")

# ============================================================================
# FIGURE 5: EIGENVALUE HISTOGRAMS (Example Languages)
# ============================================================================

print("Generating Figure 5: Eigenvalue histograms...")

# Select representative languages (one per family)
representatives = {}
for family in top_families:
    iso_list = langs_df.loc[langs_df['family'] == family, 'iso_code'].tolist()
    if iso_list:
        representatives[family] = iso_list[0]
        
# Load spectra
from pathlib import Path
SPECTRA_DIR = "outputs/spectra"

fig, axes = plt.subplots(1, 5, figsize=(16, 3), sharey=True)

for i, (family, iso) in enumerate(representatives.items()):
    eigvals = np.load(Path(SPECTRA_DIR) / f"{iso}_eigvals.npy")
    
    axes[i].hist(eigvals, bins=50, color=FAMILY_COLORS[family], alpha=0.7, edgecolor='black')
    axes[i].set_xlabel('Eigenvalue', fontsize=10)
    axes[i].set_title(f"{family}\n({iso})", fontsize=11)
    axes[i].grid(alpha=0.3)

axes[0].set_ylabel('Frequency', fontsize=10)
fig.suptitle('Laplacian Eigenvalue Distributions (Representative Languages)', 
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(Path(OUTPUT_DIR) / "eigenvalue_histograms.png", dpi=DPI, bbox_inches='tight')
plt.close()

print("  ✓ Saved: eigenvalue_histograms.png")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print("VISUALIZATION COMPLETED")
print(f"{'='*70}")
print(f"\nGenerated {5} figures in {OUTPUT_DIR}/:")
print("  1. mds_clustering.png")
print("  2. distance_heatmap.png")
print("  3. tsne_clustering.png")
print("  4. summary_4panel.png")
print("  5. eigenvalue_histograms.png")
print(f"\nAll figures at {DPI} DPI\n")
