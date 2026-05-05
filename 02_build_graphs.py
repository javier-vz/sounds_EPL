#!/usr/bin/env python3
"""
STEP 2: Build Graph-of-Words from UDHR texts
Paper subset only: 44 lenguas de 5 familias
"""

import os
import pickle
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter

print("=" * 70)
print("STEP 2: CONSTRUCCIÓN DE GRAFOS (PAPER SUBSET)")
print("=" * 70)

# Paths
FAMILY_CSV = "americas_families.csv"
UDHR_DIR = "udhr"
OUT_DIR = "outputs/pickles"

os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SIZE = 2  # paper baseline

# Familias del paper
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


def clean(iso, sentences):
    punct = r'``!"#$%&\¿()*+,-./:;<=>?@[\]_{|}\'\''
    table = str.maketrans({k: None for k in punct})

    cleaned = []
    for s in sentences:
        s = s.translate(table)
        tokens = [
            w.lower().translate(table)
            for w in s.split()
            if w and not w.isdigit()
        ]
        tokens = [w for w in tokens if len(w) > 1]
        if tokens:
            cleaned.append(tokens)

    # Skip header lines
    skip = {'zro': 6, 'tca': 7, 'gyr': 9}.get(iso, 5)
    return cleaned[skip:] if len(cleaned) > skip else cleaned


def gow(sentences, window_size=2):
    tokens = [t for s in sentences for t in s]
    cooc = defaultdict(int)

    for i, w1 in enumerate(tokens):
        for j in range(i + 1, min(i + window_size + 1, len(tokens))):
            w2 = tokens[j]
            if w1 != w2:
                edge = (w1, w2) if w1 <= w2 else (w2, w1)
                cooc[edge] += 1

    G = nx.Graph()
    for (w1, w2), weight in cooc.items():
        G.add_edge(w1, w2, weight=weight)

    return G


# 1. Cargar metadata
print("\n[1/5] Cargando metadata...")
df_fam = pd.read_csv(FAMILY_CSV)
df_fam = df_fam[df_fam["family"].isin(PAPER_FAMILIES)].copy()

lang_to_family = dict(zip(df_fam["iso_code"], df_fam["family"]))

print(f"   Lenguas candidatas en familias del paper: {len(lang_to_family)}")
for fam_code, fam_name in PAPER_FAMILIES.items():
    n = (df_fam["family"] == fam_code).sum()
    print(f"   {fam_name:12s} ({fam_code}): {n:2d}")


# 2. Leer UDHR
print("\n[2/5] Leyendo archivos UDHR...")
languages_raw = {}
missing = []

for iso in sorted(df_fam["iso_code"].dropna().unique()):
    path = os.path.join(UDHR_DIR, f"udhr_{iso}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        languages_raw[iso] = lines
    else:
        missing.append(iso)

print(f"   Cargados: {len(languages_raw)}")
print(f"   Faltantes: {len(missing)}")
if missing:
    print("   ISOs faltantes:", ", ".join(sorted(missing)))


# 3. Limpiar y refiltrar al subset real del paper
print("\n[3/5] Limpiando y filtrando subset del paper...")
cleaned_sentences = {}
for iso, sents in languages_raw.items():
    cleaned = clean(iso, sents)
    if cleaned:
        cleaned_sentences[iso] = cleaned

paper_df = df_fam[df_fam["iso_code"].isin(cleaned_sentences.keys())].copy()

# Verificación explícita de conteos del paper
observed_counts = paper_df["family"].value_counts().to_dict()

print("   Conteos observados tras cleaning:")
for fam_code, fam_name in PAPER_FAMILIES.items():
    obs = observed_counts.get(fam_code, 0)
    exp = EXPECTED_COUNTS[fam_code]
    print(f"   {fam_name:12s}: {obs:2d} (esperado {exp})")

total_obs = len(paper_df)
print(f"   Total observado: {total_obs} (esperado {EXPECTED_TOTAL})")

if observed_counts != EXPECTED_COUNTS or total_obs != EXPECTED_TOTAL:
    raise ValueError(
        "El subset después del cleaning no coincide con el paper. "
        "Revisa faltantes UDHR o reglas de cleaning antes de continuar."
    )

paper_isos = sorted(paper_df["iso_code"].tolist())
paper_languages = {iso: cleaned_sentences[iso] for iso in paper_isos}

total_tokens = sum(sum(len(s) for s in paper_languages[iso]) for iso in paper_isos)
print(f"   Lenguas finales del paper: {len(paper_languages)}")
print(f"   Total tokens: {total_tokens:,}")


# 4. Construir grafos solo para el subset del paper
print(f"\n[4/5] Construyendo grafos (window={WINDOW_SIZE})...")
graphs = {}
stats = []

for iso in paper_isos:
    G = gow(paper_languages[iso], WINDOW_SIZE)
    graphs[iso] = G

    fam_code = lang_to_family[iso]
    stats.append({
        "iso_code": iso,
        "family": fam_code,
        "family_name": PAPER_FAMILIES[fam_code],
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
    })

print(f"   Grafos construidos: {len(graphs)}")


# 5. Guardar
print("\n[5/5] Guardando archivos...")
with open(os.path.join(OUT_DIR, "graphs.pkl"), "wb") as f:
    pickle.dump(graphs, f)

with open(os.path.join(OUT_DIR, "languages.pkl"), "wb") as f:
    pickle.dump(paper_languages, f)

# Mantengo este nombre para compatibilidad con scripts viejos
with open(os.path.join(OUT_DIR, "clean_languages.pkl"), "wb") as f:
    pickle.dump(paper_isos, f)

# Y agrego uno explícito para el paper
with open(os.path.join(OUT_DIR, "paper_languages_44.pkl"), "wb") as f:
    pickle.dump(paper_isos, f)

pd.DataFrame(stats).to_csv(os.path.join(OUT_DIR, "graph_stats.csv"), index=False)

# Resumen por familia
graph_stats_df = pd.DataFrame(stats)
summary = (
    graph_stats_df
    .groupby(["family", "family_name"])
    .agg(
        n_languages=("iso_code", "count"),
        nodes_mean=("nodes", "mean"),
        nodes_std=("nodes", "std"),
        edges_mean=("edges", "mean"),
        edges_std=("edges", "std"),
    )
    .reset_index()
)
summary.to_csv(os.path.join(OUT_DIR, "graph_size_by_family_paper.csv"), index=False)

print("\n   Estadísticas globales:")
print(f"      Nodos (mean): {graph_stats_df['nodes'].mean():.0f}")
print(f"      Edges (mean): {graph_stats_df['edges'].mean():.0f}")
print(f"      Density (mean): {graph_stats_df['density'].mean():.4f}")

print("\n" + "=" * 70)
print("✅ PASO 2 COMPLETADO")
print(f"   Subset final del paper: {len(graphs)} lenguas / 5 familias")
print(f"   Archivos en: {OUT_DIR}/")
print("   Siguiente: python 03_compute_spectra.py")
print("=" * 70)