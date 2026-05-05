#!/usr/bin/env python3
"""
01_prepare_data.py - SIMPLIFIED VERSION
========================================
Generates americas_families.csv with the exact format needed by script 02.

Output columns: iso_code, family, name, latitude, longitude
"""

import pandas as pd

print("="*70)
print("STEP 1: PREPARAR DATOS DE FAMILIAS AMERICANAS")
print("="*70)

# Load Glottolog data
print("\nCargando datos de Glottolog...")
languoid = pd.read_csv("languoid.csv")
geo = pd.read_csv("languages_and_dialects_geo.csv")

print(f"  Languoid: {len(languoid)} entradas")
print(f"  Geo: {len(geo)} entradas")

# Merge
merged = languoid.merge(geo, left_on='id', right_on='glottocode', how='left')
print(f"  Merged: {len(merged)} entradas")

# Filter Americas
americas = merged[merged['macroarea'].isin(['North America', 'South America'])].copy()
print(f"  Américas: {len(americas)} lenguas")

# Create output with required columns
output = pd.DataFrame({
    'iso_code': americas['iso639P3code'],
    'family': americas['family_id'],
    'name': americas['name_x'],
    'latitude': americas['latitude_y'],
    'longitude': americas['longitude_y']
})

# Remove rows without ISO code
output = output[output['iso_code'].notna()].copy()
output = output[output['family'].notna()].copy()

print(f"\nCon ISO code válido: {len(output)} lenguas")

# Save
OUTPUT_FILE = "americas_families.csv"
output.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Guardado: americas_families.csv")
print(f"   Columnas: {list(output.columns)}")
print(f"   Lenguas: {len(output)}")

# Stats
family_counts = output['family'].value_counts()
large_families = family_counts[family_counts >= 5]

print(f"\n   Familias grandes (≥5 lenguas):")
for fam, count in large_families.head(10).items():
    fam_name = str(fam)[:40]
    print(f"      {fam_name:40s}: {count:3d}")

print("\n" + "="*70)
print("✅ PASO 1 COMPLETADO")
print("   Siguiente: python 02_build_graphs.py")
print("="*70)
