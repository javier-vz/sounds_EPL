# Instrucciones para Windows

## Ejecución en Windows

Este repositorio incluye scripts `.bat` para ejecutar en Windows (además de los `.sh` para Linux/Mac).

---

## Inicio Rápido

### 1. Instalar Python

Asegúrate de tener Python 3.8+ instalado:
```cmd
python --version
```

Si no está instalado, descárgalo de: https://www.python.org/downloads/

**IMPORTANTE:** Durante la instalación, marca la opción "Add Python to PATH"

---

### 2. Instalar dependencias

Abre **CMD** o **PowerShell** en la carpeta del proyecto:

```cmd
pip install -r requirements.txt
```

Si tienes problemas de permisos, usa:
```cmd
pip install --user -r requirements.txt
```

---

### 3. Preparar datos

Crea la carpeta `udhr\` y coloca los archivos de texto UDHR:
```cmd
mkdir udhr
```

Coloca también `languoid.csv` y `languages_and_dialects_geo.csv` en la carpeta raíz.

---

## Ejecutar el pipeline

### Opción A: Pipeline completo (2-3 horas)

Doble clic en:
```
run_full_pipeline.bat
```

O desde CMD:
```cmd
run_full_pipeline.bat
```

Esto ejecuta todos los pasos 1-7 automáticamente.

---

### Opción B: Solo análisis del revisor (10 minutos)

Si ya ejecutaste los pasos 1-6 anteriormente, solo ejecuta:

```
run_reviewer_analyses.bat
```

Esto genera:
- Tabla de tamaño de grafos por familia
- Scores macro-F1 para Table 4

---

### Opción C: Scripts individuales

Ejecutar paso por paso:

```cmd
python 01_prepare_data.py
python 02_build_graphs.py
python 03_compute_spectra.py
python 04_synthesize_sounds.py
python 05_analyze_spectra.py
python 06_visualize_results.py
python 07_reviewer_requested_analyses.py
```
