@echo off
REM ============================================================================
REM run_full_pipeline.bat
REM =====================
REM Executes the complete sonification pipeline from raw data to final figures.
REM Windows version
REM
REM Usage:
REM   run_full_pipeline.bat
REM
REM Expected runtime: ~2-3 hours on modern CPU
REM ============================================================================

echo ========================================================================
echo EPL SONIFICATION PIPELINE - FULL EXECUTION (Windows)
echo ========================================================================
echo.

REM Check Python
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo Checking required files...
if not exist "01_prepare_data.py" (
    echo ERROR: Missing script files
    pause
    exit /b 1
)

if not exist "udhr" (
    echo WARNING: udhr/ directory not found - create it and add UDHR texts
)

echo.
echo ========================================================================
echo STEP 1/7: Prepare Data
echo ========================================================================
python 01_prepare_data.py
if errorlevel 1 (
    echo ERROR in Step 1
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo STEP 2/7: Build Graphs
echo ========================================================================
python 02_build_graphs.py
if errorlevel 1 (
    echo ERROR in Step 2
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo STEP 3/7: Compute Spectra
echo ========================================================================
python 03_compute_spectra.py
if errorlevel 1 (
    echo ERROR in Step 3
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo STEP 4/7: Synthesize Audio
echo ========================================================================
echo NOTE: This step takes the longest (~30-60 minutes)
python 04_synthesize_sounds.py
if errorlevel 1 (
    echo ERROR in Step 4
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo STEP 5/7: Analyze Features
echo ========================================================================
python 05_analyze_spectra.py
if errorlevel 1 (
    echo ERROR in Step 5
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo STEP 6/7: Generate Visualizations
echo ========================================================================
python 06_visualize_results.py
if errorlevel 1 (
    echo ERROR in Step 6
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo STEP 7/7: Reviewer Requested Analyses
echo ========================================================================
echo NOTE: Computes graph size table + macro-F1 scores
python 07_reviewer_requested_analyses.py
if errorlevel 1 (
    echo ERROR in Step 7
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo PIPELINE COMPLETED SUCCESSFULLY
echo ========================================================================
echo.
echo Outputs generated:
echo   - Audio files:  outputs\audio\
echo   - Figures:      outputs\figures\
echo   - Data:         outputs\analysis\
echo   - Spectra:      outputs\spectra\
echo.
echo Next steps:
echo   1. Check outputs\figures\ for publication-ready plots
echo   2. Listen to audio in outputs\audio\
echo   3. Review metrics in outputs\analysis\classification_results_with_f1.csv
echo.
echo Paper: Sonification of language networks (EPL-26-100127)
echo ========================================================================
echo.
pause
