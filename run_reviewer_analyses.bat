@echo off
REM ============================================================================
REM run_reviewer_analyses.bat
REM ==========================
REM Executes ONLY the reviewer-requested analyses (Step 7/7)
REM
REM Prerequisites:
REM   - Scripts 1-6 must have been executed previously
REM   - Requires existing outputs in outputs/ directory
REM
REM Usage:
REM   run_reviewer_analyses.bat
REM
REM Expected runtime: ~10 minutes
REM ============================================================================

echo ========================================================================
echo REVIEWER REQUESTED ANALYSES - EPL-26-100127
echo ========================================================================
echo.
echo This script computes:
echo   1. Graph size statistics by family (Table for Methods)
echo   2. Macro-F1 scores for all variants (Updated Table 4)
echo.
echo Prerequisites: Scripts 1-6 must have been run previously
echo.
pause

echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    pause
    exit /b 1
)

echo Checking required files...
if not exist "07_reviewer_requested_analyses.py" (
    echo ERROR: Missing 07_reviewer_requested_analyses.py
    pause
    exit /b 1
)

if not exist "outputs\pickles\graph_stats.csv" (
    echo ERROR: Missing graph_stats.csv - Run scripts 1-2 first
    pause
    exit /b 1
)

if not exist "outputs\analysis\mfcc_features" (
    echo ERROR: Missing MFCC features - Run script 5 first
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo EXECUTING REVIEWER ANALYSES
echo ========================================================================
echo.

python 07_reviewer_requested_analyses.py

if errorlevel 1 (
    echo.
    echo ERROR: Analysis failed
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo COMPLETED SUCCESSFULLY
echo ========================================================================
echo.
echo Generated files:
echo   - outputs\analysis\graph_size_by_family.csv
echo   - outputs\analysis\graph_size_by_family_paper.csv
echo   - outputs\analysis\classification_results_with_f1.csv
echo.
echo Next steps:
echo   1. Check console output above for LaTeX table code
echo   2. Copy graph size table to Methods section
echo   3. Replace Table 4 in Results with updated version (includes F1)
echo.
echo The LaTeX code was printed above - scroll up to see it!
echo.
pause
