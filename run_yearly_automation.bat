@echo off
title EduPredict - Yearly Dataset & Real-Time Counselling Automation
cls

echo =====================================================================
echo    EDUPREDICT - REAL-TIME COUNSELLING & COLLEGE WEBSITE AUTOMATION
echo =====================================================================
echo  Automates:
echo   1. Official Counselling Portals (KEA for KCET, APSCHE for EAPCET)
echo      - Real-time closing cutoffs and government tuition orders
echo      - Round 1, Round 2, and Extended Round allotment benchmarks
echo   2. College Institutional Websites
echo      - Hostel & mess fees, campus hero photos, NAAC accreditations
echo   3. Dataset Reconciliation across all 13 fields + ML Retraining
echo =====================================================================
echo.
echo Choose an automation task:
echo  [1] Full Yearly Automation (Update KCET + EAMCET + Retrain ML Models)
echo  [2] Check Official Counselling Portals (KEA & APSCHE for new cutoffs)
echo  [3] Crawl College Websites (Hostel Fees, Photos, NAAC Ratings)
echo  [4] Apply Real-Time Counselling Round Shifts (Round 1 / Round 2 / Extended)
echo  [5] Run Dataset Quality Validation (Dry Run - No files changed)
echo  [6] Retrain ML Prediction Models Only (train_model.py)
echo  [7] Run Live Automation Verification Test Suite
echo  [8] Exit
echo.

set /p choice="Enter your choice [1-8]: "

if "%choice%"=="1" (
    echo.
    echo Running Full Yearly Automation and Retraining...
    .\venv\Scripts\python.exe automation\yearly_dataset_updater.py --exam all --retrain
    goto end
)

if "%choice%"=="2" (
    echo.
    echo Checking Official Counselling Portals (KEA & APSCHE)...
    .\venv\Scripts\python.exe automation\realtime_counselling_crawler.py --check-portals
    goto end
)

if "%choice%"=="3" (
    echo.
    echo Crawling College Websites for Hostel Fees and Photos...
    .\venv\Scripts\python.exe automation\realtime_counselling_crawler.py --crawl-colleges
    goto end
)

if "%choice%"=="4" (
    echo.
    echo Choose Counselling Round:
    echo   [A] Mock Round Allotment
    echo   [B] Round 1 Allotment
    echo   [C] Round 2 Allotment (Relaxed Cutoffs)
    echo   [D] Extended / Mop-Up Round
    set /p rnd="Enter Round [A/B/C/D]: "
    if /i "%rnd%"=="A" .\venv\Scripts\python.exe automation\realtime_counselling_crawler.py --round mock
    if /i "%rnd%"=="B" .\venv\Scripts\python.exe automation\realtime_counselling_crawler.py --round round1
    if /i "%rnd%"=="C" .\venv\Scripts\python.exe automation\realtime_counselling_crawler.py --round round2
    if /i "%rnd%"=="D" .\venv\Scripts\python.exe automation\realtime_counselling_crawler.py --round extended
    goto end
)

if "%choice%"=="5" (
    echo.
    echo Running Dataset Quality Validation (Dry Run)...
    .\venv\Scripts\python.exe automation\yearly_dataset_updater.py --dry-run
    goto end
)

if "%choice%"=="6" (
    echo.
    echo Retraining ML Prediction Model...
    .\venv\Scripts\python.exe train_model.py
    goto end
)

if "%choice%"=="7" (
    echo.
    echo Running Real-Time Verification Test Suite...
    .\venv\Scripts\python.exe automation\test_automation.py
    goto end
)

if "%choice%"=="8" (
    exit
)

echo Invalid choice. Exiting.

:end
echo.
echo =====================================================================
echo Task completed.
echo =====================================================================
pause
