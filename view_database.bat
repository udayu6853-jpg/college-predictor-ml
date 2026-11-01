@echo off
title Smart College Predictor - SQL Database Viewer
color 0b
echo ============================================================
echo Connecting to Cloud PostgreSQL Database on Render...
echo ============================================================
echo.
if exist ".\venv\Scripts\python.exe" (
    .\venv\Scripts\python.exe view_database.py
) else (
    python view_database.py
)
echo.
echo ============================================================
echo Press any key to close this window...
echo ============================================================
pause >nul

