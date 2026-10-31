@echo off
echo ==============================================
echo Pushing Smart College Predictor Project to GitHub...
echo ==============================================

echo [1/3] Adding files to git...
git add .

echo [2/3] Committing changes...
git commit -m "Remove 4 KPI metric cards (Colleges Database, Your Predictions, Latest Search, Model Confidence) from dashboard"

echo [3/3] Pushing to GitHub (origin main)...
git push origin main

echo ==============================================
echo Done! Project pushed successfully to GitHub.
echo ==============================================
pause
