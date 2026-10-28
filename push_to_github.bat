@echo off
echo ==============================================
echo Pushing EduPredict Project to GitHub...
echo ==============================================

echo [1/3] Adding files to git...
git add .

echo [2/3] Committing changes...
git commit -m "Update live 2026 cutoffs and fees across KCET & EAMCET datasets with retrained ML model"

echo [3/3] Pushing to GitHub (origin main)...
git push origin main

echo ==============================================
echo Done! Project pushed successfully to GitHub.
echo ==============================================
pause
