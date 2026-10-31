@echo off
echo ==============================================
echo Pushing Smart College Predictor Project to GitHub...
echo ==============================================

echo [1/3] Adding files to git...
git add .

echo [2/3] Committing changes...
git commit -m "Add real email and password authentication with Sign In form, 'or' divider, and Google OAuth on login page"

echo [3/3] Pushing to GitHub (origin main)...
git push origin main

echo ==============================================
echo Done! Project pushed successfully to GitHub.
echo ==============================================
pause
