@echo off
setlocal
title FROSS Unigram 20k - 5000 Steps
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0TRAIN_FROSS_UNIGRAM_5000.ps1"
if errorlevel 1 (
  echo.
  echo Training did not complete. The checkpoint was kept when possible.
  echo Check .fross-local-training\unigram-20k-5000\logs\
) else (
  echo.
  echo Training script finished. Check the metrics and generated sample before deployment.
)
echo.
pause
