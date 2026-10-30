@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0TRAIN_FROSS.ps1" -Stage check
if errorlevel 1 (
  echo.
  echo Check failed. Check .fross-brain16-training\logs\checkpoint-check.log
) else (
  echo.
  echo Brain 1.6 checkpoint check completed.
)
pause
