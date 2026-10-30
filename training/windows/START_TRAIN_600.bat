@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0TRAIN_FROSS.ps1" -Stage 600
if errorlevel 1 (
  echo.
  echo Training stage 600 failed. Check .fross-brain15-training\logs\stage-600.log
) else (
  echo.
  echo Stage 600 completed. Next, run CONTINUE_TO_3000.bat
)
pause
