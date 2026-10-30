@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0TRAIN_FROSS.ps1" -Stage 75000
if errorlevel 1 (
  echo.
  echo Stage 75000 failed. Check .fross-brain16-training\logs\stage-75000.log
) else (
  echo.
  echo Stage 3 complete. Next, run CONTINUE_TO_100000.bat
)
pause
