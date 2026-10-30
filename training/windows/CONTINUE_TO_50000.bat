@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0TRAIN_FROSS.ps1" -Stage 50000
if errorlevel 1 (
  echo.
  echo Stage 50000 failed. Check .fross-brain16-training\logs\stage-50000.log
) else (
  echo.
  echo Stage 2 complete. Next, run CONTINUE_TO_75000.bat
)
pause
