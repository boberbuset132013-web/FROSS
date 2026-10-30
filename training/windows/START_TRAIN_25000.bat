@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0TRAIN_FROSS.ps1" -Stage 25000
if errorlevel 1 (
  echo.
  echo Stage 25000 failed. Check .fross-brain16-training\logs\stage-25000.log
) else (
  echo.
  echo Stage 1 complete. Next, run CONTINUE_TO_50000.bat
)
pause
