@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0TRAIN_FROSS.ps1" -Stage 100000
if errorlevel 1 (
  echo.
  echo Stage 100000 failed. Check .fross-brain16-training\logs\stage-100000.log
) else (
  echo.
  echo All 100000 training steps are complete. Upload the int8 model from .fross-brain16-training in FROSS Settings.
)
pause
