@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0TRAIN_FROSS.ps1" -Stage 3000
if errorlevel 1 (
  echo.
  echo Continuation failed. Check .fross-brain15-training\logs\stage-3000.log
) else (
  echo.
  echo Training to 3000 total steps completed. Upload .fross-brain15-training\fross-transformer-upload.pt in FROSS Settings.
)
pause
