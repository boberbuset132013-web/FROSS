@echo off
setlocal
title FROSS Resume from Saved Checkpoint to 5000
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0RESUME_FROSS_5000.ps1"
if errorlevel 1 (
  echo.
  echo Resume did not complete. The last saved checkpoint was preserved.
  echo Check .fross-local-training\unigram-20k-5000\logs\resume-5000.log
) else (
  echo.
  echo Resume script finished. Check the final metrics and sample.
)
echo.
pause
