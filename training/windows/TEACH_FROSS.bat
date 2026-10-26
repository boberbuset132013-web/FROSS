@echo off
setlocal
set "TEACHER_MODEL=%~1"
if not defined TEACHER_MODEL set "TEACHER_MODEL=qwen2.5:3b"

where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher "py" was not found. Install Python 3.12+ or add py.exe to PATH.
  pause
  exit /b 1
)

echo Generating teacher examples locally with model: %TEACHER_MODEL%
echo This does not train the teacher model. It only adds question-answer examples for FROSS.
echo Make sure Ollama is running and this model is already installed.
py -3 "%~dp0..\distillation\teacher_distill.py" --model "%TEACHER_MODEL%" --count 60 --batch-size 6
if errorlevel 1 (
  echo.
  echo Teacher generation failed. Check the message above and verify the local Ollama model name.
) else (
  echo.
  echo Teacher data generation finished. Next, run START_TRAIN_600.bat.
)
pause
