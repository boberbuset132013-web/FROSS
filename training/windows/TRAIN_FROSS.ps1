param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("600", "3000", "check")]
    [string]$Stage
)

$ErrorActionPreference = "Stop"
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$pythonScript = Join-Path $repoRoot "fross_transformer.py"
$corpus = Join-Path $repoRoot "data\fross_corpus.jsonl"
$workDir = Join-Path $repoRoot ".fross-local-training"
$checkpoint = Join-Path $workDir "fross-transformer.pt"
$venvDir = Join-Path $repoRoot ".venv-training"
$pythonExe = Join-Path $venvDir "Scripts\python.exe"
$logDir = Join-Path $workDir "logs"

if (-not (Test-Path $pythonScript)) {
    throw "Не найден fross_transformer.py. Скачай и распакуй весь репозиторий, затем запускай BAT из папки training\windows."
}
if (-not (Test-Path $corpus)) {
    throw "Не найден корпус data\fross_corpus.jsonl."
}

New-Item -ItemType Directory -Force -Path $workDir, $logDir | Out-Null

if (-not (Test-Path $pythonExe)) {
    Write-Host "Создаю отдельное Python-окружение для локального обучения..."
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        & $pyLauncher.Source -3 -m venv $venvDir
        if ($LASTEXITCODE -ne 0) { throw "Не удалось создать venv через py -3." }
    } else {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonCommand) { throw "Python не найден. Установи Python 3.12+ с python.org и повтори." }
        & $pythonCommand.Source -m venv $venvDir
        if ($LASTEXITCODE -ne 0) { throw "Не удалось создать venv через python." }
    }
}

Write-Host "Python: $pythonExe"
& $pythonExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Не удалось обновить pip." }

& $pythonExe -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Устанавливаю CPU-версию PyTorch. Скачивание может занять некоторое время..."
    & $pythonExe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw "Не удалось установить PyTorch CPU. Проверь интернет и версию Python." }
}

$env:FROSS_DEVICE = "cpu"
$env:FROSS_TORCH_THREADS = "2"

switch ($Stage) {
    "600" {
        if (Test-Path $checkpoint) {
            throw "Чекпоинт уже существует: $checkpoint. Чтобы не потерять обучение, запусти CONTINUE_TO_3000.bat или перемести этот файл вручную."
        }
        $arguments = @(
            $pythonScript, "train",
            "--corpus", $corpus,
            "--output", $checkpoint,
            "--steps", "600",
            "--schedule-steps", "3000",
            "--batch-size", "4",
            "--block-size", "320",
            "--vocab-size", "5000",
            "--seed", "1337"
        )
        $logPath = Join-Path $logDir "stage-600.log"
        Write-Host ""
        Write-Host "Этап 1/2: обучение с нуля до 600 шагов; расписание рассчитано на 3000."
    }
    "3000" {
        if (-not (Test-Path $checkpoint)) {
            throw "Чекпоинт не найден. Сначала успешно запусти START_TRAIN_600.bat."
        }
        $arguments = @(
            $pythonScript, "train",
            "--corpus", $corpus,
            "--output", $checkpoint,
            "--steps", "3000",
            "--schedule-steps", "3000",
            "--batch-size", "4",
            "--block-size", "320",
            "--vocab-size", "5000",
            "--seed", "1337",
            "--resume"
        )
        $logPath = Join-Path $logDir "stage-3000.log"
        Write-Host ""
        Write-Host "Этап 2/2: продолжаю с 600 до суммарных 3000 шагов."
    }
    "check" {
        if (-not (Test-Path $checkpoint)) {
            throw "Чекпоинт не найден. Сначала заверши обучение."
        }
        $arguments = @(
            $pythonScript, "generate",
            "--output", $checkpoint,
            "--text", "Привет, FROSS! Объясни простыми словами, что такое нейросеть.",
            "--max-new-tokens", "64"
        )
        $logPath = Join-Path $logDir "checkpoint-check.log"
        Write-Host ""
        Write-Host "Проверяю, что чекпоинт открывается и модель генерирует ответ."
    }
}

Write-Host "Лог: $logPath"
& $pythonExe @arguments 2>&1 | Tee-Object -FilePath $logPath
if ($LASTEXITCODE -ne 0) {
    throw "Команда завершилась с ошибкой. Скопируй лог $logPath для диагностики."
}

if ($Stage -eq "600") {
    Write-Host ""
    Write-Host "Этап 600 завершён. Следующий шаг — CONTINUE_TO_3000.bat."
} elseif ($Stage -eq "3000") {
    Write-Host ""
    Write-Host "Готово: чекпоинт обучен до суммарных 3000 шагов: $checkpoint"
} else {
    Write-Host ""
    Write-Host "Проверка завершена. Выше показан тестовый ответ модели."
}
