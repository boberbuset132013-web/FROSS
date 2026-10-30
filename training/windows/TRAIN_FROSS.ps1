param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("25000", "50000", "75000", "100000", "check")]
    [string]$Stage
)

$ErrorActionPreference = "Stop"
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$pythonScript = Join-Path $repoRoot "fross_transformer.py"
$baseCorpus = Join-Path $repoRoot "data\fross_corpus.jsonl"
$teacherCorpus = Join-Path $repoRoot "data\fross_teacher_corpus.jsonl"
$prepareCorpus = Join-Path $repoRoot "training\distillation\prepare_corpus.py"
$workDir = Join-Path $repoRoot ".fross-brain16-training"
$corpus = Join-Path $workDir "training-corpus.jsonl"
$checkpoint = Join-Path $workDir "fross-transformer-1.6-64m-training.pt"
$uploadCheckpoint = Join-Path $workDir "fross-transformer-1.6-64m-int8.pt"
$venvDir = Join-Path $repoRoot ".venv-training"
$pythonExe = Join-Path $venvDir "Scripts\python.exe"
$logDir = Join-Path $workDir "logs"

if (-not (Test-Path $pythonScript)) {
    throw "Missing fross_transformer.py. Extract the whole repository ZIP and run this from training\windows."
}
if (-not (Test-Path $baseCorpus)) {
    throw "Missing data\fross_corpus.jsonl."
}
if (-not (Test-Path $teacherCorpus)) {
    throw "Missing data\fross_teacher_corpus.jsonl. Re-download the current repository ZIP."
}
if (-not (Test-Path $prepareCorpus)) {
    throw "Missing training\distillation\prepare_corpus.py."
}

New-Item -ItemType Directory -Force -Path $workDir, $logDir | Out-Null

if (-not (Test-Path $pythonExe)) {
    Write-Host "Creating an isolated Python environment for local Brain 1.6 training..."
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        & $pyLauncher.Source -3 -m venv $venvDir
        if ($LASTEXITCODE -ne 0) { throw "Could not create the venv using py -3." }
    } else {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonCommand) { throw "Python was not found. Install Python 3.12+ from python.org and retry." }
        & $pythonCommand.Source -m venv $venvDir
        if ($LASTEXITCODE -ne 0) { throw "Could not create the venv using python." }
    }
}

Write-Host "Python executable: $pythonExe"
& $pythonExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip." }

& $pythonExe -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing CPU-only PyTorch..."
    & $pythonExe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw "Could not install CPU PyTorch. Check internet access and your Python version." }
}

& $pythonExe -c "import sentencepiece" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing SentencePiece Unigram tokenizer..."
    & $pythonExe -m pip install "sentencepiece>=0.2.0,<0.3"
    if ($LASTEXITCODE -ne 0) { throw "Could not install SentencePiece. Check internet access." }
}

$env:FROSS_DEVICE = "cpu"
$env:FROSS_TORCH_THREADS = "2"

if (-not (Test-Path $corpus)) {
    Write-Host "Preparing and validating the combined training corpus..."
    & $pythonExe $prepareCorpus --base $baseCorpus --teacher $teacherCorpus --output $corpus
    if ($LASTEXITCODE -ne 0) { throw "Could not validate and merge the training corpora." }
} else {
    Write-Host "Using the preserved training corpus so resume checks remain consistent."
}

switch ($Stage) {
    "25000" {
        if (Test-Path $checkpoint) {
            throw "A Brain 1.6 checkpoint already exists at $checkpoint. Continue with stage 50000 or move the checkpoint to a backup."
        }
        $arguments = @(
            $pythonScript, "train", "--corpus", $corpus, "--output", $checkpoint,
            "--steps", "25000", "--schedule-steps", "100000",
            "--batch-size", "1", "--block-size", "4096", "--vocab-size", "8000",
            "--seed", "1337"
        )
        $logPath = Join-Path $logDir "stage-25000.log"
        Write-Host "Stage 1/4: train Brain 1.6 from its new initialization to 25,000 cumulative steps."
    }
    "50000" {
        if (-not (Test-Path $checkpoint)) { throw "Checkpoint not found. Run stage 25000 first." }
        $arguments = @(
            $pythonScript, "train", "--corpus", $corpus, "--output", $checkpoint,
            "--steps", "50000", "--schedule-steps", "100000",
            "--batch-size", "1", "--block-size", "4096", "--vocab-size", "8000",
            "--seed", "1337", "--resume"
        )
        $logPath = Join-Path $logDir "stage-50000.log"
        Write-Host "Stage 2/4: resume Brain 1.6 to 50,000 cumulative steps."
    }
    "75000" {
        if (-not (Test-Path $checkpoint)) { throw "Checkpoint not found. Complete stage 50000 first." }
        $arguments = @(
            $pythonScript, "train", "--corpus", $corpus, "--output", $checkpoint,
            "--steps", "75000", "--schedule-steps", "100000",
            "--batch-size", "1", "--block-size", "4096", "--vocab-size", "8000",
            "--seed", "1337", "--resume"
        )
        $logPath = Join-Path $logDir "stage-75000.log"
        Write-Host "Stage 3/4: resume Brain 1.6 to 75,000 cumulative steps."
    }
    "100000" {
        if (-not (Test-Path $checkpoint)) { throw "Checkpoint not found. Complete stage 75000 first." }
        $arguments = @(
            $pythonScript, "train", "--corpus", $corpus, "--output", $checkpoint,
            "--steps", "100000", "--schedule-steps", "100000",
            "--batch-size", "1", "--block-size", "4096", "--vocab-size", "8000",
            "--seed", "1337", "--resume"
        )
        $logPath = Join-Path $logDir "stage-100000.log"
        Write-Host "Stage 4/4: complete Brain 1.6 training at 100,000 cumulative steps."
    }
    "check" {
        if (-not (Test-Path $checkpoint)) { throw "Checkpoint not found. Complete training first." }
        $arguments = @(
            $pythonScript, "generate", "--output", $checkpoint,
            "--text", "Объясни простыми словами, как работает нейронная сеть и приведи пример.",
            "--max-new-tokens", "96"
        )
        $logPath = Join-Path $logDir "checkpoint-check.log"
        Write-Host "Checking that the full training checkpoint loads and generates a sample."
    }
}

Write-Host "Log file: $logPath"
& $pythonExe @arguments 2>&1 | Tee-Object -FilePath $logPath
if ($LASTEXITCODE -ne 0) {
    throw "The command failed. Copy the log file at $logPath for diagnosis."
}

if ($Stage -eq "100000") {
    Write-Host "Training checkpoint saved: $checkpoint"
    Write-Host "Creating the compact int8 inference package for FROSS Settings..."
    & $pythonExe $pythonScript package --output $checkpoint --package-output $uploadCheckpoint
    if ($LASTEXITCODE -ne 0) { throw "Could not create the compact int8 inference package." }
    Write-Host "Upload this file in FROSS Settings: $uploadCheckpoint"
    Write-Host "The full training checkpoint remains available for future resume runs: $checkpoint"
} elseif ($Stage -eq "check") {
    Write-Host "Checkpoint check complete."
} else {
    Write-Host "Stage $Stage complete. Continue with the next cumulative training stage."
}
