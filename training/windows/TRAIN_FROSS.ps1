param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("600", "3000", "check")]
    [string]$Stage
)

$ErrorActionPreference = "Stop"
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$pythonScript = Join-Path $repoRoot "fross_transformer.py"
$baseCorpus = Join-Path $repoRoot "data\fross_corpus.jsonl"
$teacherCorpus = Join-Path $repoRoot "data\fross_teacher_corpus.jsonl"
$prepareCorpus = Join-Path $repoRoot "training\distillation\prepare_corpus.py"
$corpus = Join-Path $repoRoot ".fross-brain15-training\training-corpus.jsonl"
$workDir = Join-Path $repoRoot ".fross-brain15-training"
$checkpoint = Join-Path $workDir "fross-transformer-1.5-training.pt"
$uploadCheckpoint = Join-Path $workDir "fross-transformer-upload.pt"
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
    throw "Missing training\distillation\prepare_corpus.py. Re-download the current repository ZIP."
}

New-Item -ItemType Directory -Force -Path $workDir, $logDir | Out-Null

if (-not (Test-Path $pythonExe)) {
    Write-Host "Creating an isolated Python environment for local training..."
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

Write-Host "Preparing the student training corpus from base and teacher examples..."
& $pythonExe $prepareCorpus --base $baseCorpus --teacher $teacherCorpus --output $corpus
if ($LASTEXITCODE -ne 0) { throw "Could not validate and merge the training corpora." }


switch ($Stage) {
    "600" {
        if (Test-Path $checkpoint) {
            throw "A checkpoint already exists at $checkpoint. To avoid overwriting work, run CONTINUE_TO_3000.bat or move the checkpoint manually."
        }
        $arguments = @(
            $pythonScript, "train",
            "--corpus", $corpus,
            "--output", $checkpoint,
            "--steps", "600",
            "--schedule-steps", "3000",
            "--batch-size", "2",
            "--block-size", "500",
            "--vocab-size", "25000",
            "--seed", "1337"
        )
        $logPath = Join-Path $logDir "stage-600.log"
        Write-Host ""
        Write-Host "Stage 1/2: training from scratch to 600 steps with a 3000-step schedule."
    }
    "3000" {
        if (-not (Test-Path $checkpoint)) {
            throw "Checkpoint not found. Run START_TRAIN_600.bat successfully first."
        }
        $arguments = @(
            $pythonScript, "train",
            "--corpus", $corpus,
            "--output", $checkpoint,
            "--steps", "3000",
            "--schedule-steps", "3000",
            "--batch-size", "2",
            "--block-size", "500",
            "--vocab-size", "25000",
            "--seed", "1337",
            "--resume"
        )
        $logPath = Join-Path $logDir "stage-3000.log"
        Write-Host ""
        Write-Host "Stage 2/2: resuming from 600 to 3000 total steps."
    }
    "check" {
        if (-not (Test-Path $checkpoint)) {
            throw "Checkpoint not found. Complete training first."
        }
        $arguments = @(
            $pythonScript, "generate",
            "--output", $checkpoint,
            "--text", "Hello FROSS! Explain what a neural network is in simple terms.",
            "--max-new-tokens", "64"
        )
        $logPath = Join-Path $logDir "checkpoint-check.log"
        Write-Host ""
        Write-Host "Checking that the checkpoint loads and the model can generate a reply."
    }
}

Write-Host "Log file: $logPath"
& $pythonExe @arguments 2>&1 | Tee-Object -FilePath $logPath
if ($LASTEXITCODE -ne 0) {
    throw "The command failed. Copy the log file at $logPath for diagnosis."
}

if ($Stage -eq "600") {
    Write-Host ""
    Write-Host "Stage 600 complete. Next, run CONTINUE_TO_3000.bat."
} elseif ($Stage -eq "3000") {
    Write-Host ""
    Write-Host "Training checkpoint saved: $checkpoint"
        Write-Host "Creating a compact inference-only model file for FROSS Settings..."
        & $pythonExe $pythonScript package --output $checkpoint --package-output $uploadCheckpoint
        if ($LASTEXITCODE -ne 0) { throw "Could not create the compact upload checkpoint." }
        Write-Host ""
        Write-Host "UPLOAD THIS FILE IN FROSS SETTINGS:"
        Write-Host $uploadCheckpoint
        Write-Host "The full training checkpoint remains available for future resume runs: $checkpoint"
} else {
    Write-Host ""
    Write-Host "Checkpoint check complete. The generated sample is shown above."
}
