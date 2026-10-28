param()

$ErrorActionPreference = "Stop"
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$pythonScript = Join-Path $repoRoot "fross_transformer.py"
$baseCorpus = Join-Path $repoRoot "data\fross_corpus.jsonl"
$teacherCorpus = Join-Path $repoRoot "data\fross_teacher_corpus.jsonl"
$prepareCorpus = Join-Path $repoRoot "training\distillation\prepare_corpus.py"
$workDir = Join-Path $repoRoot ".fross-local-training\unigram-20k-5000"
$corpus = Join-Path $workDir "training-corpus.jsonl"
$checkpoint = Join-Path $workDir "fross-transformer.pt"
$venvDir = Join-Path $repoRoot ".venv-unigram-training"
$pythonExe = Join-Path $venvDir "Scripts\python.exe"
$logDir = Join-Path $workDir "logs"

function Invoke-PythonLogged {
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$LogPath
    )
    $logParent = Split-Path -Parent $LogPath
    if ($logParent) { New-Item -ItemType Directory -Force -Path $logParent | Out-Null }
    $writer = [System.IO.StreamWriter]::new($LogPath, $true, [System.Text.UTF8Encoding]::new($false))
    $writer.WriteLine("===== $(Get-Date -Format s) | $PythonPath $($Arguments -join ' ') =====")
    $writer.Flush()
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $PythonPath -u @Arguments 2>&1 | ForEach-Object {
            $line = [string]$_
            Write-Host $line
            [Console]::Out.Flush()
            $writer.WriteLine($line)
            $writer.Flush()
        }
        $nativeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
        $writer.Dispose()
    }
    return $nativeExitCode
}

foreach ($required in @($pythonScript, $baseCorpus, $teacherCorpus, $prepareCorpus)) {
    if (-not (Test-Path $required)) { throw "Missing required file: $required. Download and extract the complete repository ZIP." }
}
New-Item -ItemType Directory -Force -Path $workDir, $logDir | Out-Null

if (-not (Test-Path $pythonExe)) {
    Write-Host "Creating an isolated Python environment..."
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        & $pyLauncher.Source -3 -m venv $venvDir
        if ($LASTEXITCODE -ne 0) { throw "Could not create Python virtual environment." }
    } else {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonCommand) { throw "Python 3 is missing. Install Python from python.org and retry." }
        & $pythonCommand.Source -m venv $venvDir
        if ($LASTEXITCODE -ne 0) { throw "Could not create Python virtual environment." }
    }
}

Write-Host "Python: $pythonExe"
& $pythonExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip." }

# Install NumPy explicitly: PyTorch may warn during startup if it is missing.
& $pythonExe -m pip install "numpy>=2.0"
if ($LASTEXITCODE -ne 0) { throw "NumPy installation failed." }

# Use find_spec instead of importing modules for checks. Importing a missing module
# prints a traceback to stderr, which Windows PowerShell 5.1 can treat as a terminating
# NativeCommandError when ErrorActionPreference is Stop.
& $pythonExe -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing CPU PyTorch..."
    & $pythonExe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed. Check Python compatibility and internet access." }
}
& $pythonExe -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('sentencepiece') else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing SentencePiece Unigram tokenizer..."
    & $pythonExe -m pip install "sentencepiece>=0.2.0,<1"
    if ($LASTEXITCODE -ne 0) { throw "SentencePiece installation failed." }
}

$env:FROSS_DEVICE = "cpu"
$env:FROSS_TORCH_THREADS = "2"
$env:PYTHONUNBUFFERED = "1"

Write-Host "Live progress is printed every 5 steps; checkpoints and snapshots are saved every 200 steps."
Write-Host "Validating and preparing the combined training corpus..."
& $pythonExe $prepareCorpus --base $baseCorpus --teacher $teacherCorpus --output $corpus
if ($LASTEXITCODE -ne 0) { throw "Corpus validation failed; training has not started." }

if (-not (Test-Path $checkpoint)) {
    $stage600 = @(
        $pythonScript, "train", "--corpus", $corpus, "--output", $checkpoint,
        "--steps", "600", "--schedule-steps", "5000", "--batch-size", "4",
        "--block-size", "320", "--vocab-size", "20000", "--seed", "1337",
        "--checkpoint-interval", "200", "--log-interval", "5"
    )
    $log600 = Join-Path $logDir "stage-600.log"
    Write-Host ""
    Write-Host "Stage 1/2: training a NEW Unigram model to 600 steps (target vocabulary 20000)."
    Write-Host "Log: $log600"
    $trainExitCode = Invoke-PythonLogged -PythonPath $pythonExe -Arguments $stage600 -LogPath $log600
    if ($trainExitCode -ne 0) { throw "Stage 600 failed. Keep the checkpoint and inspect $log600." }
} else {
    Write-Host "A checkpoint for this experiment already exists; it will not be overwritten."
}

$env:FROSS_CHECKPOINT_PATH = $checkpoint
$completedSteps = & $pythonExe -c "import os,torch; p=torch.load(os.environ['FROSS_CHECKPOINT_PATH'],map_location='cpu',weights_only=True); print(int(p.get('metrics',{}).get('training_steps',0)))"
Remove-Item Env:FROSS_CHECKPOINT_PATH
if ($LASTEXITCODE -ne 0) { throw "Could not inspect the saved checkpoint. It has been preserved." }
$completedSteps = [int]("$completedSteps".Trim())

# If an interrupted run left a checkpoint below step 600, resume that stage first.
if ($completedSteps -lt 600) {
    $resume600 = @(
        $pythonScript, "train", "--corpus", $corpus, "--output", $checkpoint,
        "--steps", "600", "--schedule-steps", "5000", "--batch-size", "4",
        "--block-size", "320", "--vocab-size", "20000", "--seed", "1337",
        "--checkpoint-interval", "200", "--log-interval", "5"
    )
    if ($completedSteps -gt 0) { $resume600 += "--resume" }
    $log600 = Join-Path $logDir "stage-600.log"
    Write-Host ""
    Write-Host "Checkpoint contains $completedSteps steps; resuming stage 1 to 600 before stage 2."
    Write-Host "Log: $log600"
    $trainExitCode = Invoke-PythonLogged -PythonPath $pythonExe -Arguments $resume600 -LogPath $log600
    if ($trainExitCode -ne 0) { throw "Stage 600 continuation failed. Check $log600; checkpoint preserved." }
    $completedSteps = 600
}

if ($completedSteps -lt 5000) {
    $continueArgs = @(
        $pythonScript, "train", "--corpus", $corpus, "--output", $checkpoint,
        "--steps", "5000", "--schedule-steps", "5000", "--batch-size", "4",
        "--block-size", "320", "--vocab-size", "20000", "--seed", "1337",
        "--checkpoint-interval", "200", "--log-interval", "5", "--resume"
    )
    $log5000 = Join-Path $logDir "stage-5000.log"
    Write-Host ""
    Write-Host "Stage 2/2: continuing from $completedSteps to 5000 TOTAL steps."
    Write-Host "Log: $log5000"
    $trainExitCode = Invoke-PythonLogged -PythonPath $pythonExe -Arguments $continueArgs -LogPath $log5000
    if ($trainExitCode -ne 0) { throw "Continuation failed. Keep the checkpoint and inspect $log5000." }
} elseif ($completedSteps -eq 5000) {
    Write-Host "This experiment already has 5000 training steps. No retraining was started."
} else {
    throw "Unexpected checkpoint step count: $completedSteps. Checkpoint preserved; no new training started."
}

Write-Host ""
Write-Host "Training stage finished. Running a short checkpoint load/generation check..."
$checkLog = Join-Path $logDir "checkpoint-check.log"
$checkArgs = @(
    $pythonScript, "generate", "--output", $checkpoint,
    "--text", "Explain in simple terms what a neural network is.",
    "--max-new-tokens", "80"
)
$checkExitCode = Invoke-PythonLogged -PythonPath $pythonExe -Arguments $checkArgs -LogPath $checkLog
if ($checkExitCode -ne 0) { throw "Checkpoint verification failed. The saved checkpoint has been preserved." }

Write-Host ""
Write-Host "Checkpoint: $checkpoint"
Write-Host "Saved model snapshots: $(Join-Path $workDir 'checkpoints')"
Write-Host "Logs: $logDir"
Write-Host "This is a local checkpoint only; it has NOT been uploaded to Hostless or the live website."
