param()

$ErrorActionPreference = "Stop"
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$pythonScript = Join-Path $repoRoot "fross_transformer.py"
$workDir = Join-Path $repoRoot ".fross-local-training\unigram-20k-5000"
$corpus = Join-Path $workDir "training-corpus.jsonl"
$checkpoint = Join-Path $workDir "fross-transformer.pt"
$pythonExe = Join-Path $repoRoot ".venv-unigram-training\Scripts\python.exe"
$logDir = Join-Path $workDir "logs"
$logPath = Join-Path $logDir "resume-5000.log"

foreach ($required in @($pythonScript, $corpus, $checkpoint, $pythonExe)) {
    if (-not (Test-Path $required)) {
        throw "Missing required file: $required. Do not delete the checkpoint or training-corpus.jsonl. Restore the full repository ZIP if source files are missing."
    }
}
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$env:FROSS_DEVICE = "cpu"
$env:FROSS_TORCH_THREADS = "2"
$env:PYTHONUNBUFFERED = "1"

$env:FROSS_CHECKPOINT_PATH = $checkpoint
$completedStepsText = & $pythonExe -u -c "import os,torch; p=torch.load(os.environ['FROSS_CHECKPOINT_PATH'],map_location='cpu',weights_only=True); print(int(p.get('metrics',{}).get('training_steps',0)))"
$inspectExitCode = $LASTEXITCODE
Remove-Item Env:FROSS_CHECKPOINT_PATH
if ($inspectExitCode -ne 0) { throw "Could not inspect checkpoint. It was preserved." }
$completedSteps = [int]("$completedStepsText".Trim())
if ($completedSteps -lt 1 -or $completedSteps -gt 5000) {
    throw "Unexpected saved step count $completedSteps. Checkpoint preserved; no training started."
}
if ($completedSteps -eq 5000) {
    Write-Host "Checkpoint already contains 5000 steps. Nothing to resume."
    exit 0
}

Write-Host ""
Write-Host "FROSS RESUME MODE"
Write-Host "Saved checkpoint: step $completedSteps/5000"
Write-Host "Training corpus will NOT be rebuilt or overwritten."
Write-Host "Logs: $logPath"
Write-Host "Live progress every 5 steps; resumable checkpoint every 200 steps."
Write-Host "If the corpus hash differs from the checkpoint, training will stop safely."

$writer = [System.IO.StreamWriter]::new($logPath, $true, [System.Text.UTF8Encoding]::new($false))
$writer.WriteLine("===== $(Get-Date -Format s) | resume from step $completedSteps to 5000 =====")
$writer.Flush()
$previousPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    & $pythonExe -u $pythonScript train --corpus $corpus --output $checkpoint --steps 5000 --schedule-steps 5000 --batch-size 4 --block-size 320 --vocab-size 20000 --seed 1337 --checkpoint-interval 200 --log-interval 5 --resume 2>&1 | ForEach-Object {
        $line = [string]$_
        Write-Host $line
        [Console]::Out.Flush()
        $writer.WriteLine($line)
        $writer.Flush()
    }
    $trainExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousPreference
    $writer.Dispose()
}
if ($trainExitCode -ne 0) {
    throw "Resume stopped with exit code $trainExitCode. The last saved checkpoint was preserved. Inspect $logPath and run this file again after fixing the reported error."
}

Write-Host ""
Write-Host "Resume completed. Verifying checkpoint load and generation..."
$checkLog = Join-Path $logDir "checkpoint-check.log"
$checkWriter = [System.IO.StreamWriter]::new($checkLog, $true, [System.Text.UTF8Encoding]::new($false))
$checkWriter.WriteLine("===== $(Get-Date -Format s) | post-resume check =====")
$checkWriter.Flush()
try {
    $ErrorActionPreference = "Continue"
    & $pythonExe -u $pythonScript generate --output $checkpoint --text "Explain in simple terms what a neural network is." --max-new-tokens 80 2>&1 | ForEach-Object {
        $line = [string]$_
        Write-Host $line
        [Console]::Out.Flush()
        $checkWriter.WriteLine($line)
        $checkWriter.Flush()
    }
    $checkExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousPreference
    $checkWriter.Dispose()
}
if ($checkExitCode -ne 0) { throw "Checkpoint generation check failed; saved model was preserved. Inspect $checkLog." }

Write-Host ""
Write-Host "Checkpoint: $checkpoint"
Write-Host "Resume log: $logPath"
Write-Host "Snapshots: $(Join-Path $workDir 'checkpoints')"
