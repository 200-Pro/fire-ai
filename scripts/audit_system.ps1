param(
    [string]$LocalRoot = "C:\fire-ai-local",
    [string]$CourseRoot = ""
)

$ErrorActionPreference = "Continue"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($CourseRoot)) {
    $candidate = Join-Path $ProjectRoot "..\.."
    if (Test-Path -LiteralPath (Join-Path $candidate "Data\DFire")) {
        $CourseRoot = (Resolve-Path $candidate).Path
    }
}

Write-Output "=== Computer ==="
Write-Output "Name: $env:COMPUTERNAME"
Write-Output "Processor: $env:PROCESSOR_IDENTIFIER"
Write-Output "Logical processors: $([Environment]::ProcessorCount)"

Write-Output "`n=== NVIDIA ==="
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    & $nvidiaSmi.Source --query-gpu=name,driver_version,memory.total --format=csv,noheader
} else {
    Write-Output "nvidia-smi: NOT FOUND"
}

Write-Output "`n=== Local environment ==="
$python = Join-Path $LocalRoot ".venv\Scripts\python.exe"
$yolo = Join-Path $LocalRoot ".venv\Scripts\yolo.exe"
foreach ($name in @("data", "weights", "runs", "mlruns")) {
    $path = Join-Path $LocalRoot $name
    Write-Output ("{0}: {1}" -f $path, $(if (Test-Path -LiteralPath $path) { "OK" } else { "MISSING" }))
}
if (Test-Path -LiteralPath $python) {
    & $python -c "import sys, torch, cv2, ultralytics; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('OpenCV:', cv2.__version__); print('Ultralytics:', ultralytics.__version__)"
} else {
    Write-Output "Virtual environment: MISSING"
}
if (Test-Path -LiteralPath $yolo) {
    & $yolo settings
}

Write-Output "`n=== Git ==="
git -C $ProjectRoot status --short --branch
git -C $ProjectRoot remote -v
git -C $ProjectRoot log -3 --oneline --decorate

Write-Output "`n=== Shared data ==="
if (-not [string]::IsNullOrWhiteSpace($CourseRoot)) {
    $dfire = Join-Path $CourseRoot "Data\DFire"
    $splitArchive = Get-ChildItem -LiteralPath $dfire -Filter "*.zip" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne "D-Fire.zip" } |
        Sort-Object Length |
        Select-Object -First 1
    $sharedPaths = @(
        (Join-Path $dfire "D-Fire.zip"),
        (Join-Path $dfire "training_results\yolo26n_smoke_test_3ep\weights\best.pt")
    )
    if ($splitArchive) {
        $sharedPaths += $splitArchive.FullName
    } else {
        Write-Output "Split archive: MISSING"
    }
    foreach ($path in $sharedPaths) {
        if (Test-Path -LiteralPath $path) {
            $item = Get-Item -LiteralPath $path
            Write-Output ("{0}: OK ({1} bytes)" -f $path, $item.Length)
        } else {
            Write-Output ("{0}: MISSING" -f $path)
        }
    }
} else {
    Write-Output "CourseRoot not detected. Pass -CourseRoot explicitly."
}

Write-Output "`n=== Secrets ==="
Write-Output ("Project .env present: {0}" -f (Test-Path -LiteralPath (Join-Path $ProjectRoot ".env")))
Write-Output "Secret values are intentionally not printed."
