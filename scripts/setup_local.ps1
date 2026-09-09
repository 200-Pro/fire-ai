$localRoot = "C:\fire-ai-local"
$venvPython = "$localRoot\.venv\Scripts\python.exe"
$venvYolo = "$localRoot\.venv\Scripts\yolo.exe"
$requirementsFile = Join-Path $PSScriptRoot "..\requirements.txt"

New-Item -ItemType Directory -Force `
  "$localRoot\data\raw", `
  "$localRoot\data\processed", `
  "$localRoot\weights", `
  "$localRoot\runs\fire", `
  "$localRoot\mlruns" | Out-Null

if (-not (Test-Path -LiteralPath $venvPython)) {
  py -3.12 -m venv "$localRoot\.venv"
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& $venvPython -m pip install -r $requirementsFile

& $venvYolo settings `
  datasets_dir="C:/fire-ai-local/data" `
  weights_dir="C:/fire-ai-local/weights" `
  runs_dir="C:/fire-ai-local/runs"

& $venvPython (Join-Path $PSScriptRoot "verify_environment.py")
& $venvYolo checks

