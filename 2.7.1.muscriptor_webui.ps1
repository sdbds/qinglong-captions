param(
    [string]$BindHost = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8222,
    [Alias("ModelSize")]
    [ValidateSet("small", "medium", "large")]
    [string]$Model = "large",
    [ValidatePattern("^(auto|cpu|cuda(:[0-9]+)?)$")]
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:HF_HOME = Join-Path $PSScriptRoot "huggingface"

$UvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $UvCommand) {
    Write-Error "uv was not found on PATH. Run 1.install-uv-qinglong.ps1 first."
    exit 1
}

$PythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Output "Creating shared project environment: .venv"
    & $UvCommand.Source venv .venv
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
$PythonExe = (Resolve-Path $PythonExe).Path

$InstallArguments = @(
    "pip",
    "install",
    "--no-build-isolation",
    "--python",
    $PythonExe,
    "-r",
    "pyproject.toml",
    "--extra",
    "muscriptor-local"
)
Write-Output "Installing MuScriptor into shared .venv"
& $UvCommand.Source @InstallArguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$DisplayHost = if ($BindHost -in @("0.0.0.0", "::")) { "127.0.0.1" } else { $BindHost }
Write-Output "MuScriptor WebUI: http://${DisplayHost}:$Port"
Write-Output "Model: $Model | Device: $Device"
Write-Output "Press Ctrl+C to stop the server."

& $PythonExe -m muscriptor.main serve `
    --host $BindHost `
    --port $Port `
    --model $Model `
    --device $Device
exit $LASTEXITCODE
