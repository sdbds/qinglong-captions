[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet("muscriptor", "sheetsage2", "dettato")]
    [string]$Backend = "muscriptor",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:PYTHONPATH = "$PSScriptRoot$([System.IO.Path]::PathSeparator)$($env:PYTHONPATH)"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:HF_HOME = Join-Path $PSScriptRoot "huggingface"

$UvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $UvCommand) {
    Write-Error "uv was not found on PATH. Run 1.install-uv-qinglong.ps1 first."
    exit 1
}

if ($Backend -eq "sheetsage2") {
    & $UvCommand.Source run --no-project --script "./module/sheetsage2_tool.py" @Arguments
    exit $LASTEXITCODE
}

$PythonCandidates = @(
    "./.venv/Scripts/python.exe",
    "./venv/Scripts/python.exe",
    "./.venv/bin/python",
    "./venv/bin/python"
)
$PythonExe = $null
foreach ($Candidate in $PythonCandidates) {
    if (Test-Path $Candidate) {
        $PythonExe = (Resolve-Path $Candidate).Path
        break
    }
}
if (-not $PythonExe) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) {
        $PythonExe = $PythonCommand.Source
    }
}
if (-not $PythonExe) {
    Write-Error "Python was not found."
    exit 1
}

$RuntimeExtra = if ($Backend -eq "dettato") { "dettato-local" } else { "muscriptor-local" }
$InstallArguments = @(
    "pip",
    "install",
    "--no-build-isolation",
    "--python",
    $PythonExe,
    "-r",
    "pyproject.toml",
    "--extra",
    $RuntimeExtra
)
if ($Backend -eq "dettato" -and -not $env:UV_TORCH_BACKEND) {
    $InstallArguments += @("--torch-backend", "auto")
}
& $UvCommand.Source @InstallArguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($Backend -eq "dettato") {
    & $PythonExe -m module.dettato_tool @Arguments
} else {
    & $PythonExe -m module.muscriptor_tool.cli batch @Arguments
}
exit $LASTEXITCODE
