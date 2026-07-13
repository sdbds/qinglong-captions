param(
    [string]$BindHost = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8222,
    [Alias("ModelSize")]
    [ValidateSet("small", "medium", "large")]
    [string]$Model = "large",
    [ValidatePattern("^(auto|cpu|cuda(:[0-9]+)?)$")]
    [string]$Device = "auto",
    [ValidateRange(0, 1024)]
    [int]$BatchSize = 0,
    [switch]$NoBrowser
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
$UrlHost = if ($DisplayHost.Contains(":")) { "[$DisplayHost]" } else { $DisplayHost }
$WebUrl = "http://${UrlHost}:$Port"
Write-Output "MuScriptor WebUI: $WebUrl"
Write-Output "Model: $Model | Device: $Device | Batch size: $BatchSize (0 = recorded VRAM profile)"
Write-Output "Press Ctrl+C to stop the server."

$ServerArguments = @(
    "-m",
    "module.muscriptor_tool.webui",
    "--host",
    $BindHost,
    "--port",
    $Port,
    "--model",
    $Model,
    "--device",
    $Device,
    "--batch-size",
    $BatchSize
)

$ServerProcess = $null
$ExitCode = 1
try {
    $ServerProcess = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $ServerArguments `
        -NoNewWindow `
        -PassThru

    if (-not $NoBrowser) {
        Write-Output "Waiting for the WebUI server before opening the browser..."
        $ServerReady = $false
        while (-not $ServerProcess.HasExited) {
            try {
                $Health = Invoke-WebRequest `
                    -Uri "$WebUrl/health" `
                    -UseBasicParsing `
                    -TimeoutSec 2
                if ($Health.StatusCode -eq 200) {
                    $ServerReady = $true
                    break
                }
            } catch {}
            Start-Sleep -Milliseconds 750
            $ServerProcess.Refresh()
        }
        if (-not $ServerReady -and $ServerProcess.HasExited) {
            throw "MuScriptor server exited before the WebUI became ready (code $($ServerProcess.ExitCode))."
        }
        try {
            Start-Process $WebUrl
        } catch {
            Write-Warning "Could not open the default browser. Open $WebUrl manually."
        }
    }

    $ServerProcess.WaitForExit()
    $ExitCode = $ServerProcess.ExitCode
} finally {
    if ($null -ne $ServerProcess -and -not $ServerProcess.HasExited) {
        Stop-Process -Id $ServerProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
exit $ExitCode
