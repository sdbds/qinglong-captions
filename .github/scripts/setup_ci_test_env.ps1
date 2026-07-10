$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

Set-Location $env:GITHUB_WORKSPACE
python -m pip install --upgrade pip uv wheel_stub setuptools wheel tomli
if ($LASTEXITCODE -ne 0) { throw "install uv failed" }

$RunnerPython = (Get-Command python).Source
uv venv --python $RunnerPython
if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }

$PythonExe = if ($IsWindows) {
    (Resolve-Path "./.venv/Scripts/python.exe").Path
} else {
    (Resolve-Path "./.venv/bin/python").Path
}
uv pip install --python $PythonExe setuptools wheel wheel_stub
if ($LASTEXITCODE -ne 0) { throw "uv pip bootstrap failed" }

$IndexStrategy = if ([string]::IsNullOrWhiteSpace($env:UV_INDEX_STRATEGY)) {
    "unsafe-best-match"
} else {
    $env:UV_INDEX_STRATEGY
}
$UvExportProjectDir = Join-Path $env:RUNNER_TEMP "qinglong-ci-test-project"
Remove-Item $UvExportProjectDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $UvExportProjectDir | Out-Null
$CiPyproject = Join-Path $UvExportProjectDir "pyproject.toml"
python .github/scripts/render_ci_test_project.py --source pyproject.toml --output $CiPyproject
if ($LASTEXITCODE -ne 0) { throw "render ci test project failed" }

$RequirementsFile = Join-Path $env:RUNNER_TEMP "qinglong-ci-test-requirements.txt"
uv export --project $UvExportProjectDir --python $RunnerPython --index-strategy $IndexStrategy --group test --no-emit-project --format requirements-txt --output-file $RequirementsFile
if ($LASTEXITCODE -ne 0) { throw "uv export failed" }
uv pip install --python $PythonExe --index-strategy $IndexStrategy -r $RequirementsFile
if ($LASTEXITCODE -ne 0) { throw "uv pip install failed" }

"VENV_PYTHON=$PythonExe" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
