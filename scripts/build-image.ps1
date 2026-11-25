param(
    [ValidateSet('kev-0.8b','kev-4b','kev-9b','kev-27b')][string]$Model,
    [switch]$Cuda,
    [switch]$Download,
    [string]$OutputArchive = '',
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$manage = Join-Path $PSScriptRoot 'manage.py'
$modelArgs = @()
if ($Model) { $modelArgs += @('--model', $Model) }
if ($Download) {
    & $Python $manage download @modelArgs
    if ($LASTEXITCODE -ne 0) { throw 'Model download failed' }
}
$buildArgs = @('build') + $modelArgs
if ($Cuda) { $buildArgs += '--cuda' }
& $Python $manage @buildArgs
if ($LASTEXITCODE -ne 0) { throw 'Image build failed' }
if ($OutputArchive) {
    $exportArgs = @('export', '--archive', $OutputArchive)
    if ($Cuda) { $exportArgs += '--cuda' }
    & $Python $manage @exportArgs
    if ($LASTEXITCODE -ne 0) { throw 'Image export failed' }
}

