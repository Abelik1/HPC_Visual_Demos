[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^[1-9][0-9]*$')]
    [string]$JobId,

    [string]$UserName = 'abelik',
    [string]$HostName = 'login.brainplusplus.bg',
    [ValidateRange(1, 65535)]
    [int]$Port = 2226,
    [string]$RemoteRunsRoot = '/weka/ehpc-school-2026/abelik/runs'
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$localRunsRoot = Join-Path $projectRoot 'runs'
$runName = "galaxy_collision_3d_$JobId"
$remoteRun = "$UserName@$HostName`:$RemoteRunsRoot/$runName"

New-Item -ItemType Directory -Force -Path $localRunsRoot | Out-Null

# Copy the whole saved-run contract: frames, meta/status, reveal image, and
# the interactive 3-D states.  Re-running this command refreshes the local
# copy after further remote frames have been written.
& scp -P $Port -r $remoteRun $localRunsRoot
if ($LASTEXITCODE -ne 0) {
    throw "Discoverer transfer failed for $remoteRun (scp exit code $LASTEXITCODE)."
}

$localRun = Join-Path $localRunsRoot $runName
if (-not (Test-Path -LiteralPath (Join-Path $localRun 'meta.json'))) {
    throw "Transfer finished but $localRun does not contain meta.json."
}

Write-Host "Saved Discoverer run to $localRun"
Write-Host 'Start the local dashboard with: python app.py'
