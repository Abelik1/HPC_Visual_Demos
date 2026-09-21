# Obtain a CINECA SSH certificate and open a session on Leonardo.
#
# Leonardo does not use authorized_keys.  The cluster trusts CINECA's CA and
# accepts only certificates it signed, so an ordinary keypair is not enough.
# This requests a fresh certificate (valid 12 hours) and connects with it.
#
#   .\scripts\leonardo_login.ps1 -Email you@example.org -User abelik00
#
# The certificate request opens a browser: enter your HPC password and the
# one-time code from your authenticator app.  Re-run whenever the 12 hours
# lapse; -CertOnly refreshes the certificate without opening a session, which
# is what rsync and scp need.
#
# Where jobs are launched unattended (the dashboard, the demo stand), add
# -NoPassword: their batch-mode ssh cannot type a key passphrase.
#
#   .\scripts\leonardo_login.ps1 -Email you@example.org -User abelik00 -CertOnly -NoPassword

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Email,
    [Parameter(Mandatory = $true)][string]$User,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\cineca_leonardo",
    [string]$LoginHost = 'login.leonardo.cineca.it',
    [switch]$NoPassword,
    [switch]$CertOnly
)

$ErrorActionPreference = 'Stop'

$step = Get-Command step -ErrorAction SilentlyContinue
if (-not $step) {
    $step = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter step.exe -Recurse -ErrorAction SilentlyContinue |
            Select-Object -First 1
    if (-not $step) { throw "step CLI not found.  Install it with: winget install Smallstep.step" }
}
if ($step.Source) { $stepExe = $step.Source } else { $stepExe = $step.FullName }

# Bootstrap the CA on first use, so this works on a fresh exhibition machine.
if (-not $env:STEPPATH) { $env:STEPPATH = [Environment]::GetEnvironmentVariable('STEPPATH', 'User') }
if ($env:STEPPATH) { $stepHome = $env:STEPPATH } else { $stepHome = "$env:USERPROFILE\.step" }
if (-not (Test-Path (Join-Path $stepHome 'config\defaults.json'))) {
    Write-Host 'Bootstrapping the CINECA certificate authority...' -ForegroundColor Cyan
    & $stepExe ca bootstrap --ca-url=https://sshproxy.hpc.cineca.it `
        --fingerprint 2ae1543202304d3f434bdc1a2c92eff2cd2b02110206ef06317e70c1c1735ecd
    if ($LASTEXITCODE -ne 0) { throw 'CA bootstrap failed.' }
}

$certArgs = @('ssh', 'certificate', $Email, $KeyPath, '--provisioner', 'cineca-hpc', '--force')
if ($NoPassword) { $certArgs += @('--no-password', '--insecure') }

Write-Host "Requesting a certificate for $Email ..." -ForegroundColor Cyan
& $stepExe @certArgs
if ($LASTEXITCODE -ne 0) { throw 'Certificate request failed.' }

# step writes the private key to $KeyPath and the certificate beside it.
Write-Host "Key  : $KeyPath"          -ForegroundColor Green
Write-Host "Cert : $KeyPath-cert.pub" -ForegroundColor Green

if ($CertOnly) {
    Write-Host "`nUse it with:  ssh -i `"$KeyPath`" $User@$LoginHost" -ForegroundColor Yellow
    return
}

ssh -i "$KeyPath" "$User@$LoginHost"
