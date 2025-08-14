# Run as Administrator. Block user 'CHILD-USER-NAME' from launching Ollama.

try { chcp 65001 > $null } catch {}
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$IsAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $IsAdmin) { Write-Error "Run this script as Administrator."; exit 1 }

$user = "$env:COMPUTERNAME\CHILD-USER-NAME"

# Discover possible ollama.exe locations
$exePaths = @()
foreach ($root in @("C:\Program Files","C:\Program Files (x86)","C:\Users")) {
  try {
    if (Test-Path $root) {
      $exePaths += Get-ChildItem -LiteralPath $root -Filter "ollama.exe" -Recurse -ErrorAction SilentlyContinue |
                   ForEach-Object { $_.FullName }
    }
  } catch {}
}
$exePaths = $exePaths | Where-Object { $_ -match "\\Ollama\\ollama\.exe$" } | Sort-Object -Unique

if ($exePaths.Count -eq 0) {
  Write-Host "No ollama.exe found by scan. Will still apply to common dirs."
}

# Deny Read/Execute to user on exe and its folder (with inheritance)
foreach ($exe in $exePaths) {
  try {
    $dir = Split-Path $exe -Parent
    cmd /c "icacls `"$dir`" /deny `"$user`":(OI)(CI)(RX)" | Out-Null
    cmd /c "icacls `"$exe`" /deny `"$user`":(RX)"         | Out-Null
    Write-Host "Applied DENY on: $exe"
  } catch {
    Write-Host "Failed on: $exe ; $($_.Exception.Message)"
  }
}

# Apply also on typical install dirs (in case of updates)
$extraDirs = @(
  "C:\Program Files\Ollama",
  "C:\Users\sunpi\AppData\Local\Programs\Ollama"
)
foreach ($p in $extraDirs) {
  if (Test-Path $p) {
    try {
      cmd /c "icacls `"$p`" /deny `"$user`":(OI)(CI)(RX)" | Out-Null
      Write-Host "Applied DENY on dir: $p"
    } catch {
      Write-Host "Failed on dir: $p ; $($_.Exception.Message)"
    }
  }
}

# Hide Start Menu shortcuts (best effort; ignore access errors)
$commonStart = "C:\ProgramData\Microsoft\Windows\Start Menu\Programs"
$userStart   = "C:\Users\CHILD-USER-NAME\AppData\Roaming\Microsoft\Windows\Start Menu\Programs"

try {
  if (Test-Path $commonStart) {
    Get-ChildItem -LiteralPath $commonStart -Filter "*Ollama*.lnk" -Recurse -ErrorAction SilentlyContinue |
      ForEach-Object { try { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue } catch {} }
  }
} catch {}

try {
  if (Test-Path $userStart) {
    Get-ChildItem -LiteralPath $userStart -Filter "*Ollama*.lnk" -Recurse -ErrorAction SilentlyContinue |
      ForEach-Object { try { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue } catch {} }
  }
} catch {}

Write-Host "Done. Test from 'CHILD-USER-NAME' account: Ollama should be blocked."
