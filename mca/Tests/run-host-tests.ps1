param(
  [string]$Compiler = 'gcc',
  [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$testBuild = Join-Path $PSScriptRoot '.build'
New-Item -ItemType Directory -Force $testBuild | Out-Null
$sources = @('Core\Src\mca_profile.c','Core\Src\mca.c','Core\Src\mca_sim.c','Core\Src\mca_frame.c','Tests\reference_sim.c','Tests\test_mca.c') | ForEach-Object { Join-Path $root $_ }
$exe = Join-Path $testBuild 'test_mca.exe'
& $Compiler -std=c11 -O2 -Wall -Wextra -Werror -pedantic ('-I'+(Join-Path $root 'Core\Inc')) @sources -lm -o $exe
if ($LASTEXITCODE -ne 0) { throw 'Host compilation failed' }
Push-Location $testBuild
try {
  & $exe
  if ($LASTEXITCODE -ne 0) { throw 'C tests failed' }
  & $Python (Join-Path $PSScriptRoot 'verify_frame.py')
  if ($LASTEXITCODE -ne 0) { throw 'Independent frame verification failed' }
} finally { Pop-Location }
