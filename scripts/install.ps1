# Windows counterpart of install.sh: fetches the models, builds the app and
# runs its installer. Safe to re-run: models already in src-tauri\resources\
# are skipped.
#   scripts\install.ps1              # models + build + install
#   scripts\install.ps1 -ModelsOnly  # just the models, for `bun run tauri dev`
#   scripts\install.ps1 -NoMedium    # skip medium.en (539 MB); download it from Settings later
#   scripts\install.ps1 -InstallDeps # first install the build tools with winget
# If PowerShell says running scripts is disabled:
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1
param([switch]$ModelsOnly, [switch]$NoMedium, [switch]$InstallDeps)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Res = Join-Path $Root 'src-tauri\resources'

# Installers write PATH and VULKAN_SDK to the registry; this session only sees
# them after re-reading it.
function Sync-Env {
  $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
    [Environment]::GetEnvironmentVariable('Path', 'User')
  foreach ($scope in 'Machine', 'User') {
    $v = [Environment]::GetEnvironmentVariable('VULKAN_SDK', $scope)
    if ($v) { $env:VULKAN_SDK = $v }
  }
}

function Test-VCTools {
  $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
  (Test-Path $vswhere) -and
    (& $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
}

# Skips whatever is already there. A failed install only warns: the checks
# before the build report anything still missing.
function Install-Dep($Id, [scriptblock]$Have, $Override) {
  if (& $Have) { Write-Host "have $Id"; return $false }
  Write-Host "installing $Id"
  $wargs = @('install', '-e', '--id', $Id, '--accept-package-agreements', '--accept-source-agreements')
  if ($Override) { $wargs += @('--override', $Override) }
  winget @wargs | Out-Host
  if ($LASTEXITCODE -ne 0) { Write-Warning "winget exited with $LASTEXITCODE for $Id" }
  Sync-Env
  return $true
}

if ($InstallDeps) {
  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw 'winget not found; install "App Installer" from the Microsoft Store'
  }
  # MSVC first: rustup's MSVC toolchain links with it.
  Install-Dep 'Microsoft.VisualStudio.2022.BuildTools' { Test-VCTools } `
    '--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended' | Out-Null
  if (Install-Dep 'Rustlang.Rustup' { Get-Command cargo -ErrorAction SilentlyContinue }) {
    rustup default stable
  }
  Install-Dep 'Oven-sh.Bun' { Get-Command bun -ErrorAction SilentlyContinue } | Out-Null
  Install-Dep 'Kitware.CMake' { Get-Command cmake -ErrorAction SilentlyContinue } | Out-Null
  Install-Dep 'LLVM.LLVM' { Test-Path "$env:ProgramFiles\LLVM\bin\libclang.dll" } | Out-Null
  Install-Dep 'KhronosGroup.VulkanSDK' { $env:VULKAN_SDK } | Out-Null
}

# Downloads to a .part file first so an interrupted download isn't mistaken
# for a finished one on the next run. curl.exe ships with Windows 10 and up,
# and is far faster than Invoke-WebRequest on Windows PowerShell 5.
function Fetch($Url, $Name) {
  $dest = Join-Path $Res $Name
  if (Test-Path $dest) { Write-Host "have $Name"; return }
  Write-Host "fetching $Name"
  curl.exe -fL --progress-bar -o "$dest.part" $Url
  if ($LASTEXITCODE -ne 0) { throw "download failed: $Name" }
  Move-Item "$dest.part" $dest
}
New-Item -ItemType Directory -Force $Res | Out-Null
$HF = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main'
Fetch "$HF/ggml-tiny.en-q5_1.bin" 'ggml-tiny.en-q5_1.bin'
Fetch 'https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx' 'silero_vad.onnx'
if (-not $NoMedium) { Fetch "$HF/ggml-medium.en-q5_0.bin" 'ggml-medium.en-q5_0.bin' }

# The camera models' converter needs Nix, which doesn't run on Windows.
if (Test-Path (Join-Path $Res 'vision')) { Write-Host 'have vision\' }
else { Write-Host 'skipping camera models: scripts/convert-vision-models.sh needs Nix (the app works audio-only without them)' }

if ($ModelsOnly) { exit 0 }

foreach ($tool in 'bun', 'cargo', 'cmake') {
  if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
    throw "$tool not found; run with -InstallDeps, or see Install in README.md"
  }
}
if (-not (Test-VCTools)) {
  throw 'MSVC build tools not found; run with -InstallDeps, or see Install in README.md'
}
# bindgen needs libclang; LLVM's installer doesn't put it on PATH.
if (-not $env:LIBCLANG_PATH -and (Test-Path "$env:ProgramFiles\LLVM\bin\libclang.dll")) {
  $env:LIBCLANG_PATH = "$env:ProgramFiles\LLVM\bin"
}
# whisper.cpp's Vulkan backend builds against the SDK; its installer sets this.
if (-not $env:VULKAN_SDK) {
  throw 'VULKAN_SDK is not set; run with -InstallDeps, or install the Vulkan SDK (https://vulkan.lunarg.com/sdk/home) and open a new terminal'
}

# whisper.cpp's Vulkan shader build nests ~245 characters under the target dir,
# and MSBuild fails past Windows' 260-character path limit (PathTooLongException)
# unless long paths are enabled. Enable them (one UAC prompt) and build in a
# short directory unless one is already set.
$fs = 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem'
if ((Get-ItemProperty $fs -ErrorAction SilentlyContinue).LongPathsEnabled -ne 1) {
  Write-Host 'enabling Windows long paths (asks for admin)'
  try {
    Start-Process powershell -Verb RunAs -Wait -ArgumentList '-NoProfile', '-Command',
      "Set-ItemProperty '$fs' -Name LongPathsEnabled -Value 1 -Type DWord"
  } catch {
    throw 'long paths must be enabled to build; accept the admin prompt, or set HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled to 1'
  }
}
if (-not $env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR = "$env:SystemDrive\sb" }

Set-Location $Root
bun install
if ($LASTEXITCODE -ne 0) { throw 'bun install failed' }
bun run tauri build --bundles nsis
if ($LASTEXITCODE -ne 0) { throw 'build failed' }
$setup = Get-ChildItem (Join-Path $env:CARGO_TARGET_DIR 'release\bundle\nsis\*.exe') |
  Sort-Object LastWriteTime | Select-Object -Last 1
Write-Host "running $($setup.FullName)"
Start-Process $setup.FullName -Wait
