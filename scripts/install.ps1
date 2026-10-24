# Windows counterpart of install.sh: fetches the models, builds the app and
# runs its installer. Safe to re-run: models already in src-tauri\resources\
# are skipped.
#   scripts\install.ps1              # models + build + install
#   scripts\install.ps1 -ModelsOnly  # just the models, for `bun run tauri dev`
#   scripts\install.ps1 -NoMedium    # skip medium.en (539 MB); download it from Settings later
# If PowerShell says running scripts is disabled:
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1
param([switch]$ModelsOnly, [switch]$NoMedium)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Res = Join-Path $Root 'src-tauri\resources'

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
    throw "$tool not found; see Install in README.md"
  }
}
# whisper.cpp's Vulkan backend builds against the SDK; its installer sets this.
if (-not $env:VULKAN_SDK) {
  throw 'VULKAN_SDK is not set; install the Vulkan SDK (https://vulkan.lunarg.com/sdk/home) and open a new terminal'
}

Set-Location $Root
bun install
if ($LASTEXITCODE -ne 0) { throw 'bun install failed' }
bun run tauri build --bundles nsis
if ($LASTEXITCODE -ne 0) { throw 'build failed' }
$setup = Get-ChildItem (Join-Path $Root 'src-tauri\target\release\bundle\nsis\*.exe') |
  Sort-Object LastWriteTime | Select-Object -Last 1
Write-Host "running $($setup.FullName)"
Start-Process $setup.FullName -Wait
