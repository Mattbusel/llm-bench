# Install llm-bench on Windows from the latest GitHub Release.
#   irm https://raw.githubusercontent.com/Mattbusel/llm-bench/main/install.ps1 | iex
# Pin a version first with:  $env:LLM_BENCH_VERSION = "v0.2.1"
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$repo = "Mattbusel/llm-bench"
$bin = "llm-bench"
$target = "x86_64-pc-windows-msvc"
$dest = Join-Path $env:LOCALAPPDATA "Programs\$bin"

if (-not [Environment]::Is64BitOperatingSystem) { throw "llm-bench needs 64-bit Windows. Or build it: cargo install llm-bench" }

$tag = $env:LLM_BENCH_VERSION
if (-not $tag) {
    $tag = (Invoke-RestMethod "https://api.github.com/repos/$repo/releases/latest" -Headers @{ "User-Agent" = "llm-bench-installer" }).tag_name
}
if (-not $tag) { throw "Could not find the latest release. Set `$env:LLM_BENCH_VERSION = 'v0.2.1' and retry." }

$name = "$bin-$tag-$target"
$base = "https://github.com/$repo/releases/download/$tag"
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("$bin-" + [Guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    Write-Host "Downloading $name.zip"
    Invoke-WebRequest "$base/$name.zip" -OutFile "$tmp\$name.zip" -UseBasicParsing
    Invoke-WebRequest "$base/SHA256SUMS.txt" -OutFile "$tmp\SHA256SUMS.txt" -UseBasicParsing

    $line = Get-Content "$tmp\SHA256SUMS.txt" | Where-Object { $_ -match "\s$([regex]::Escape("$name.zip"))$" } | Select-Object -First 1
    if (-not $line) { throw "No checksum for $name.zip in SHA256SUMS.txt" }
    $expected = ($line -split "\s+")[0].ToLower()
    $actual = (Get-FileHash "$tmp\$name.zip" -Algorithm SHA256).Hash.ToLower()
    if ($expected -ne $actual) { throw "Checksum mismatch (expected $expected, got $actual)" }
    Write-Host "Checksum OK"

    Expand-Archive "$tmp\$name.zip" -DestinationPath $tmp -Force
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Copy-Item "$tmp\$name\$bin.exe" "$dest\$bin.exe" -Force
} finally {
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $userPath) { $userPath = "" }
if (($userPath -split ";") -notcontains $dest) {
    [Environment]::SetEnvironmentVariable("Path", ($userPath.TrimEnd(";") + ";" + $dest).TrimStart(";"), "User")
    Write-Host "Added $dest to your user PATH (open a new terminal to use it)."
}
$env:Path = "$env:Path;$dest"
Write-Host ("Installed " + (& "$dest\$bin.exe" --version) + " to $dest\$bin.exe")
Write-Host "Next: llm-bench models"
