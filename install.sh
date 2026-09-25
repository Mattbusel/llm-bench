#!/bin/sh
# Install llm-bench from the latest GitHub Release (Linux x86_64, macOS arm64/x86_64).
#   curl -fsSL https://raw.githubusercontent.com/Mattbusel/llm-bench/main/install.sh | sh
# Options (environment): LLM_BENCH_VERSION=v0.2.1 to pin, INSTALL_DIR=/somewhere to change ~/.local/bin
set -eu

REPO="Mattbusel/llm-bench"
BIN="llm-bench"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"

say() { printf '%s\n' "$*"; }
fail() { printf 'llm-bench installer: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "needs '$1' but it is not installed"; }

need uname; need tar; need mktemp
if command -v curl >/dev/null 2>&1; then
  fetch() { curl -fsSL "$1" -o "$2"; }
  fetch_stdout() { curl -fsSL "$1"; }
elif command -v wget >/dev/null 2>&1; then
  fetch() { wget -qO "$2" "$1"; }
  fetch_stdout() { wget -qO- "$1"; }
else
  fail "needs curl or wget"
fi

os=$(uname -s); arch=$(uname -m)
case "$os-$arch" in
  Linux-x86_64|Linux-amd64) target="x86_64-unknown-linux-gnu" ;;
  Darwin-arm64|Darwin-aarch64) target="aarch64-apple-darwin" ;;
  Darwin-x86_64) target="x86_64-apple-darwin" ;;
  MINGW*|MSYS*|CYGWIN*) fail "on Windows run this in PowerShell instead: irm https://raw.githubusercontent.com/$REPO/main/install.ps1 | iex" ;;
  *) fail "no prebuilt binary for $os $arch. Install with Rust instead: cargo install llm-bench" ;;
esac

tag="${LLM_BENCH_VERSION:-}"
if [ -z "$tag" ]; then
  tag=$(fetch_stdout "https://api.github.com/repos/$REPO/releases/latest" \
        | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n 1)
  [ -n "$tag" ] || fail "could not find the latest release (GitHub API rate limit?). Set LLM_BENCH_VERSION=v0.2.1 and retry."
fi

name="$BIN-$tag-$target"
base="https://github.com/$REPO/releases/download/$tag"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT INT TERM

say "Downloading $name.tar.gz"
fetch "$base/$name.tar.gz" "$tmp/$name.tar.gz" || fail "download failed: $base/$name.tar.gz"
fetch "$base/SHA256SUMS.txt" "$tmp/SHA256SUMS.txt" || fail "could not download SHA256SUMS.txt"

expected=$(grep " $name.tar.gz\$" "$tmp/SHA256SUMS.txt" | cut -d ' ' -f 1)
[ -n "$expected" ] || fail "no checksum for $name.tar.gz in SHA256SUMS.txt"
if command -v sha256sum >/dev/null 2>&1; then
  actual=$(sha256sum "$tmp/$name.tar.gz" | cut -d ' ' -f 1)
else
  need shasum; actual=$(shasum -a 256 "$tmp/$name.tar.gz" | cut -d ' ' -f 1)
fi
[ "$expected" = "$actual" ] || fail "checksum mismatch (expected $expected, got $actual)"
say "Checksum OK"

tar -xzf "$tmp/$name.tar.gz" -C "$tmp"
mkdir -p "$INSTALL_DIR"
cp "$tmp/$name/$BIN" "$INSTALL_DIR/$BIN"
chmod +x "$INSTALL_DIR/$BIN"
if [ "$os" = "Darwin" ]; then xattr -d com.apple.quarantine "$INSTALL_DIR/$BIN" 2>/dev/null || true; fi

say "Installed $("$INSTALL_DIR/$BIN" --version) to $INSTALL_DIR/$BIN"
case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *) say "Note: $INSTALL_DIR is not on your PATH. Add this to your shell profile:"
     say "  export PATH=\"$INSTALL_DIR:\$PATH\"" ;;
esac
say "Next: llm-bench models"
