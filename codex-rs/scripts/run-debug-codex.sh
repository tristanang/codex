#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODEX_RS_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SOURCE_HOME="${CODEX_SOURCE_HOME:-$HOME/.codex}"
DEBUG_HOME="${CODEX_DEBUG_HOME:-$HOME/.codex-debug}"
BIN_PATH="${CODEX_RS_ROOT}/target/debug/codex"

print_usage() {
  cat <<'USAGE'
Usage:
  run-debug-codex.sh [--resync] [--] [codex args...]

Description:
  Runs the local debug Codex binary with an isolated CODEX_HOME.

Defaults:
  SOURCE_HOME = $HOME/.codex      (override with CODEX_SOURCE_HOME)
  DEBUG_HOME  = $HOME/.codex-debug (override with CODEX_DEBUG_HOME)

Behavior:
  - Ensures DEBUG_HOME exists.
  - Copies config.toml and github-copilot-auth.json from SOURCE_HOME
    into DEBUG_HOME if missing.
  - With --resync, forces overwrite from SOURCE_HOME.
  - Launches: CODEX_HOME=DEBUG_HOME ./target/debug/codex ...
USAGE
}

resync=false
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  print_usage
  exit 0
fi

if [[ "${1:-}" == "--resync" ]]; then
  resync=true
  shift
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

mkdir -p "$DEBUG_HOME"

copy_if_needed() {
  local name="$1"
  local src="$SOURCE_HOME/$name"
  local dst="$DEBUG_HOME/$name"

  if [[ ! -f "$src" ]]; then
    return 0
  fi

  if [[ "$resync" == "true" ]]; then
    cp "$src" "$dst"
    return 0
  fi

  if [[ ! -f "$dst" ]]; then
    cp "$src" "$dst"
  fi
}

copy_if_needed "config.toml"
copy_if_needed "github-copilot-auth.json"

if [[ ! -x "$BIN_PATH" ]]; then
  echo "Debug binary not found at: $BIN_PATH" >&2
  echo "Build it with: (cd $CODEX_RS_ROOT && cargo build -p codex-cli)" >&2
  exit 1
fi

echo "Using isolated CODEX_HOME: $DEBUG_HOME"
exec env CODEX_HOME="$DEBUG_HOME" "$BIN_PATH" "$@"
