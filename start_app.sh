#!/usr/bin/env bash
# Start the WeAuto control panel (menu-bar app + web server + bot supervisor).
#
# This is the modern entry point: no terminal window needed. It sets up the
# same venv / dependencies / env files as start_rpa.sh, then launches the
# control panel which supervises the bot as a subprocess and exposes a
# read-only web UI. The bot's stdout is captured by the supervisor (not tee'd
# to a terminal), so this script runs cleanly in the background.
#
# Usage:
#   ./start_app.sh                       # menu-bar app + web + bot
#   ./start_app.sh --headless            # no menu bar (SSH / CI); web + bot only
#   ./start_app.sh --no-bot              # panel only, do not spawn the bot
#   ./start_app.sh --host 0.0.0.0 --port 8721   # allow remote access
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${UV_PROJECT_ENVIRONMENT:-.venv312}"
UV_PYTHON="${UV_PYTHON:-3.12}"
PLAYWRIGHT_MARKER="$VENV_DIR/.playwright_chromium_installed"

_needs_refresh() {
  local marker="$1"
  shift
  if [[ ! -f "$marker" ]]; then
    return 0
  fi
  local dep=""
  for dep in "$@"; do
    if [[ -f "$dep" && "$dep" -nt "$marker" ]]; then
      return 0
    fi
  done
  return 1
}

# Optional local env files (not committed) for secrets/endpoints.
for env_file in "$ROOT_DIR/.env.weauto" "$ROOT_DIR/.env"; do
  if [[ -f "$env_file" ]]; then
    echo "[env] loading $(basename "$env_file")"
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
done

# --- dependency setup (mirrors start_rpa.sh) ---
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
if [[ ! -x "$UV_BIN" ]] && command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
fi
if [[ ! -x "$UV_BIN" ]]; then
  echo "[fatal] standalone uv not found at $UV_BIN" >&2
  exit 1
fi

echo "[setup] syncing dependencies with uv (Python $UV_PYTHON)"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
if ! "$UV_BIN" python find "$UV_PYTHON" >/dev/null 2>&1; then
    echo "[setup] installing managed Python $UV_PYTHON"
    "$UV_BIN" python install "$UV_PYTHON"
fi
"$UV_BIN" sync --locked --python "$UV_PYTHON"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if python -c "import playwright" >/dev/null 2>&1; then
  if _needs_refresh "$PLAYWRIGHT_MARKER" requirements.txt pyproject.toml uv.lock; then
    echo "[setup] installing Playwright Chromium"
    python -m playwright install chromium
    date > "$PLAYWRIGHT_MARKER"
  fi
fi

# High-res Quartz capture leaks native memory in long-running loops on some
# macOS builds. Keep it off by default; the bot reads this env var directly.
export WEAUTO_SCREENSHOT_HIGH_RES="${WEAUTO_SCREENSHOT_HIGH_RES:-0}"

# Pass remaining args through to the control panel.
echo "[run] python -u -m app.main $*"
exec python -u -m app.main "$@"
