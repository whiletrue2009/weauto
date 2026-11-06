#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${UV_PROJECT_ENVIRONMENT:-.venv312}"
UV_PYTHON="${UV_PYTHON:-3.12}"
CONFIG_PATH="${1:-config.toml}"
EXTRA_ARGS=()
if (( "$#" > 1 )); then
  EXTRA_ARGS=("${@:2}")
fi
PLAYWRIGHT_MARKER="$VENV_DIR/.playwright_chromium_installed"
LOG_DIR="$ROOT_DIR/logs"
LOG_KEEP_MAX_FILES="${LOG_KEEP_MAX_FILES:-40}"
LOG_KEEP_MAX_TOTAL_MB="${LOG_KEEP_MAX_TOTAL_MB:-512}"

_file_size_bytes() {
  local file="$1"
  if stat -f%z "$file" >/dev/null 2>&1; then
    stat -f%z "$file"
  else
    stat -c%s "$file"
  fi
}

_prune_old_logs() {
  local keep_files="$1"
  local keep_bytes="$2"
  local idx=0
  local total=0
  local size=0
  local file=""
  while IFS= read -r file; do
    [[ -f "$file" ]] || continue
    idx=$((idx + 1))
    size="$(_file_size_bytes "$file")"
    if [[ ! "$size" =~ ^[0-9]+$ ]]; then
      size=0
    fi
    total=$((total + size))
    if (( idx > keep_files || total > keep_bytes )); then
      rm -f "$file" || true
    fi
  done < <(ls -1t "$LOG_DIR"/rpa_*.log 2>/dev/null || true)
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
source "$VENV_DIR/bin/activate"

if python -c "import playwright" >/dev/null 2>&1; then
  if _needs_refresh "$PLAYWRIGHT_MARKER" requirements.txt pyproject.toml uv.lock; then
    echo "[setup] installing Playwright Chromium"
    python -m playwright install chromium
    date > "$PLAYWRIGHT_MARKER"
  fi
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[setup] $CONFIG_PATH not found, creating from config.toml.example"
  cp config.toml.example "$CONFIG_PATH"
fi

mkdir -p "$LOG_DIR"
if [[ "$LOG_KEEP_MAX_FILES" =~ ^[0-9]+$ && "$LOG_KEEP_MAX_TOTAL_MB" =~ ^[0-9]+$ ]]; then
  _prune_old_logs "$LOG_KEEP_MAX_FILES" "$((LOG_KEEP_MAX_TOTAL_MB * 1024 * 1024))"
fi
LOG_FILE="$LOG_DIR/rpa_$(date +%Y%m%d_%H%M%S).log"
ISSUES_FILE="$LOG_DIR/issues.log"
export WEAUTO_LOG_FILE="$LOG_FILE"
# High-res Quartz capture leaks native memory in long-running loops on some
# macOS builds. Keep it off by default; users can still override with
# WEAUTO_SCREENSHOT_HIGH_RES=1 when they explicitly need it.
export WEAUTO_SCREENSHOT_HIGH_RES="${WEAUTO_SCREENSHOT_HIGH_RES:-0}"

# Preserve interactive terminal width for aligned logs even when piped to tee.
if [[ -t 1 ]]; then
  LOG_WIDTH="$(tput cols 2>/dev/null || echo 140)"
  if [[ "$LOG_WIDTH" =~ ^[0-9]+$ ]]; then
    export WEAUTO_LOG_WIDTH="$LOG_WIDTH"
  fi
fi

EXTRA_DISPLAY=""
if (( ${#EXTRA_ARGS[@]} > 0 )); then
  EXTRA_DISPLAY=" ${EXTRA_ARGS[*]}"
fi
echo "[run] python run.py --config $CONFIG_PATH$EXTRA_DISPLAY"
echo "[log] $LOG_FILE"
echo "[log] retention: max_files=$LOG_KEEP_MAX_FILES max_total_mb=$LOG_KEEP_MAX_TOTAL_MB"

# Issues filter: extract warn/error/skip lines to a separate log for easy daily review.
_issues_pipe() {
  local issues_file="$1"
  while IFS= read -r line; do
    if [[ "$line" =~ \[(warn|error|skip-)\] ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line" >> "$issues_file"
    fi
  done
}

# Keep terminal colors when running through tee, and store plain logs without ANSI codes.
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  export FORCE_COLOR="${FORCE_COLOR:-1}"
  if command -v perl >/dev/null 2>&1; then
    python -u run.py --config "$CONFIG_PATH" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 \
      | tee >(perl -pe 's/\e\[[0-9;]*[A-Za-z]//g' >> "$LOG_FILE") \
            >(_issues_pipe "$ISSUES_FILE")
  else
    python -u run.py --config "$CONFIG_PATH" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 \
      | tee -a "$LOG_FILE" >(_issues_pipe "$ISSUES_FILE")
  fi
else
  python -u run.py --config "$CONFIG_PATH" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 \
    | tee -a "$LOG_FILE" >(_issues_pipe "$ISSUES_FILE")
fi
