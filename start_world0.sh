#!/usr/bin/env bash
# World 0 launcher — bootstrap, run, and manage the concept-world agent.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ── Defaults (overridable via env or CLI flags) ──────────────────────
MODE="cli"
PROVIDER="${WORLD0_PROVIDER:-none}"
MODEL="${WORLD0_MODEL:-}"
STORE="${WORLD0_STORE:-$HOME/.pkm_world}"
HOST="${WORLD0_HOST:-127.0.0.1}"
PORT="${WORLD0_PORT:-8420}"
SPACE="${WORLD0_SPACE:-}"
PYTHON_BIN="${WORLD0_PYTHON:-}"
VENV_DIR="${WORLD0_VENV:-$ROOT_DIR/.venv}"
EXTRAS="${WORLD0_EXTRAS:-web}"   # pip extras to install during setup
SERVICE_DIR="${WORLD0_SERVICE_DIR:-$ROOT_DIR/.world0_service}"
NO_VENV=0
VERBOSE=0

# forwarded-to-subcommand args (everything after `--`)
PASSTHROUGH=()

# ── Helpers ──────────────────────────────────────────────────────────
log() { echo -e "\033[36m[world0]\033[0m $*"; }
warn() { echo -e "\033[33m[world0]\033[0m $*" >&2; }
die() { echo -e "\033[31m[world0]\033[0m $*" >&2; exit 1; }

print_usage() {
  cat <<'EOF'
World 0 launcher — cognitive concept-world for LLM agents.

Usage:
  ./start_world0.sh [mode] [options] [-- extra-args…]

Modes:
  cli              Terminal chat (default)
  web              Browser web UI (opens http://HOST:PORT)
  gui              Native desktop GUI (pywebview)
  setup            Create venv and install dependencies
  test             Run pytest across all bricks
  status           Show world/status (quick, no chat)
  reflect          Run consolidation pass
  ask "Q"          One-shot question
  learn "TEXT"     One-shot knowledge intake
  explore "NAME"   Explore one concept
  search "QUERY"   Search concepts
  connect A B      Connect two concepts
  web-search "Q"   Search the public web through the agent
  viz              Generate a graph visualization
  service-start    Start the Web GUI as a resident background service
  service-stop     Stop the resident Web GUI service
  service-restart  Restart the resident Web GUI service
  service-status   Show resident service status
  service-logs     Show resident service logs (use: service-logs -f)
  space ARGS…      Forward to `pkm space …` (list/create/use/rename/delete)
  shell            Drop into a shell with PYTHONPATH wired up

Options:
  --provider NAME    anthropic | openai | azure-openai | none  (default: none)
  --model NAME       Model override (gpt-5.4, claude-sonnet-4-6, …)
  --store PATH       Knowledge store root (default: ~/.pkm_world)
  --space NAME|ID    Use this space for this invocation
  --host HOST        Web bind host (default: 127.0.0.1)
  --port PORT        Web/GUI port  (default: 8420)
  --python PATH      Explicit Python interpreter
  --venv PATH        Virtualenv dir (default: .venv)
  --service-dir PATH Runtime dir for service pid/logs (default: .world0_service)
  --no-venv          Skip venv; use system Python directly
  --extras LIST      pip extras for `setup` (default: web)
  --verbose          Echo the final python command
  -h, --help         Show this help

Environment overrides:
  WORLD0_PROVIDER, WORLD0_MODEL, WORLD0_STORE, WORLD0_SPACE,
  WORLD0_HOST, WORLD0_PORT, WORLD0_PYTHON, WORLD0_VENV, WORLD0_EXTRAS,
  WORLD0_SERVICE_DIR

Examples:
  ./start_world0.sh setup
  ./start_world0.sh                              # interactive CLI
  ./start_world0.sh web --provider anthropic
  ./start_world0.sh web --space work --port 8421
  ./start_world0.sh ask "What is context?"
  ./start_world0.sh learn "Transformers use self-attention…"
  ./start_world0.sh explore "Context"
  ./start_world0.sh connect "Context" "Projection" -- --type supports
  ./start_world0.sh viz -- --output world0.html
  ./start_world0.sh service-start --provider none
  ./start_world0.sh service-status
  ./start_world0.sh service-logs -f
  ./start_world0.sh service-stop
  ./start_world0.sh space list
  ./start_world0.sh space create research
  ./start_world0.sh test -- -k projection         # pytest -k projection
EOF
}

# ── Arg parsing ──────────────────────────────────────────────────────
# First positional (if it matches a known mode) becomes MODE.
MODE_SET=0
REMAINING=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --) shift; PASSTHROUGH+=("$@"); break ;;
    cli|web|gui|setup|test|status|reflect|ask|learn|explore|search|connect|web-search|viz|service-start|service-stop|service-restart|service-status|service-logs|space|shell)
      if [[ $MODE_SET -eq 0 ]]; then
        MODE="$1"; MODE_SET=1
      else
        REMAINING+=("$1")
      fi
      shift ;;
    --mode)       MODE="${2:?}"; MODE_SET=1; shift 2 ;;
    --provider)   PROVIDER="${2:?}"; shift 2 ;;
    --model)      MODEL="${2:?}"; shift 2 ;;
    --store)      STORE="${2:?}"; shift 2 ;;
    --space)      SPACE="${2:?}"; shift 2 ;;
    --host)       HOST="${2:?}"; shift 2 ;;
    --port)       PORT="${2:?}"; shift 2 ;;
    --python)     PYTHON_BIN="${2:?}"; shift 2 ;;
    --venv)       VENV_DIR="${2:?}"; shift 2 ;;
    --service-dir) SERVICE_DIR="${2:?}"; shift 2 ;;
    --no-venv)    NO_VENV=1; shift ;;
    --extras)     EXTRAS="${2:?}"; shift 2 ;;
    --verbose)    VERBOSE=1; shift ;;
    -h|--help)    print_usage; exit 0 ;;
    *)            REMAINING+=("$1"); shift ;;
  esac
done
# Any un-claimed positionals (e.g. the query for `ask`, extra `space` args)
# go into the forwarded list after the mode itself consumes what it needs.

case "$PROVIDER" in
  anthropic|openai|azure-openai|none) ;;
  *) die "Invalid provider: $PROVIDER" ;;
esac

# ── Python selection / venv bootstrap ────────────────────────────────
_probe_python() {
  # Echo a candidate's path only if it actually runs as Python 3.10+.
  local cand="$1" path ver
  path="$(command -v "$cand" 2>/dev/null || true)"
  if [[ -z "$path" ]]; then return 1; fi
  ver="$("$path" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
  if [[ -z "$ver" ]]; then return 1; fi
  local major="${ver%%.*}" minor="${ver##*.}"
  if [[ "$major" != "3" || "$minor" -lt 10 ]]; then return 1; fi
  echo "$path"
}

find_system_python() {
  for cand in python3 python3.12 python3.11 python3.10 python; do
    p="$(_probe_python "$cand" || true)"
    if [[ -n "$p" ]]; then echo "$p"; return 0; fi
  done
  return 1
}

resolve_python() {
  if [[ -n "$PYTHON_BIN" ]]; then return; fi

  if [[ $NO_VENV -eq 0 && -x "$VENV_DIR/bin/python" ]]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
    return
  fi

  PYTHON_BIN="$(find_system_python || true)"
  if [[ -z "$PYTHON_BIN" ]]; then die "No working Python 3.10+ interpreter found."; fi
}

ensure_venv() {
  if [[ $NO_VENV -eq 1 ]]; then return; fi
  if [[ -x "$VENV_DIR/bin/python" ]]; then return; fi

  local sys_python
  sys_python="$(find_system_python || true)"
  if [[ -z "$sys_python" ]]; then die "Python 3.10+ is required to create a venv."; fi

  log "Creating venv at $VENV_DIR"
  "$sys_python" -m venv "$VENV_DIR"
  PYTHON_BIN="$VENV_DIR/bin/python"
}

do_setup() {
  ensure_venv
  resolve_python
  log "Upgrading pip"
  "$PYTHON_BIN" -m pip install --upgrade pip >/dev/null
  log "Installing world0[$EXTRAS] (editable)"
  "$PYTHON_BIN" -m pip install -e ".[${EXTRAS}]"
  log "Setup complete. Next: ./start_world0.sh"
}

# ── Build the python command for run-modes ───────────────────────────
build_cli_cmd() {
  CMD=(
    "$PYTHON_BIN" -m world0.agents.cli
    --store "$STORE"
    --provider "$PROVIDER"
  )
  if [[ -n "$MODEL" ]]; then CMD+=(--model "$MODEL"); fi
  if [[ -n "$SPACE" ]]; then CMD+=(--space "$SPACE"); fi
}

build_web_cmd() {
  CMD=(
    "$PYTHON_BIN" -m world0.agents.web_main
    --store "$STORE"
    --provider "$PROVIDER"
    --host "$HOST"
    --port "$PORT"
  )
  if [[ -n "$MODEL" ]]; then CMD+=(--model "$MODEL"); fi
  if [[ -n "$SPACE" ]]; then CMD+=(--space "$SPACE"); fi
}

SERVICE_PID_FILE="$SERVICE_DIR/world0-web.pid"
SERVICE_LOG_FILE="$SERVICE_DIR/world0-web.log"
SERVICE_ENV_FILE="$SERVICE_DIR/world0-web.env"

service_pid() {
  if [[ -f "$SERVICE_PID_FILE" ]]; then
    tr -d '[:space:]' < "$SERVICE_PID_FILE"
  fi
}

is_pid_running() {
  local pid="${1:-}"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

print_service_status() {
  local pid status_host status_port
  pid="$(service_pid)"
  status_host="$HOST"
  status_port="$PORT"
  if [[ -f "$SERVICE_ENV_FILE" ]]; then
    status_host="$(sed -n 's/^WORLD0_HOST=//p' "$SERVICE_ENV_FILE" | tail -1)"
    status_port="$(sed -n 's/^WORLD0_PORT=//p' "$SERVICE_ENV_FILE" | tail -1)"
    status_host="${status_host:-$HOST}"
    status_port="${status_port:-$PORT}"
  fi
  if is_pid_running "$pid"; then
    log "Resident Web GUI service is running"
    echo "pid:  $pid"
    echo "url:  http://$status_host:$status_port"
    echo "log:  $SERVICE_LOG_FILE"
    echo "env:  $SERVICE_ENV_FILE"
    return 0
  fi

  if [[ -n "$pid" ]]; then
    warn "Resident Web GUI service is not running (stale pid: $pid)"
  else
    warn "Resident Web GUI service is not running"
  fi
  echo "pid:  $SERVICE_PID_FILE"
  echo "log:  $SERVICE_LOG_FILE"
  return 1
}

wait_for_service() {
  local check_host="$HOST"
  if [[ "$check_host" == "0.0.0.0" || "$check_host" == "::" ]]; then
    check_host="127.0.0.1"
  fi
  "$PYTHON_BIN" - "$check_host" "$PORT" <<'PY'
import http.client
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
deadline = time.monotonic() + 10

while time.monotonic() < deadline:
    try:
        conn = http.client.HTTPConnection(host, port, timeout=0.5)
        conn.request("GET", "/api/agent/status")
        response = conn.getresponse()
        response.read()
        if response.status < 500:
            sys.exit(0)
    except OSError:
        time.sleep(0.2)
    finally:
        try:
            conn.close()
        except Exception:
            pass

sys.exit(1)
PY
}

start_service() {
  local pid
  mkdir -p "$SERVICE_DIR"
  pid="$(service_pid)"
  if is_pid_running "$pid"; then
    die "Resident Web GUI service is already running (pid: $pid)."
  fi

  build_web_cmd
  CMD+=(--no-open "${PASS[@]+"${PASS[@]}"}")

  {
    echo "WORLD0_ROOT=$ROOT_DIR"
    echo "WORLD0_STORE=$STORE"
    echo "WORLD0_PROVIDER=$PROVIDER"
    echo "WORLD0_MODEL=$MODEL"
    echo "WORLD0_SPACE=$SPACE"
    echo "WORLD0_HOST=$HOST"
    echo "WORLD0_PORT=$PORT"
    echo "WORLD0_PYTHON=$PYTHON_BIN"
    echo "WORLD0_STARTED_AT=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } > "$SERVICE_ENV_FILE"

  log "Starting resident Web GUI service at http://$HOST:$PORT"
  pid="$("$PYTHON_BIN" - "$SERVICE_PID_FILE" "$SERVICE_LOG_FILE" "${CMD[@]}" <<'PY'
import os
import subprocess
import sys

pid_file = sys.argv[1]
log_file = sys.argv[2]
cmd = sys.argv[3:]

os.makedirs(os.path.dirname(pid_file), exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

log = open(log_file, "ab", buffering=0)
proc = subprocess.Popen(
    cmd,
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    cwd=os.getcwd(),
    env=os.environ.copy(),
    close_fds=True,
    start_new_session=True,
)

with open(pid_file, "w", encoding="utf-8") as handle:
    handle.write(str(proc.pid))
    handle.write("\n")

print(proc.pid)
PY
)"

  if wait_for_service; then
    log "Resident Web GUI service started (pid: $pid)"
    echo "url: http://$HOST:$PORT"
    echo "log: $SERVICE_LOG_FILE"
    return 0
  fi

  warn "Service did not become ready within 10 seconds."
  if ! is_pid_running "$pid"; then
    warn "Process exited. Recent logs:"
    tail -40 "$SERVICE_LOG_FILE" 2>/dev/null || true
    exit 1
  fi
  warn "Process is still running; inspect logs with: ./start_world0.sh service-logs"
}

stop_service() {
  local pid elapsed
  pid="$(service_pid)"
  if ! is_pid_running "$pid"; then
    warn "Resident Web GUI service is not running."
    rm -f "$SERVICE_PID_FILE"
    return 0
  fi

  log "Stopping resident Web GUI service (pid: $pid)"
  kill "$pid"
  elapsed=0
  while is_pid_running "$pid" && [[ "$elapsed" -lt 10 ]]; do
    sleep 1
    elapsed=$((elapsed + 1))
  done

  if is_pid_running "$pid"; then
    warn "Service did not stop after 10 seconds; forcing termination."
    kill -9 "$pid" 2>/dev/null || true
  fi

  rm -f "$SERVICE_PID_FILE"
  log "Resident Web GUI service stopped."
}

show_service_logs() {
  mkdir -p "$SERVICE_DIR"
  if [[ ! -f "$SERVICE_LOG_FILE" ]]; then
    warn "No service log exists yet: $SERVICE_LOG_FILE"
    return 0
  fi
  if [[ "${REM[0]:-}" == "-f" || "${PASS[0]:-}" == "-f" ]]; then
    tail -f "$SERVICE_LOG_FILE"
  else
    tail -80 "$SERVICE_LOG_FILE"
  fi
}

# ── Dispatch ─────────────────────────────────────────────────────────
if [[ "$MODE" == "setup" ]]; then
  do_setup
  exit 0
fi

REM=("${REMAINING[@]+"${REMAINING[@]}"}")
PASS=("${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}")

case "$MODE" in
  service-stop)
    stop_service
    exit 0
    ;;
  service-status)
    print_service_status
    exit 0
    ;;
  service-logs)
    show_service_logs
    exit 0
    ;;
esac

# For every other mode we need a Python interpreter.
if [[ $NO_VENV -eq 0 && ! -x "$VENV_DIR/bin/python" && -z "$PYTHON_BIN" ]]; then
  warn "No venv found at $VENV_DIR — running ./start_world0.sh setup first is recommended."
fi
resolve_python

# Make sure the src/ layout is importable even without `pip install -e .`
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

run_and_exec() {
  if [[ $VERBOSE -eq 1 ]]; then
    log "exec: ${CMD[*]}"
  fi
  exec "${CMD[@]}"
}

case "$MODE" in
  cli)
    build_cli_cmd
    CMD+=("${REM[@]+"${REM[@]}"}" "${PASS[@]+"${PASS[@]}"}")
    log "Starting CLI (provider=$PROVIDER, store=$STORE${SPACE:+, space=$SPACE})"
    run_and_exec
    ;;

  web)
    build_web_cmd
    CMD+=("${PASS[@]+"${PASS[@]}"}")
    log "Starting Web UI at http://$HOST:$PORT  (provider=$PROVIDER, store=$STORE${SPACE:+, space=$SPACE})"
    run_and_exec
    ;;

  gui)
    CMD=(
      "$PYTHON_BIN" -m world0.agents.gui
      --store "$STORE"
      --provider "$PROVIDER"
      --port "$PORT"
    )
    if [[ -n "$MODEL" ]]; then CMD+=(--model "$MODEL"); fi
    if [[ -n "$SPACE" ]]; then CMD+=(--space "$SPACE"); fi
    CMD+=("${PASS[@]+"${PASS[@]}"}")
    log "Starting GUI (port=$PORT, provider=$PROVIDER, store=$STORE${SPACE:+, space=$SPACE})"
    run_and_exec
    ;;

  test)
    CMD=("$PYTHON_BIN" -m pytest "${REM[@]+"${REM[@]}"}" "${PASS[@]+"${PASS[@]}"}")
    log "Running tests"
    run_and_exec
    ;;

  status|reflect)
    build_cli_cmd
    CMD+=("$MODE" "${REM[@]+"${REM[@]}"}" "${PASS[@]+"${PASS[@]}"}")
    run_and_exec
    ;;

  ask|learn|explore|search|web-search)
    if [[ ${#REM[@]} -eq 0 ]]; then
      die "$MODE requires a text argument. Example: ./start_world0.sh $MODE \"your text\""
    fi
    build_cli_cmd
    CMD+=("$MODE" "${REM[@]}" "${PASS[@]+"${PASS[@]}"}")
    run_and_exec
    ;;

  connect)
    if [[ ${#REM[@]} -lt 2 ]]; then
      die "connect requires source and target concepts. Example: ./start_world0.sh connect \"Context\" \"Projection\" -- --type supports"
    fi
    build_cli_cmd
    CMD+=(connect "${REM[@]}" "${PASS[@]+"${PASS[@]}"}")
    run_and_exec
    ;;

  viz)
    build_cli_cmd
    CMD+=(viz "${REM[@]+"${REM[@]}"}" "${PASS[@]+"${PASS[@]}"}")
    run_and_exec
    ;;

  service-start)
    start_service
    ;;

  service-restart)
    stop_service
    start_service
    ;;

  space)
    build_cli_cmd
    CMD+=(space "${REM[@]+"${REM[@]}"}" "${PASS[@]+"${PASS[@]}"}")
    run_and_exec
    ;;

  shell)
    log "Entering subshell with PYTHONPATH=$PYTHONPATH and PYTHON=$PYTHON_BIN"
    exec "${SHELL:-/bin/bash}"
    ;;

  *)
    die "Unknown mode: $MODE"
    ;;
esac
