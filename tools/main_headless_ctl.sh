#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/pi/project-final"
PID_FILE="$PROJECT_DIR/run_main.pid"
LOG_FILE="$PROJECT_DIR/run_main_headless.log"
VENV_ACTIVATE="$PROJECT_DIR/.venv/bin/activate"
DISPLAY_WIDTH="${DISPLAY_WIDTH:-480}"
DISPLAY_HEIGHT="${DISPLAY_HEIGHT:-270}"
PROCESS_PATTERN="[p]ython(3)?( .*)? main.py"

CMD=(
  python -u main.py
  --camera pi
  --model models/lstm_pi5_ready_fp16.tflite
  --model-threads 4
  --labels Fall,No_Fall,Pre-Fall,Falling
  --thresholds-json work_csv/compare/recommended_thresholds_pi5_ready.json
  --display-width "$DISPLAY_WIDTH"
  --display-height "$DISPLAY_HEIGHT"
)

detect_gui_display() {
  if [[ -n "${DISPLAY:-}" ]]; then
    echo "$DISPLAY"
    return 0
  fi

  # Prefer xrdp/remote desktop session when present.
  if [[ -S /tmp/.X11-unix/X10 ]]; then
    echo ":10"
    return 0
  fi

  # Fallback to the highest available X display socket.
  local sock max_num=-1 num
  for sock in /tmp/.X11-unix/X*; do
    [[ -e "$sock" ]] || continue
    num="${sock##*/X}"
    if [[ "$num" =~ ^[0-9]+$ ]] && (( num > max_num )); then
      max_num="$num"
    fi
  done

  if (( max_num >= 0 )); then
    echo ":$max_num"
  else
    echo ":0"
  fi
}

usage() {
  cat <<'EOF'
Usage: tools/main_headless_ctl.sh <command>

Commands:
  start    Stop old process and start headless main.py
  start-gui Stop old process and start main.py on detected GUI DISPLAY
  stop     Stop running main.py process
  status   Show running status and process command
  verify   Show last 80 log lines and quick health checks
  monitor  Tail log in real time
  restart  Stop then start
EOF
}

is_running() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

stop_main() {
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" || true
    # Give process a moment to cleanly release camera/window resources.
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" || true
    fi
  fi

  if pgrep -af "$PROCESS_PATTERN" >/dev/null 2>&1; then
    pkill -f "$PROCESS_PATTERN" || true
    sleep 1
  fi
  if pgrep -af "$PROCESS_PATTERN" >/dev/null 2>&1; then
    pkill -9 -f "$PROCESS_PATTERN" || true
    sleep 1
  fi
  rm -f "$PID_FILE"

  if pgrep -af "$PROCESS_PATTERN" >/dev/null 2>&1; then
    echo "ERROR: still running"
    pgrep -af "$PROCESS_PATTERN" || true
    return 1
  fi
  echo "STOPPED"
}

start_main() {
  stop_main
  cd "$PROJECT_DIR"
  # shellcheck source=/dev/null
  source "$VENV_ACTIVATE"
  nohup env -u DISPLAY "${CMD[@]}" >"$LOG_FILE" 2>&1 < /dev/null &
  echo "$!" > "$PID_FILE"
  sleep 2
  if ! is_running; then
    echo "ERROR: main.py failed to start"
    [[ -f "$LOG_FILE" ]] && tail -n 80 "$LOG_FILE"
    exit 1
  fi
  echo "STARTED PID $(cat "$PID_FILE")"
}

start_gui_main() {
  stop_main
  cd "$PROJECT_DIR"
  # shellcheck source=/dev/null
  source "$VENV_ACTIVATE"
  local display_value
  display_value="$(detect_gui_display)"
  nohup env DISPLAY="$display_value" XAUTHORITY=/home/pi/.Xauthority "${CMD[@]}" >"$LOG_FILE" 2>&1 < /dev/null &
  echo "$!" > "$PID_FILE"
  sleep 2
  if ! is_running; then
    echo "ERROR: main.py failed to start (gui mode)"
    [[ -f "$LOG_FILE" ]] && tail -n 80 "$LOG_FILE"
    exit 1
  fi
  echo "STARTED GUI PID $(cat "$PID_FILE") DISPLAY=$display_value"
}

status_main() {
  if is_running; then
    local pid
    pid="$(cat "$PID_FILE")"
    echo "RUNNING PID $pid"
    ps -fp "$pid"
  else
    echo "STOPPED"
  fi
}

verify_main() {
  cd "$PROJECT_DIR"
  [[ -f "$LOG_FILE" ]] && tail -n 80 "$LOG_FILE" || echo "No log file yet"
  echo "-----"
  if grep -q "Using camera mode: pi" "$LOG_FILE" 2>/dev/null; then
    echo "OK: camera mode initialized"
  else
    echo "WARN: camera init line not found yet"
  fi
  if grep -q "Headless mode: OpenCV window disabled" "$LOG_FILE" 2>/dev/null; then
    echo "OK: headless mode active"
  else
    echo "WARN: headless marker not found yet"
  fi
  if grep -q "Cannot open camera\\|used by another process" "$LOG_FILE" 2>/dev/null; then
    echo "FAIL: camera collision detected"
    exit 2
  else
    echo "OK: no camera collision message"
  fi
}

monitor_main() {
  cd "$PROJECT_DIR"
  touch "$LOG_FILE"
  tail -f "$LOG_FILE"
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    start) start_main ;;
    start-gui) start_gui_main ;;
    stop) stop_main ;;
    status) status_main ;;
    verify) verify_main ;;
    monitor) monitor_main ;;
    restart) stop_main; start_main ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
