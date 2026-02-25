#!/usr/bin/env bash
set -euo pipefail

LABELS_DEFAULT="Fall,No_Fall,Pre-Fall,Falling"
THREADS_DEFAULT="4"
BOARD_DIR_DEFAULT="~/project-final"

usage() {
  cat <<'EOF'
Usage:
  tools/deploy_tflite_to_board.sh \
    --board-user <user> \
    --board-host <host> \
    --model-local <path/to/model.tflite> \
    --thresholds-local <path/to/recommended_thresholds.json> \
    [--board-dir ~/project-final] \
    [--labels Fall,No_Fall,Pre-Fall,Falling] \
    [--model-threads 4] \
    [--skip-install] \
    [--allow-precheck-warn]

What it does:
  1) scp model + thresholds to board
  2) install dependencies on board (unless --skip-install)
  3) run predeploy check on board
  4) run main.py with camera=pi (if precheck passes)

Precheck policy:
  - exit 0: proceed
  - exit 1: stop by default; proceed only with --allow-precheck-warn
  - exit 2: stop
EOF
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

BOARD_USER=""
BOARD_HOST=""
BOARD_DIR="$BOARD_DIR_DEFAULT"
MODEL_LOCAL=""
THRESHOLDS_LOCAL=""
LABELS="$LABELS_DEFAULT"
MODEL_THREADS="$THREADS_DEFAULT"
SKIP_INSTALL="0"
ALLOW_PRECHECK_WARN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --board-user)
      BOARD_USER="${2:-}"; shift 2 ;;
    --board-host)
      BOARD_HOST="${2:-}"; shift 2 ;;
    --board-dir)
      BOARD_DIR="${2:-}"; shift 2 ;;
    --model-local)
      MODEL_LOCAL="${2:-}"; shift 2 ;;
    --thresholds-local)
      THRESHOLDS_LOCAL="${2:-}"; shift 2 ;;
    --labels)
      LABELS="${2:-}"; shift 2 ;;
    --model-threads)
      MODEL_THREADS="${2:-}"; shift 2 ;;
    --skip-install)
      SKIP_INSTALL="1"; shift 1 ;;
    --allow-precheck-warn)
      ALLOW_PRECHECK_WARN="1"; shift 1 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      die "Unknown argument: $1" ;;
  esac
done

[[ -n "$BOARD_USER" ]] || die "--board-user is required"
[[ -n "$BOARD_HOST" ]] || die "--board-host is required"
[[ -n "$MODEL_LOCAL" ]] || die "--model-local is required"
[[ -n "$THRESHOLDS_LOCAL" ]] || die "--thresholds-local is required"

require_cmd ssh
require_cmd scp
require_cmd basename

[[ -f "$MODEL_LOCAL" ]] || die "Model file not found: $MODEL_LOCAL"
[[ -f "$THRESHOLDS_LOCAL" ]] || die "Thresholds file not found: $THRESHOLDS_LOCAL"

MODEL_FILE="$(basename "$MODEL_LOCAL")"
THRESH_FILE="$(basename "$THRESHOLDS_LOCAL")"
REMOTE="$BOARD_USER@$BOARD_HOST"

echo "[INFO] Target board: $REMOTE"
echo "[INFO] Board dir: $BOARD_DIR"
echo "[INFO] Model: $MODEL_LOCAL -> $BOARD_DIR/models/$MODEL_FILE"
echo "[INFO] Thresholds: $THRESHOLDS_LOCAL -> $BOARD_DIR/work_csv/compare/$THRESH_FILE"

echo "[STEP 1/4] Prepare directories on board..."
ssh "$REMOTE" "mkdir -p \"$BOARD_DIR/models\" \"$BOARD_DIR/work_csv/compare\""

echo "[STEP 1/4] Upload files via scp..."
scp "$MODEL_LOCAL" "$REMOTE:$BOARD_DIR/models/$MODEL_FILE"
scp "$THRESHOLDS_LOCAL" "$REMOTE:$BOARD_DIR/work_csv/compare/$THRESH_FILE"

if [[ "$SKIP_INSTALL" != "1" ]]; then
  echo "[STEP 2/4] Install runtime dependencies on board..."
  ssh "$REMOTE" "cd \"$BOARD_DIR\" && \
    python3 -m venv .venv && \
    . .venv/bin/activate && \
    python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt && \
    python -m pip install tflite-runtime"
else
  echo "[STEP 2/4] Skip install requested (--skip-install)."
fi

echo "[STEP 3/4] Run predeploy check on board..."
set +e
PRECHECK_OUTPUT="$(ssh "$REMOTE" "cd \"$BOARD_DIR\" && . .venv/bin/activate && python tools/predeploy_board_check.py --model \"models/$MODEL_FILE\" --labels \"$LABELS\" --thresholds-json \"work_csv/compare/$THRESH_FILE\" --check-imports" 2>&1)"
PRECHECK_RC=$?
set -e
echo "$PRECHECK_OUTPUT"
echo "[INFO] precheck exit code: $PRECHECK_RC"

if [[ "$PRECHECK_RC" -eq 2 ]]; then
  die "Precheck failed (exit 2). Fix issues before deploy."
fi

if [[ "$PRECHECK_RC" -eq 1 && "$ALLOW_PRECHECK_WARN" != "1" ]]; then
  die "Precheck returned warning (exit 1). Re-run with --allow-precheck-warn to continue."
fi

echo "[STEP 4/4] Start runtime on board (press Ctrl+C to stop)..."
ssh -t "$REMOTE" \
  "cd \"$BOARD_DIR\" && . .venv/bin/activate && python main.py --camera pi --model \"models/$MODEL_FILE\" --model-threads \"$MODEL_THREADS\" --labels \"$LABELS\" --thresholds-json \"work_csv/compare/$THRESH_FILE\""
