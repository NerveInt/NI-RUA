#!/bin/bash
# Start the cuRobo IK service detached (setsid nohup); PID/STATUS/log files in $CUROBO_IK_RUN_DIR.
# Builds the robot model first when model/model.json is missing or was built with another joint-limit margin.
#
#   CUROBO_PYTHON=~/miniforge3/envs/RoboDojo/bin/python CUDA_VISIBLE_DEVICES=6 bash start.sh
# Environment (defaults in brackets):
#   CUROBO_PYTHON [python3]  interpreter with cuRobo v2, torch and CUDA
#   CUDA_VISIBLE_DEVICES [0]  GPU (NI-RUA's remote manager sets it from gpu / gpu_min_free_mib)
#   CUROBO_IK_HOST [127.0.0.1]  CUROBO_IK_PORT [8341]
#   SO101_JOINT_LIMIT_MARGIN_RAD [0.05]  must equal the deployment's motion.joint_limit_margin_rad
#   SO101_JAW_OPEN_RAD [1.74533]  the deployment's gripper.jaw_open_rad (rad); the startup self-check checks the
#                                 ready pose with the jaw there too
#   CUROBO_IK_SEEDS [64]  parallel IK seeds   CUROBO_IK_MAX_BATCH [32]  poses per request
#   CUROBO_IK_RUN_DIR [<this dir>/run]  START_TIMEOUT_S [300]
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${CUROBO_PYTHON:-python3}
R=${CUROBO_IK_RUN_DIR:-$HERE/run}
HOST=${CUROBO_IK_HOST:-127.0.0.1}; PORT=${CUROBO_IK_PORT:-8341}
MARGIN=${SO101_JOINT_LIMIT_MARGIN_RAD:-0.05}
OPEN_JAW=${SO101_JAW_OPEN_RAD:-1.74533}
WAIT=${START_TIMEOUT_S:-300}
GPU=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$R"
PIDF=$R/curobo_ik.$PORT.PID; STF=$R/curobo_ik.$PORT.STATUS; LOGF=$R/curobo_ik.$PORT.log
p=$(cat "$PIDF" 2>/dev/null)
if [ -n "$p" ] && kill -0 "$p" 2>/dev/null && grep -q curobo_ik_service.py "/proc/$p/cmdline" 2>/dev/null; then
  echo "curobo_ik already running pid=$p: $(cat "$STF")"; exit 0
fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$PORT\$"; then echo "curobo_ik: port $PORT busy"; exit 1; fi
if ! "$PY" - "$HERE/model/model.json" "$MARGIN" <<'PYEOF' 2>/dev/null
import json, sys
info = json.load(open(sys.argv[1]))
sys.exit(0 if abs(info["margin_rad"] - float(sys.argv[2])) < 1e-12 else 1)
PYEOF
then
  echo "building robot model (margin $MARGIN rad)..."
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$HERE/build_model.py" --source "$HERE/model_src" --out "$HERE/model" --margin "$MARGIN" >> "$LOGF" 2>&1 \
    || { echo "model build FAILED (see $LOGF)"; tail -20 "$LOGF"; exit 1; }
fi
echo "STARTING port=$PORT gpu=$GPU $(date '+%F %T %Z')" > "$STF"
( cd "$R" || exit 1
  export CUDA_VISIBLE_DEVICES=$GPU
  setsid nohup "$PY" -u "$HERE/curobo_ik_service.py" --model "$HERE/model" --host "$HOST" --port "$PORT" \
    --num-seeds "${CUROBO_IK_SEEDS:-64}" --max-batch "${CUROBO_IK_MAX_BATCH:-32}" --open-jaw-rad "$OPEN_JAW" \
    --status "$STF" >> "$LOGF" 2>&1 < /dev/null &
  echo $! > "$PIDF" )
for i in $(seq 1 $((WAIT / 2))); do
  sleep 2
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q '"status": "ok"'; then echo "curobo_ik up after $((2 * i)) s: $(cat "$STF")"; exit 0; fi
  kill -0 "$(cat "$PIDF")" 2>/dev/null || { echo "curobo_ik FAILED (see $LOGF)"; echo "FAILED $(date '+%F %T %Z')" > "$STF"; tail -20 "$LOGF"; exit 1; }
done
echo "curobo_ik not healthy after $WAIT s"; tail -5 "$LOGF"; exit 1
