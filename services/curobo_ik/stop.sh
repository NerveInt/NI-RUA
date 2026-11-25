#!/bin/bash
# Stop the cuRobo IK service started by start.sh with the same CUROBO_IK_RUN_DIR / port (only its own PID).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=${CUROBO_IK_RUN_DIR:-$HERE/run}; PORT=${CUROBO_IK_PORT:-8341}
PIDF=$R/curobo_ik.$PORT.PID; p=$(cat "$PIDF" 2>/dev/null)
if [ -n "$p" ] && kill -0 "$p" 2>/dev/null && grep -q curobo_ik_service.py "/proc/$p/cmdline" 2>/dev/null; then
  kill "$p"; for i in $(seq 1 30); do kill -0 "$p" 2>/dev/null || break; sleep 0.5; done
  kill -0 "$p" 2>/dev/null && kill -9 "$p"
  echo "curobo_ik :$PORT stopped pid=$p"
else
  echo "curobo_ik :$PORT not running"
fi
rm -f "$PIDF"; echo "STOPPED $(date '+%F %T %Z')" > "$R/curobo_ik.$PORT.STATUS"
