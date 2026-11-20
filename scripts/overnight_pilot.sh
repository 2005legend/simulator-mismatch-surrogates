#!/usr/bin/env bash
# Overnight pilot (seeds 0, 101, 102), then RQ1 on the pilot seeds, then shut the laptop down.
#   bash scripts/overnight_pilot.sh            # run, then shut down
#   NO_SHUTDOWN=1 bash scripts/overnight_pilot.sh   # run only
# The shutdown waits 5 minutes; cancel it with:  shutdown.exe -a
# Everything is resumable: rerunning skips finished stages and cells.
cd "$(dirname "$0")/.." || exit 1
LOG=runs/pilot/logs
mkdir -p "$LOG"
export PYTHONUNBUFFERED=1
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG/status.txt"; }

note "start"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/keep_awake.ps1 &
AWAKE=$!
python -m scripts.run_pilot --stage all --out-dir runs/pilot > "$LOG/pilot.out" 2> "$LOG/pilot.err"
note "pilot stages exit code $?"

if [ -f runs/pilot/frozen_params.json ]; then
  pids=()
  for s in 0 101 102; do
    python -m scripts.run_rq1 --phase pilot --seeds "$s" --n-nom 512 \
      --params runs/pilot/frozen_params.json --out-dir runs/rq1_pilot \
      > "$LOG/rq1_seed$s.out" 2> "$LOG/rq1_seed$s.err" &
    pids+=($!)
  done
  for i in 0 1 2; do
    wait "${pids[$i]}"; note "rq1 pilot seed index $i exit code $?"
  done
else
  note "no frozen_params.json: RQ1 pilot skipped (see $LOG/pilot.err)"
fi

note "all done"
kill "$AWAKE" 2>/dev/null
if [ "${NO_SHUTDOWN:-0}" != "1" ]; then
  note "shutting down in 5 minutes (cancel: shutdown.exe -a)"
  shutdown.exe -s -t 300 -c "Pilot finished; shutting down. Cancel with: shutdown -a"
fi
