#!/usr/bin/env bash
# Completes the RQ1 grid of prereg-v1 section 6 (2 systems x 5 eps x 3 n_nom): the n_nom = 128 and
# 2048 cells that scripts/run_main.sh left out (it applied the RQ2 extra-cell rule of section 7 to
# RQ1 by mistake). Finished cells are skipped by the runner. Waits for run_main.sh to finish first
# when WAIT=1.
#   WAIT=1 WORKERS=6 bash scripts/run_rq1_nnom.sh
cd "$(dirname "$0")/.." || exit 1
export P=pilot_results/frozen_params.json LOG=runs/main_logs PYTHONUNBUFFERED=1
W=${WORKERS:-6}
if [ "${WAIT:-0}" = 1 ]; then
  until grep -q "all done" "$LOG/status.txt" 2>/dev/null; do sleep 60; done
fi
task() {
  s=$1
  python -m scripts.run_rq1 --phase main --seeds "$s" --n-nom 128 2048 --systems S1 S2 \
    --mults 0 0.25 0.5 1 2 --params "$P" --out-dir runs/rq1_main \
    > "$LOG/rq1nnom_seed$s.out" 2> "$LOG/rq1nnom_seed$s.err"
  echo "$(date '+%Y-%m-%d %H:%M:%S') rq1nnom seed $s exit $?" >> "$LOG/status.txt"
}
export -f task
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/keep_awake.ps1 &
AWAKE=$!
echo "$(date '+%Y-%m-%d %H:%M:%S') rq1nnom start, $W workers" >> "$LOG/status.txt"
seq 1 20 | xargs -P "$W" -n 1 bash -c 'task "$0"'
echo "$(date '+%Y-%m-%d %H:%M:%S') rq1nnom all done" >> "$LOG/status.txt"
kill "$AWAKE" 2>/dev/null
