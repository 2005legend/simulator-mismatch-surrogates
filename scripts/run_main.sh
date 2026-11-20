#!/usr/bin/env bash
# Main runs (seeds 1-20), after the pred-v1 tag: RQ1, Family A, Family F.
# One task per (family, seed); WORKERS tasks run at once (default 6). Resumable: rerun to continue.
#   bash scripts/run_main.sh
#   WORKERS=8 bash scripts/run_main.sh
# RQ1 follows prereg-v1: both systems at n_nom = 512 (all levels, extra ones included), and the
# extra S1 cells at n_nom = 128 and 2048 at eps = 0 and at 1 x eps_ref (the level whose pilot r = 1).
cd "$(dirname "$0")/.." || exit 1
export P=pilot_results/frozen_params.json LOG=runs/main_logs PYTHONUNBUFFERED=1
W=${WORKERS:-6}
mkdir -p "$LOG"
task() {
  kind=$1; s=$2
  case $kind in
    rq1) python -m scripts.run_rq1 --phase main --seeds "$s" --n-nom 512 --params "$P" --out-dir runs/rq1_main &&
         python -m scripts.run_rq1 --phase main --seeds "$s" --n-nom 128 2048 --systems S1 --mults 0 1 \
           --params "$P" --out-dir runs/rq1_main ;;
    A) python -m scripts.run_acquisition --family A --phase main --seeds "$s" --params "$P" --out-dir runs/acq_main ;;
    F) python -m scripts.run_acquisition --family F --phase main --seeds "$s" --params "$P" --out-dir runs/acq_main ;;
  esac > "$LOG/${kind}_seed$s.out" 2> "$LOG/${kind}_seed$s.err"
  echo "$(date '+%Y-%m-%d %H:%M:%S') $kind seed $s exit $?" >> "$LOG/status.txt"
}
export -f task
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/keep_awake.ps1 &
AWAKE=$!
echo "$(date '+%Y-%m-%d %H:%M:%S') start, $W workers" >> "$LOG/status.txt"
{ for k in rq1 A F; do for s in $(seq 1 20); do echo "$k $s"; done; done; } |
  xargs -P "$W" -n 2 bash -c 'task "$0" "$1"'
echo "$(date '+%Y-%m-%d %H:%M:%S') all done" >> "$LOG/status.txt"
kill "$AWAKE" 2>/dev/null
