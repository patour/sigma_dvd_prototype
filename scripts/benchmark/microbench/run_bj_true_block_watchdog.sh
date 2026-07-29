#!/bin/bash
# Memory watchdog for the BJ true-block isolation experiment.
# KILLS the experiment (driver process group + Ray workers) if TOTAL system
# used memory (MemTotal - MemAvailable) exceeds 225 GB, or MemAvailable drops
# below 10 GB (secondary guard against other-tenant growth) -- before the
# host starts swapping or the kernel OOM killer picks a victim for us.
set -u
cd /home/exx/workspace/sdvd/sigma_dvd_prototype

LIMIT_USED_KB=$((225 * 1024 * 1024))   # 225 GB
MIN_AVAIL_KB=$((10 * 1024 * 1024))     # 10 GB
RUNLOG=${RUNLOG:-logs/bj_true_block_isolation_20260727.log}
MEMLOG=${MEMLOG:-scripts/benchmark/microbench/bj_true_block_isolation.memlog}

setsid venv/bin/python -u \
  scripts/benchmark/microbench/run_bj_true_block_isolation_mi200k.py \
  netlist/netlist_brcm_sampled/distributed_pkl_mi200k_v2 \
  --json scripts/benchmark/microbench/results_bj_true_block_isolation_mi200k.json \
  "$@" \
  > "$RUNLOG" 2>&1 &
PID=$!

echo "watchdog: driver pid=$PID limit=225GB used / 10GB avail; runlog=$RUNLOG" | tee "$MEMLOG"

kill_all() {
  echo "WATCHDOG: $1 -- killing experiment (pgid $PID + ray)" | tee -a "$MEMLOG" >&2
  kill -TERM -- -"$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null
  sleep 15
  kill -KILL -- -"$PID" 2>/dev/null
  kill -KILL "$PID" 2>/dev/null
  venv/bin/ray stop --force >/dev/null 2>&1
  exit 137
}

while kill -0 "$PID" 2>/dev/null; do
  read -r used_kb avail_kb <<< "$(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{print t-a, a}' /proc/meminfo)"
  rss_kb=$(awk '/VmRSS/{print $2}' "/proc/$PID/status" 2>/dev/null || echo 0)
  echo "$(date +%H:%M:%S) used=$((used_kb / 1048576))GB avail=$((avail_kb / 1048576))GB driver_rss=$((rss_kb / 1048576))GB" >> "$MEMLOG"
  if [ "$used_kb" -gt "$LIMIT_USED_KB" ]; then
    kill_all "system used $((used_kb / 1048576)) GB > 225 GB"
  fi
  if [ "$avail_kb" -lt "$MIN_AVAIL_KB" ]; then
    kill_all "MemAvailable $((avail_kb / 1048576)) GB < 10 GB"
  fi
  sleep 5
done

wait "$PID"
rc=$?
echo "watchdog: experiment exited rc=$rc" | tee -a "$MEMLOG"
exit $rc
