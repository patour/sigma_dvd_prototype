#!/bin/bash
# Attachable memory watchdog for benchmark runs (campaign standard).
# Usage: mem_watchdog_attach.sh <driver_pid> <memlog>
# Same guard philosophy as run_bj_true_block_watchdog.sh but attaches to an
# already-running driver instead of launching it. Kills the driver (+ its
# children + `ray stop --force`) BEFORE the kernel OOM killer fires or the
# host starts swapping:
#   - system used (MemTotal - MemAvailable) > LIMIT_USED_GB   (default 225)
#   - MemAvailable < MIN_AVAIL_GB                             (default 10)
#   - swap-in-use grows > SWAP_LIMIT_GB above attach-time use (default 2)
set -u
PID=${1:?driver pid}
MEMLOG=${2:?memlog path}
LIMIT_USED_KB=$(( ${LIMIT_USED_GB:-225} * 1024 * 1024 ))
MIN_AVAIL_KB=$(( ${MIN_AVAIL_GB:-10} * 1024 * 1024 ))
SWAP_LIMIT_KB=$(( ${SWAP_LIMIT_GB:-2} * 1024 * 1024 ))
RAY_BIN=${RAY_BIN:-/home/exx/workspace/sdvd/sigma_dvd_prototype/venv/bin/ray}

swap_used0=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
echo "watchdog attach: pid=$PID limits: used>$((LIMIT_USED_KB/1048576))GB avail<$((MIN_AVAIL_KB/1048576))GB swap_growth>$((SWAP_LIMIT_KB/1048576))GB (swap at attach: $((swap_used0/1024))MB)" | tee "$MEMLOG"

kill_all() {
  echo "WATCHDOG: $1 -- killing driver $PID + children + ray" | tee -a "$MEMLOG" >&2
  pkill -TERM -P "$PID" 2>/dev/null
  kill -TERM "$PID" 2>/dev/null
  sleep 15
  pkill -KILL -P "$PID" 2>/dev/null
  kill -KILL "$PID" 2>/dev/null
  "$RAY_BIN" stop --force >/dev/null 2>&1
  exit 137
}

while kill -0 "$PID" 2>/dev/null; do
  read -r used_kb avail_kb swap_used_kb <<< "$(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}/SwapTotal/{st=$2}/SwapFree/{sf=$2}END{print t-a, a, st-sf}' /proc/meminfo)"
  rss_kb=$(awk '/VmRSS/{print $2}' "/proc/$PID/status" 2>/dev/null || echo 0)
  ray_rss_kb=$(ps -eo rss,comm 2>/dev/null | awk '/ray::|raylet/{s+=$1}END{print s+0}')
  echo "$(date +%H:%M:%S) used=$((used_kb/1048576))GB avail=$((avail_kb/1048576))GB swap=$((swap_used_kb/1024))MB driver=$((rss_kb/1048576))GB ray_workers=$((ray_rss_kb/1048576))GB" >> "$MEMLOG"
  [ "$used_kb" -gt "$LIMIT_USED_KB" ] && kill_all "system used $((used_kb/1048576))GB > $((LIMIT_USED_KB/1048576))GB"
  [ "$avail_kb" -lt "$MIN_AVAIL_KB" ] && kill_all "MemAvailable $((avail_kb/1048576))GB < $((MIN_AVAIL_KB/1048576))GB"
  [ $((swap_used_kb - swap_used0)) -gt "$SWAP_LIMIT_KB" ] && kill_all "swap grew $(((swap_used_kb - swap_used0)/1024))MB since attach"
  sleep 5
done
echo "watchdog: driver $PID exited cleanly at $(date +%H:%M:%S)" | tee -a "$MEMLOG"
