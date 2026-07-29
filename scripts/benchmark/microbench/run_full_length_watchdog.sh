#!/bin/bash
# Full-length 2000-step winning-config run with a memory watchdog.
# Kills the run if MemAvailable drops below 15 GB (before the OOM killer does).
set -u
cd /home/exx/workspace/sdvd/sigma_dvd_prototype

venv/bin/python -u scripts/benchmark/microbench/run_full_length_winning_config.py \
  netlist/netlist_brcm_sampled/distributed_pkl_mi200k_v2 \
  --json scripts/benchmark/microbench/results_full_length_mi200k.json &
PID=$!

while kill -0 "$PID" 2>/dev/null; do
  avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if [ "$avail_kb" -lt 15728640 ]; then
    echo "WATCHDOG: MemAvailable ${avail_kb} KB < 15 GB -- killing run" >&2
    kill -TERM "$PID"
    sleep 15
    kill -KILL "$PID" 2>/dev/null
    venv/bin/ray stop --force >/dev/null 2>&1
    exit 137
  fi
  sleep 10
done

wait "$PID"
exit $?
