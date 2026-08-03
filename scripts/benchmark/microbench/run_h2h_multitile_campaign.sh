#!/bin/bash
# netlist_multi_tile cross-check of the §7.16 NN/BDD campaign (2026-08-03).
# Runs the 100-step protocol (cold DC @1e-8 + 100-step BE dt=5ps) for the
# champion base and the NN/BJ candidate family on the tiny 9-tile PDN, then
# the full tile-block eigen census (--sample 9 = census, no extrapolation).
# Every run is attached to the campaign watchdog per the standing constraint.
set -u
cd /home/exx/workspace/sdvd/sigma_dvd_prototype
PKL=netlist/netlist_multi_tile/distributed_pkl
MB=scripts/benchmark/microbench
PY=venv/bin/python
WD=$MB/mem_watchdog_attach.sh

run_one() {
  local name=$1; shift
  local runlog=logs/h2h_multitile_${name}.log
  local memlog=$MB/h2h_multitile_${name}.memlog
  echo "=== [$name] $* ==="
  setsid $PY -u "$@" > "$runlog" 2>&1 &
  local pid=$!
  bash "$WD" "$pid" "$memlog" &
  local wpid=$!
  wait "$pid"
  local rc=$?
  kill "$wpid" 2>/dev/null
  wait "$wpid" 2>/dev/null
  echo "=== [$name] rc=$rc ==="
  tail -n 4 "$runlog"
}

H2H=$MB/run_neumann_h2h_mi200k.py

run_one champion        $H2H $PKL --two-level-base jacobi \
    --json $MB/results_champion_100step_multitile.json
run_one neumann_reg0    $H2H $PKL --two-level-base neumann --neumann-reg 0 \
    --json $MB/results_neumann_reg0_multitile.json
run_one neumann_reg1em3 $H2H $PKL --two-level-base neumann --neumann-reg 1e-3 \
    --json $MB/results_neumann_reg1em3_multitile.json
run_one neumann_reg1em5 $H2H $PKL --two-level-base neumann --neumann-reg 1e-5 \
    --json $MB/results_neumann_reg1em5_multitile.json
run_one bj_neverasm     $H2H $PKL --two-level-base block_jacobi \
    --json $MB/results_bj_neverassemble_multitile.json

run_one spectra $MB/probe_tile_block_spectra_mi200k.py $PKL --sample 9 \
    --json $MB/results_tile_block_spectra_multitile.json

echo "campaign done"
