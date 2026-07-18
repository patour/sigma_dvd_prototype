# Interface-Solve Acceleration: the Remaining Lever for the 10× BRCM Goal

> Supersedes the completed distributed-10x refactor plan that previously lived in this file (all phases 0/A/V/B/D landed, commits `51b4b65..f25d209` + follow-ups; history in `docs/brcm_distributed_runtime_optimization.md` §7 and git log).

## Context

B1 balanced retiling fixed the tile side of the BRCM transient loop (RHS+recovery 4.36 → 1.49 s/step at 107 tiles) but exploded the interface system: 70,734 → 190,867 unknowns, S = 1,282M nnz (14.3 GB, union of dense per-tile port blocks). Direct backsolve: 11.5 s/step (47.8 GB factor, bandwidth-bound). CG (block-Jacobi, rtol 1e-12, single-threaded assembled CSR matvec): ~180 warm iters × 1.75 s = 329 s/step. §7.4 conclusion: **no tiling reaches 10×; the interface solver must change.** Target: **≤0.3–0.5 s/step interface solve at the 107-tile split on the BRCM host** (dev-host equivalent ≈0.1–0.2 s/step; dev host measured ~3× faster on the identical supernodal factor).

Development testbed: `netlist/netlist_brcm_sampled` (8.28M-node contraction proxy, §7.6-validated: interface system **structurally identical** to BRCM — same 70,734 unknowns / 493.5M nnz at 36 tiles). Dev host: RTX 6000 Ada 48 GB (~960 GB/s), cupy not yet installed.

### Feasibility arithmetic (drives the design)

Per-matvec traffic at 190,867 unknowns: dense S_i blocks fp64 12.5 GB (fp32 6.2 GB), assembled CSR 15.4 GB, block-Jacobi apply ~2.7 GB. Floors: dev CPU ≈45–60 ms/matvec fp64, BRCM host ≈130–190 ms, GPU ≈13–16 ms fp64 / 7–8 ms fp32.
- CPU-threaded path alone lands ~1.1–1.3 s/step on dev at ~15 iters — a ~10× win over direct-at-107-tiles but **misses the target**; it is the mandatory-everywhere fallback, not the target vehicle.
- GPU whole-PCG path hits the dev target (8–15 iters × ~21 ms ≈ 0.15–0.3 s).
- Iteration budget: 180 warm iters measured (bj, 1e-12) → rtol 1e-8 ≈ 100–130 → + coarse space target ≤30 (stretch 10–15).
- Worker-side matvec adds **zero** aggregate bandwidth on one box — Stage 5 pays off only multi-node.

### User decisions (binding)

- **Accuracy:** production default `interface_cg_rtol` = **1e-8** (≤1 µV on ~105 mV signals); override flag retained; rtol sweep study validates before the default changes.
- **GPU (RESOLVED 2026-07-18):** BRCM host is **CPU-only** — CPU threaded path is the critical path, **fp32 tilewise matvec promoted to critical path** (Stage 2), Stage 3 coarse-space iteration cut is make-or-break. GPU (Stage 4) remains an *optional* backend (cupy, lazy import, CPU fallback mandatory) — validated on the dev GPU, deployable only where a GPU exists.
- **Multi-node (updated 2026-07-18):** undecided for BRCM production → Stage 5 stays gated; re-assess after Stage 3 lands the measured CPU-only s/step.
- **Scope:** core CG acceleration + worker-side/s-step CG + island-detection persistence. Decompose multi-RHS lockstep OUT of scope.
- **Process:** Stages 1–4 implemented via Workflow orchestration — coding agents: **Sonnet 5 (`claude-sonnet-5`), 1M context, effort xhigh**; reviewers: **Opus 4.8 (`claude-opus-4-8`), 1M context, effort xhigh**; spec-compliance review THEN code-quality review by two distinct reviewer agents, fix loop until both clean. No senior-engineer/principal-code-reviewer custom agents. **After each stage** (workflow clean + gates green): run `/code-review xhigh --fix` on the stage's changes, apply surviving findings, fold fixes in before pushing (user-mandated 2026-07-18).

### Current-state anchors (verified)

- `src/distributed/interface_iterative.py`: `InterfaceCGSolver` (:161-239) — `matvec_mode='assembled'|'tilewise'`; tilewise (:256-287) is a **serial** Python loop over dense S_i + `np.bincount` scatter; warm start already internal (`self._x0` cached after each solve, :625); block-Jacobi (:324-482) with serial apply (:469-478) and 4 GB silent-fallback cap (:119); `AUTO_CG_FACTOR_MEMORY_BUDGET_BYTES` 32 GB hardcoded (:99); `build_interface_solver` factory (:724-814) is dead code — both factor paths construct inline.
- `src/distributed/result_factorization.py`: DC CG wiring :1102-1199, transient :1742-1825; atol/maxiter/strict not plumbed; tilewise requires non-streaming S_i gather (falls back to assembled :1144-1160); `refactor()` always downgrades tilewise→assembled (:1441-1456, :2082-2097); DC island detection un-persisted (238 s proxy / 900 s BRCM per run).
- `src/distributed/solver_td.py`: per-step solve `trans_ctx.interface_lu(global_rhs)` :845 (QS :405); "Assemble + solve" timer bundles RHS finalization with solve (:827-846); no CG-iteration logging in `loop_stats`.
- `src/distributed/tile_worker.py`: S_i dense `(n_p,n_p)` float64; freed after ship/stream; **no apply-to-vector RPC**.
- Contexts hold `_tile_index_maps` (P_i int32 maps), `_S_global`, `_cg_solver` (result.py:492-500).
- CLI :656-697 (`--interface-solver/-matvec-mode/-preconditioner/-cg-rtol`), YAML :1195-1208.
- Tests: `tests/distributed/test_interface_iterative.py` (1,326 lines; two-tile fixture + sampled markers). BRCM CG log confirms genuine block-Jacobi at 190K (3,486 MB < 4 GB cap).

### Memory decision

Keep both S_global CSR and dense S_i by default in tilewise CG (~30.6 GB total at 190K vs 63 GB for direct — strictly better). Opt-in `interface_drop_s_global` frees `ctx._S_global` after extracting BJ blocks/diag/S_extra/S_c (`save()` then raises with guidance). Not default — breaks save/refactor for ~15 GB.

### Confirmed pre-existing defects to fix en route (from external plan review, code-verified)

- **(D1) Tilewise pad-port corruption**: `interface_tile_index_maps` filters Dirichlet/pad nodes out of the maps (result_factorization.py:838-843, and the bulk path `if n in interface_node_to_idx` ~:1058) but the tilewise matvec pairs the *filtered* map with the *full* `S_i` (interface_iterative.py:264-266 — no slicing) → dimension mismatch/corruption for any tile with a pad port. Fix in Stage 2: at solver setup compute kept-position array per tile and slice `S_i[np.ix_(pos, pos)]` once (the dropped pad rows/cols are exactly what the assembly's unknown×unknown slice discards; their effect is already in `rhs_dirichlet`). Regression test with a pad-on-tile-port fixture.
- **(D2) S_extra built by giant subtraction**: `_build_s_extra_coo` (result_factorization.py:304-352) materializes Σ P_iᵀS_iP_i as a full COO (~25 GB temporaries at 107-tile scale) and computes `S_global − S_tile_sum`, leaving FP cancellation residue as spurious nnz. Fix in Stage 2: construct S_extra *directly* by stamping the mode's package edge list onto unknown indices, plus the island-penalty diagonal. **Mode-dependent (review-confirmed gap)**: DC stamps `package_edges` (resistive only); **transient stamps `combined_edges` = resistive + `C_coeff`·package-cap edges** (result_factorization.py:1577-1582) — S_extra^TD therefore depends on dt and method (`C_coeff` = 1/dt_ps BE, 2/dt_ps TR) and must be rebuilt inside every `prepare_transient` (trivially cheap; never shared with the DC solver instance). Equivalence tests: direct-vs-subtraction on a small fixture *for both modes* (transient fixture must include package caps); assert the tilewise operator (ΣS_i + S_extra incl. penalties) matches the mode's `S_global` on random vectors for BE and TR at two different dt values.

---

## Stage 0 — Baselines + microbenchmarks on the proxy (no production code)

Scripts in `scripts/benchmark/microbench/` (excluded from perf-baseline comparison):
1. Re-parse proxy at the split regime: `sigma-dvd parse netlist/netlist_brcm_sampled --net VDD_VAR --backend ray --max-interior 100000 -o netlist/netlist_brcm_sampled/distributed_pkl_mi100k`. Record tile count, n_interface, S nnz (expect ~190K / ~1.3B analog).
2. 20-step transient runs: direct; CG assembled/bj/1e-12; CG 1e-8. Reproduces the §7.4 blowup locally; dev baseline table.
3. `bench_ray_rtt.py`: ~107 actors, broadcast 1.5 MB vector + gather, 100 rounds, p50/p95 RTT (+ 30 KB per-actor slice variant). Stage 5 go/no-go number.
4. `bench_interface_matvec.py`: on the mi100k bundle — CSR matvec (1 thread), serial tilewise, **threaded-tilewise prototype (the exact Stage 2 design)**, fp32 variants, STREAM-triad ceiling.
5. GPU microbench (`--gpu`; `uv pip install cupy-cuda12x` into `venv/`): device CSR SpMV vs size-bucketed batched dense GEMV, fp32/fp64, H2D/D2H costs. Decides Stage 4 layout.
6. **rtol sweep**: 20-step 107-tile runs at rtol ∈ {1e-12, 1e-10, 1e-8, 1e-7, 1e-6} vs direct — max|ΔV|, peak-drop delta, iters/step. **rtol bounds the residual, not the voltage error (‖e‖ ≲ κ·rtol·‖x‖) — the ≤1 µV claim is empirical, so the production default is *chosen from this sweep* (1e-8 is the candidate, drop to 1e-9 if the measured max|ΔV| margin is thin), and every subsequent proxy measurement re-reports max|ΔV| vs direct as a standing accuracy gate.**
7. **Ask user to check BRCM-host GPU + node count** (record in docs).

Deliverable: docs §7.7 measurement table.

**Stage 0 COMPLETE (2026-07-17/18, full data in docs §7.7).** Key outcomes that adjust later stages:
- **rtol default 1e-8 confirmed** (max|ΔV| 166 nV vs direct, 6× inside budget; 1e-7 fails at 1.66 µV — margin real but thin). Iters 130→42/step at 36 tiles from rtol alone.
- **Finding 0 (new, blocking)**: non-streaming S_i gather + COO assembly needs >190 GB driver RSS at the 64-tile/167K regime (watchdog-killed twice); streaming slice-and-copy gather of the same blocks peaks at 26.5 GB. The Stage 2 D2 direct-stamping + tilewise-without-full-assembly work is a **hard requirement** for 107-tile BRCM, not an optimization. Also: use `tiles_per_worker` packing to cap concurrent tile-factor memory (116 unpacked actors OOM'd mi100k).
- **CPU matvec floor is ~3× worse than planned**: threaded tilewise best = 150 ms at **8 threads** (inverted scaling above 8 — accumulator zero-fill/reduction overhead; revisit scatter design in Stage 2), serial BJ 990 ms (must thread). Stage 2 CPU path ≈ 0.27 s/iter → ~4 s/step at 15 warm iters — fallback only, firmly off-target.
- **GPU confirms Stage 4 layout**: device CSR fp64 26.8 ms (1188 GB/s) ties batched dense (25.3 ms) → device-resident assembled CSR fp64. GPU iteration ≈ 30–45 ms ⇒ dev target ≤0.2 s/step needs warm iters ≤~5–10: **Stage 3 coarse space and Stage 4 GPU are jointly load-bearing**. H2D/D2H negligible (0.4 ms). fp64 fits (24/48 GB).
- **Ray RTT p50 ≈ 16 ms at 116 actors** (latency-bound, same for broadcast and slices) — per-iteration worker-side matvec pays 16 ms/iter; Stage 5 remains gated on multi-node need.
- fp32 CPU probe invalid (mixed-dtype GEMV fell off BLAS); redo in Stage 2 only if BRCM host is CPU-only.
- User input received (2026-07-18): BRCM host **CPU-only** (fp32 CPU study → critical path; GPU optional), node count **undecided** (Stage 5 stays gated).

## Stage 1 — Instrumentation, plumbing, small fixes — **COMPLETE (1a-1d: ea6d329; 1e: 6791891, 2026-07-18)**

> Landed with the full review battery (stage workflow + /code-review xhigh --fix rounds + Opus fix-verification + negative-tested regression tests). Island detection measured 59× faster at 18-tile scale with exact parity. New latents flagged for Stage 2: D1 also manifests in solve_dc RHS scatter for tile-resident-pad ports; sp.bmat crash in `_compute_schur_partial` for zero-port tiles. Stage 0's Finding 0 (S_global COO assembly >190 GB at split regime) upgrades the Stage 2 memory work: CG-tilewise must be able to run WITHOUT ever assembling S_global (BJ from tile blocks, S_extra by direct stamping), not merely free it afterwards.

- **1a Timer split + CG iters logging** (`solver_td.py`): split :827-846 into `cum_rhs_final_time` and pure `cum_solve_time` (same for QS :398-406); per-step `_cg_solver.stats['last_cg_iters']` → `loop_stats` (`cg_iters_mean/max`, per-step list) + periodic INFO line. `run_perf_baseline.py` schema extended additively (existing keys keep meaning).
- **1b Plumb `atol/maxiter/strict`; rtol default → 1e-8**: new settings + CLI flags; change default at interface_iterative.py:171, factory :731, CLI, and the four `settings.get('interface_cg_rtol', …)` fallbacks. Tests: pin `rtol=1e-12` in all tight CG-vs-direct assertions (unit + equivalence suite); add `test_rtol_1e8_dc_error_below_1uV` (sampled marker).
- **1c Host-aware budgets**: `interface_factor_memory_budget` default `'auto'` = `min(32 GB, 0.4×psutil RAM)` in `auto_select_interface_solver`; `BLOCK_JACOBI_MAX_FACTOR_BYTES` → setting `interface_block_jacobi_max_bytes` default `'auto'` = `min(8 GB, 0.1×RAM)`, louder fallback warning.
- **1d Route both factor paths (and both refactors) through `build_interface_solver`** — extend factory signature (S_extra, atol, maxiter, strict, use_streaming for the downgrade logic); kill the four inline constructions.
- **1e Island detection via parse-time pre-clean + connectivity summaries + coordinator union-find** (adopted design — eliminates both the 238 s proxy / 900 s BRCM Schur-BFS cost AND the duplicated worker-side component traversal at every model creation):
  - **Parse side — universal pre-clean, single-pass** (in `DistributedNetlistParser.parse_and_dump()`; promotes the existing split-path machinery to all tiles **without re-loading pkls**): the global port set is hoisted *ahead* of the parse pass — (step 0, cheap + parallel) `collect_shared_boundary_nodes` (parser.py:859) streams just the `*`-prefixed boundary declarations of every `.ckt` (no graph build; already exists — the no-pkl model path uses it) → `shared_bnd`; package.ckt parse → `die_nodes`. (Step 1) each parse map task then receives `(paths, shared_bnd, die_nodes)` and, **in memory before its single pkl write**: parse tile → `_pre_clean_tile_data(tile_data, port_nodes = (all_nodes ∩ shared_bnd) ∪ (all_nodes ∩ die_nodes), threshold=5)` — the *existing* retile.py function with port-set semantics bit-identical to worker removal (tile-resident pads deliberately excluded from the port set, matching today) → sets `pre_cleaned_full=True` → compute summary → write the *clean* pkl once, return stats + summary. No second pass, no reload/rewrite; the cost is one extra streaming boundary scan (a fraction of full parse, also parallel). (Step 2) `_apply_tile_splits` unchanged — it loads only oversized tiles' pkls (its own parent pre-clean step becomes redundant since parents arrive clean); **sub-tiles get their threshold-1 pass there** (post-split, port candidates including the cut-interface nodes) — run, not assumed no-op (the B1 RC-coupled-component lesson) — and their summaries replace the parent's. One component traversal per tile total, reused to emit the summaries.
  - **Summaries** (decision-free consequence of the pre-clean; extend `_pre_clean_tile_data` to return its component decomposition): per *kept* component: interface-candidate nodes (component ∩ (shared_bnd ∪ die_nodes)), component node count, `has_pad` (tile-resident V-source in component — used only for island liveness, never for removal). Aggregated into `metadata.pkl` under `CONNECTIVITY_SUMMARY_VERSION`, alongside `parser_interface_set = shared_bnd ∪ die_nodes` (post-split). ≈10–20 MB.
  - **Model creation — one trust assertion**: `create_distributed_model` finalizes `interface_nodes = bundle.shared_boundary_nodes | die_attachment_nodes` (model.py:585-588) exactly as today; new assertion `interface_nodes == metadata.parser_interface_set`. On the pkl path this equality is structural (both sides derive from the same metadata.pkl the parser wrote) — the assertion guards the `.ckt`-rescan fallback (model.py:398-426, a re-derivation), future model.py drift, and stale/edited metadata. Mismatch → WARNING, trust flag off → full legacy path (worker removal + Schur BFS).
  - **Worker setup**: `_build_block_system` **skips `_remove_floating_islands` entirely** when `tile_data.pre_cleaned_full` and trust holds (the pkl is already clean — no adjacency dict, no BFS, no mutation); legacy bundles run today's path unchanged. `build_block_system_from_edges`/`BlockMatrixSystem` untouched — same call site, same inputs, bit-identical G blocks since the removal rule and port set are identical.
  - **Coordinator side** (new `detect_interface_islands_from_summaries` in `pgmath/schur.py`, invoked in `prepare()` **before factorization**, ~ms): **heuristic-free** union-find over all interface unknowns as singletons — union each summarized component's interface-candidates ∩ finalized interface; union package-resistor endpoints (g>0, non-ground); mark pad-connected roots (component `has_pad` + package pad/Dirichlet endpoints). Unreached singletons island themselves (matches the BFS's zero-row behavior). Islands = padless roots. **Transient mode**: additionally union package-capacitor endpoints; summaries are cap-independent, so one parse-time summary serves both DC and TD (compute both sets in one pass).
  - **Exactness argument — reviewed against the actual BFS implementation (`find_interface_islands`, pgmath/schur.py:731-823), record in the module docstring.** The BFS's adjacency predicate is `S[u,v] != 0.0` off-diagonal (exact-zero test, no epsilon), liveness is "component contains a pad node", pads are reachable **only** through `extra_edges` (g>0, non-ground endpoints), and there are no size heuristics — so the union-find must replicate exactly three relations, and does:
    1. *Tile coupling*: S[u,v] = Σ_i S_i[u,v] + package stamps. For ports u,v in **different** local resistive components of a tile, G_ii⁻¹ block-diagonalizes ⇒ S_i[u,v] = 0.0 exactly (the stored structural zero the value-aware BFS skips). For u,v in the **same** component, the M-matrix sign structure makes S_i[u,v] strictly negative (G_pp[u,v] ≤ 0 minus a strictly positive correction — same-sign accumulation, no cancellation), and the cross-tile sum keeps the sign ⇒ nonzero. Hence BFS-edge(u,v) ⟺ same-tile-component(u,v) ∨ package-edge(u,v) — exactly the union-find relation. (FP underflow to exact 0.0 would need couplings < ~1e-308 mS — physically impossible; the oracle tests stand guard regardless.)
    2. *Filters*: tile components computed excluding ground '0' (ground edges are diagonal-only — existing rule); package unions replicate the BFS's `g <= 0` skip and `'0'`-endpoint skip verbatim.
    3. *Liveness*: package pad endpoints ("virtual pads") ⇒ mark root live — identical. **One deliberate divergence (BFS bug found in this review):** a component kept alive only by a *tile-resident* Dirichlet pad (e.g. `additional_vsrcs` on a die node) has NO pad adjacency in S_global — the pad column is sliced into `rhs_dirichlet` — so the current BFS mislabels the healthy, nonsingular component as an island and penalty-pins it to ~Vdd. The summary's per-component `has_pad` flag fixes this. Document it, emit a WARNING when the flag rescues a component the BFS would have penalized, and verify against the **flat oracle** (not the BFS) on a dedicated tile-resident-pad fixture; the summary-vs-BFS oracle tests exempt exactly this case.
    Transient: A_ip = G_ip and tile caps are grounded/diagonal (`build_grounded_capacitance_diags`) ⇒ transient S_i has the *same* off-diagonal structure as DC S_i, so the R-only tile summaries remain valid; the only TD difference is coordinator-side (union `combined_edges` = resistive + C_coeff·package-cap, matching :1577-1582 — c_fF>0 ⟺ C_coeff·c>0, so the g>0 filter is C_coeff-invariant). Complexity drops from O(S.nnz) ≈ 1.3B to O(Σ boundary incidences + package edges) ≈ 300K.
  - **Fallback matrix** (decided once at model creation, applied consistently): no summaries / version mismatch / trust assertion failed / `island_detection='schur_bfs'` override → workers run legacy `_remove_floating_islands` AND prepare runs the legacy Schur-BFS (with an INFO log recommending re-parse). The flow is all-new or all-legacy — never a mix. BFS remains the test oracle. A7 topology-context caching unchanged (results still stored there).
  - **Tests**: (i) **end-to-end parity** — parse the same netlist with pre-clean on vs forced-legacy: identical per-tile `boundary_nodes`, `BlockMatrixSystem` dimensions, S_global (exact), island sets, and DC/transient solutions (netlist_test + two-tile fixtures, in the equivalence suite); (ii) oracle equivalence — identical island sets summary-union-find vs Schur-BFS on fixtures engineered with: a pad-less boundary component, a cap-coupled-only component (DC: island, TD-with-package-cap: not), a package-resistor-bridged component, a multi-component single tile (structural-zero case), a **removal-created island** (boundary node whose components are dropped in both adjacent tiles → unreached singleton == BFS zero-row island); (iii) **sub-tile threshold-1-at-parse fixture** — a component connected only through a cut-interface node survives the split-path pre-clean; (iv) the **tile-resident-pad fixture** asserts the documented divergence (summary: live, BFS: island) and validates against the flat oracle's voltages; (v) **trust-assertion test** — tamper `parser_interface_set`, assert WARNING + full legacy path + correct results; (vi) legacy-bundle fallback (strip summaries, assert legacy path + INFO); (vii) netlist_test/netlist_sampled DC+TD identical-set assertions; proxy measurement records detect-islands wall time (expect ~ms at prepare, from 238 s) and the parser log reports the pre-clean pass overhead.

Gates (here and every stage): `pytest tests/distributed -m unit`; equivalence suite (68+6 xfail, rtol pins added); `run_perf_baseline.py --compare … --max-regress 10%`; four-notebook regression bit-identical (small systems resolve to direct — untouched). Proxy 20-step re-run into §7.7.

**Gate-coverage fix (review point 7 — the standing gates exercise small *direct* solves, not the target path):** add a **forced-CG equivalence matrix** to the per-stage gates: parametrize the existing equivalence tests over `interface_solver=cg` × `{assembled, tilewise}` × `{block_jacobi, two_level}` × (Stage 4+: `{cpu, cuda}`) at pinned rtol 1e-12 on netlist_test/netlist_sampled, asserting the usual exactness tolerances. And every 107-tile proxy measurement run must report **max|ΔV| vs the direct reference** alongside timings — perf numbers without the correctness diff don't count as a passed gate.

## Stage 2 — Threaded tilewise matvec + threaded block-Jacobi apply

`interface_iterative.py` tilewise branch:
- **Fix D1 first**: per-tile kept-position slicing `S_i_kept = S_i[np.ix_(pos, pos)]` at solver setup so filtered index maps and block dimensions always agree; pad-on-tile-port regression fixture.
- **Fix D2**: replace `_build_s_extra_coo` subtraction with direct stamping of package extra_edges + island-penalty diagonal; random-vector operator-equivalence test vs `S_global`.
- `matvec_threads: int|'auto'` (setting/CLI; auto = `min(32, cpu_count, n_tiles)`); persistent `ThreadPoolExecutor` owned by solver (lazy, with `close()`+finalizer).
- Static LPT partition of tiles by n_p² cost; per-thread accumulator rows in a preallocated `(n_threads, n)` array; thread work = gather → GEMV (releases GIL) → bincount into own row; reduce + S_extra term at the end. If bincount GIL contention shows in Stage 0 prototype, concatenate per-thread scatters into fewer bincount calls.
- Guard BLAS nesting with `threadpoolctl.threadpool_limits(1)` around the pool region.
- Add `_matmat(X)` (per-tile GEMM + scatter) — Stage 3 setup needs it.
- Threaded BJ apply: same pool, disjoint ownership slices (no accumulators), cho_solve releases GIL.
- **SPD-safe fallback**: replace the `np.linalg.pinv` fallback for ill-conditioned BJ blocks (interface_iterative.py:447-455) with an eigh-based PSD projection (clip eigenvalues to ≥ε·λ_max) — pinv of a numerically indefinite block can yield a non-PSD preconditioner, voiding CG's convergence guarantee. Test with a constructed indefinite block.
- `interface_matvec_mode` default → `'auto'` (tilewise when S_i present, else assembled; explicit values honored).
- Fix `refactor()` downgrade: when workers attached (they must be for `factor()` anyway), re-gather S_i via `factor_and_compute_schur` and rebuild tilewise; else keep assembled fallback + warning.
- `interface_matvec_dtype: float64|float32` experimental (fp32 storage + fp64 accumulate; residual floor ~1e-7 ⇒ pair with rtol ≥1e-7). Promoted to critical path iff BRCM host is CPU-only.
- `interface_drop_s_global` opt-in (see Memory decision).

Tests: threaded-vs-serial ≤1e-13 rel (two-tile + randomized ~30-tile synthetic), thread-count invariance (1/2/8), matmat vs column matvecs, threaded BJ vs serial, auto-mode resolution, refactor-rebuilds-tilewise, drop-s-global-blocks-save.
**Gate:** matvec ≤60 ms at 190K on dev (≤1.2 × 12.5 GB / measured STREAM BW); solve/step ≥5× vs Stage 0 CG baseline; equivalence exact at pinned 1e-12.

## Stage 3 — Coarse-space two-level preconditioner

New `src/distributed/interface_coarse.py`.

**Why it's needed (the algorithmic gap):** CG iterations ≈ √κ(M⁻¹S). Block-Jacobi is a *local* preconditioner — each application corrects error only within a tile's own port block, so a die-spanning smooth error mode is corrected one tile-hop per iteration. Spectrally, M_BJ⁻¹S carries a cluster of ~O(T) small eigenvalues whose eigenvectors are near-constant per tile; CG resolves them essentially one at a time. This is the measured ~180 warm iters at 107 tiles (κ ~ 1/H² in DDM terms), and it worsens with further splitting. Relaxing rtol truncates the tail but cannot remove the cluster. The fix is a second, tiny *global* correction that solves exactly those per-tile-constant modes directly; with it, condition (and iteration count) becomes essentially independent of tile count (Nicolaides coarse space; additive Schwarz two-level theory, Toselli–Widlund).

**Construction (all inputs already on the context):**
1. Multiplicity `m = np.bincount(np.concatenate(list(tile_index_maps.values())), minlength=n)` — m[g] = number of tiles whose port list contains interface node g (cut planes 2, corners 3–4).
2. Partition-of-unity basis Z sparse CSR (n × T′), column j: `Z[g,j] = 1/m[g]` for g in `tile_index_maps[j]`. Rows sum to 1 ⇒ the constant vector (lowest-energy mode of the SPD interface operator) lies exactly in range(Z). nnz(Z) = Σ n_ports,i (~233K at 107 tiles), 1–2 entries per row.
3. Coarse operator: `SZ = linear_op._matmat(Z.toarray())` — per tile a dense GEMM `S_i @ Z[idx_i,:]` scattered into a dense (n × T′) ≈ 165 MB result. One pass over the 12.5 GB of S_i blocks but with T′ columns of arithmetic ⇒ compute-bound GEMM, a few hundred ms–couple of seconds, **once per factored context** (DC and transient separately — S differs by the C/dt term). `_matmat` must include the S_extra package term so the coarse space sees package coupling. Retain SZ for the deflation contingency.
4. `S_c = Zᵀ @ SZ` (T′×T′ dense, ~108²) → `cho_factor` with the existing jitter+pinv fallback idiom; log cond(S_c).
5. Per-iteration apply (right-to-left): `w = Zᵀr` (sparse gather-sum ~400K flops) → `y = cho_solve(S_c, w)` → `r_c = Z·y` (scatter). Sub-ms — invisible next to the 12.5 GB matvec; per-iteration cost stays ≈ matvec + BJ apply.

**Application — additive two-level:** `M⁻¹r = M_BJ⁻¹r + Z S_c⁻¹ Zᵀr`. Symmetric and PD (SPD + PSD sum) ⇒ valid for unmodified `scipy.sparse.linalg.cg`; Stage 3 is pure preconditioner code, zero solve-driver changes. Rejected alternatives: multiplicative/hybrid (better constants but an extra S-apply per application — doubles matvec traffic, our dominant cost); BNN (needs per-tile Neumann solves we don't have cheaply).

**Edge cases (each gets a test):**
- Island-penalized nodes (`ctx._removed_interface_nodes`): `apply_island_penalty`'s 1e5 diagonals would penalty-dominate S_c and neuter the coarse correction — zero those rows of Z.
- Unowned nodes (package/die unknowns in no tile port map; BJ treats them as identity): add one extra indicator column (T′ = T + 1) so they're not invisible to the coarse space.
- All-zero columns (tile fully islanded/Dirichlet) ⇒ S_c singular — drop the column before factoring, log it.
- **Structural rank deficiency on regular partitions**: on a grid-like tiling where interface nodes have even multiplicity and the tile-adjacency graph is bipartite (checkerboard 2-coloring), the alternating-sign combination of Z's columns vanishes on every shared node — Z (and hence S_c) is genuinely rank-deficient by ≥1. This is the *expected* case for B1's rectangular bisections, not a corner case. Therefore factor S_c via `eigh` with a PSD pseudo-inverse (zero out eigenvalues ≤ ε·λ_max) instead of plain cho/pinv — preserves symmetry and PSD-ness of the coarse term (Z S_c⁺ Zᵀ stays PSD, so M stays SPD for CG). Log the detected rank. Test: 2×2 checkerboard fixture asserting rank(S_c) = T′−1 handled without failure and iters still improve.
- Warm-start interaction: within one transient S, Z, S_c are fixed — nothing rebuilds per step; CG warm start composes with the coarse space (this is where 20–50 cold iters drop to ~5–20 warm).

**Fallback ladder if the gate is missed:** (1) A-DEF2 deflation — `M⁻¹(I − S·Z S_c⁻¹ Zᵀ) + Z S_c⁻¹ Zᵀ` with projected initial guess, typically another 1.5–2× fewer iters; needs a hand-rolled PCG loop, which Stage 4's GPU path builds anyway (reuses retained SZ). (2) GenEO-lite enrichment — append the 2–4 lowest eigenvectors of each tile's already-factored dense BJ block to Z (T′ ≈ 300–500, still trivial); this is the robust coarse space when iteration counts stay high due to PDN heterogeneity (orders-of-magnitude layer-conductance contrast is exactly the regime where per-tile-constant spaces underperform).

**Wiring:** new preconditioner value `'two_level'` (CLI choices, YAML, factory); resolved default when CG+tilewise (107-tile regime); `block_jacobi` stays default elsewhere (small systems resolve to direct — notebooks/equivalence untouched).

**Expected numbers:** ~180 warm (bj, 1e-12) → ~100–130 at rtol 1e-8 → 20–50 cold / ~5–20 warm with the coarse space. At 10–15 warm iters: ~0.6–0.9 s/step interface solve with the Stage 2 CPU matvec on dev, ~0.15–0.3 s with the Stage 4 GPU matvec — the iteration cut multiplies whichever matvec backend wins, which is why Stage 3 precedes Stage 4.

Tests: `Zᵀ S Z == S_c` vs assembled S incl. S_extra (two-tile), strict iters reduction vs bj (two-tile + sampled), islanded-tile column, unowned package nodes, all-zero-column drop, 1e-12 agreement with direct.
**Gate:** warm iters/step ≤30 at rtol 1e-8 on 107-tile proxy (record cold too); coarse setup ≤ a few seconds.

## Stage 4 — GPU mode (cupy, optional)

New `src/distributed/interface_gpu.py`, lazy cupy import:
- Layout per Stage 0 data (expected: device-resident assembled CSR — within ~20% of batched dense blocks, far simpler; batched GEMV only if ≥1.3× win).
- **Whole PCG loop on device**: rhs H2D + x D2H per step (~1.5 MB); host syncs only for dot-product scalars. BJ on device = precomputed dense block inverses (from existing cho factors, ~2.7 GB) as size-bucketed padded batched GEMM, disjoint scatter. Coarse solve on device (T'≈108 trivial). `CudaInterfacePCG.__call__(rhs)->np.ndarray`, warm start device-resident, stats/strict/rtol semantics shared with `InterfaceCGSolver` via a small stopping-rule mixin (no drift).
- Setting `interface_compute: cpu|cuda|auto` (auto = cuda iff cupy imports AND device mem ≥1.3× estimated bytes) + CLI; automatic CPU fallback on import/OOM. GPU state never persisted (rebuilt at factor).
- `interface_gpu_dtype`: **fp64 default** (15.4 GB CSR fits the 48 GB card; fp64 needs no accuracy caveats). fp32 mode is experimental and only meaningful for smaller GPUs / the BRCM host: the CG recurrence residual drifts from the true residual at fp32, so rtol acceptance **must not** rely on the recurrence alone — require either (a) a final true-residual check against the fp64 CPU tilewise operator (one ~50 ms CPU matvec per solve, acceptance-only) with a one-shot fp64 refinement correction if it misses, or (b) hard-pair fp32 with rtol ≥ 1e-7 plus the Stage 0-style empirical max|ΔV| study. Promote fp32 beyond experimental only with (a) implemented.

Tests (skipif no cupy, except fallback test which monkeypatches the import): matvec vs CPU ≤1e-12 fp64, PCG solution match, warm-start persistence, BJ apply match, auto-fallback-without-cupy, sampled DC vs direct, fp32 true-residual acceptance path.
**Gate:** interface solve ≤0.2 s/step on dev at 107-tile proxy, rtol 1e-8, warm.

## Stage 5 (gated) — Worker-side / s-step CG

Go/no-go with Stage 0 numbers: per-step ≈ I × (RTT + t_gemv_max). **Descope to a design note if** Stage 4 meets target AND BRCM host has/gets a GPU, or if BRCM deployment turns out single-node (no bandwidth gain — physics). Proceed if BRCM is CPU-only AND multi-node.

**Correction to the original s-step sketch (review point, confirmed):** batching s CG iterations into one round trip is *mathematically impossible* with only per-tile S_i on workers — S = Σ P_iᵀS_iP_i, so computing S²x requires the global scatter-add after the first apply (a tile's second apply needs neighbor contributions on shared nodes). True communication-avoiding matrix-powers kernels need s-hop ghost replication of neighbor blocks, which for *dense* port blocks on a 2-D tile adjacency is a large multiple of the data — cost it in the design note, expect it to be unattractive. The realistic ladder is therefore:
1. Plain per-iteration `apply_schur` RPC — one broadcast + gather per iteration; viable iff RTT is small (few ms at ~107 actors, Stage 0 measures it).
2. **Pipelined CG (Ghysels–Vanroose)** — same communication *count* but overlaps the dot-product reduction with the matvec, hiding one RTT per iteration; modest numerical-stability cost, well understood; needs the hand-rolled PCG loop Stage 4 already builds.
3. Ghost-replicated CA-CG — only if 1–2 fail and multi-node CPU is truly the production mode; design note first, implementation only with explicit sign-off.

Sketch for 1–2: `TileWorker.retain_schur()` so `_cached_schur` survives B3 reads; RPCs `apply_schur(x_local)` / `apply_schur_block(X_local)`; matvec mode `'worker'` in the factory; coordinator keeps only S_extra + preconditioner. Worker-side settings propagate via `TileWorker.configure` (Ray globals pitfall). Also removes the streaming×tilewise incompatibility for good.

---

## Lifecycle-parity checklist (applied to EVERY stage that adds solver state)

New state introduced by this project — thread pools (Stage 2), retained/sliced S_i and S_extra (Stage 2), SZ/S_c coarse operators (Stage 3), GPU device arrays (Stage 4), worker-retained `_cached_schur` (Stage 5) — must be threaded through the full context lifecycle, which the original sketch under-specified:

- `release()`: close thread pools, free device memory (`cupy` mempool), drop S_i/SZ/S_c, RPC workers to drop retained Schur blocks. No leaked actors/VRAM after release (test: repeated prepare/release loop, assert RSS/VRAM stable).
- `save()`/`load()`: dense S_i, SZ/S_c, and GPU state are **never persisted** (documented; rebuilt at factor/refactor). `save()` under `interface_drop_s_global` raises with guidance.
- `refactor()`: rebuilds whatever the resolved mode needs (re-gather S_i for tilewise, re-upload GPU, rebuild coarse operators) — no silent mode downgrades (extends the Stage 2 refactor fix).
- Adjoint (`solver_adjoint.py` checks `_interface_solver_mode`): must either work with each new mode or fall back explicitly with a warning — test per new mode.
- Decompose multi-sweep (`analyze_distributed_decomposition`): define warm-start semantics across sweeps — call `reset_warm_start()` between victims (different masked RHS trajectories; stale x0 is a slow-convergence trap, not a correctness one — but make it deterministic). Test: two sequential `solve_transient` calls on one context, assert iteration counts are reproducible.
- Settings propagation: all new settings through the three plumbing layers (CLI → `model.settings` → YAML role configs), and worker-side flags via `TileWorker.configure` (Ray module-globals pitfall). One test per new setting asserting it reaches the object that consumes it.

## Verification (end-to-end)

1. Every stage: the four standing gates + 20-step 107-tile proxy measurement appended to `docs/brcm_distributed_runtime_optimization.md` §7.7.x (solve s/step via new split timer, iters mean/max, matvec ms, precond ms, RSS peak).
2. Correctness: CG at pinned rtol 1e-12 stays exact vs direct (equivalence suite); rtol 1e-8 error ≤1 µV vs direct on proxy DC + 20-step transient.
3. Final: full-length (2000-step) proxy transient with the winning config vs §7.6 baseline (0.626 s/step, solve 0.156); projection onto BRCM shares per the §7.6 rule; user re-measures on BRCM host (commands recorded in memory + docs).
4. Success criterion: 107-tile proxy interface solve ≤0.1–0.2 s/step (dev) ⇒ projected ≤0.3–0.5 s/step BRCM-host GPU (or documented CPU-only landing ~1–1.5 s/step with fp32), enabling loop ≈ 1.49 + solve ≈ 2 s/step at 107 tiles on BRCM.
