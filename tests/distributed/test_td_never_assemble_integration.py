"""TD never-assemble work package: netlist_multi_tile end-to-end smoke.

Companion to ``test_td_never_assemble.py`` (unit fixtures) -- this file
covers the same ``_factor_transient_context_no_s_global`` path against a
real parsed netlist, per the work package's gate 4:

  "netlist_multi_tile smoke: transient never-assemble vs assembled --
  max|dV| <= 1e-9 V at pinned 1e-12, and never-assemble path logs confirm
  no S_global assembly."

Requires ``netlist/netlist_multi_tile/distributed_pkl`` (re-parse via
``python -m distributed parse netlist/netlist_multi_tile --net VDD -o
netlist/netlist_multi_tile/distributed_pkl`` if missing) AND
``model.island_detection_mode == 'summaries'`` (the never-assemble
precondition) -- skips gracefully otherwise.
"""
from __future__ import annotations

import logging
import os

import pytest

# Finding 10: import the canonical netlist path from tests/fixtures.py
# instead of re-deriving it via a dirname chain (tests/CLAUDE.md: "always
# import these instead of hardcoding paths") -- `tests/` and
# `tests/distributed/` are both real packages (`__init__.py` present), so
# `tests.fixtures` is importable directly, matching the precedent in
# tests/distributed/test_island_summaries_integration.py
# (`from tests.fixtures import NETLIST_TEST`).
from tests.fixtures import NETLIST_MULTI_TILE

pytestmark = pytest.mark.integration

NETLIST_MULTI_TILE_DIR = str(NETLIST_MULTI_TILE)


def _never_assemble_settings(rtol=1e-12, atol=1e-14):
    return {
        'interface_solver': 'cg',
        'interface_matvec_mode': 'tilewise',
        'interface_preconditioner': 'block_jacobi',
        'interface_cg_rtol': rtol,
        'interface_cg_atol': atol,
        'interface_drop_s_global': True,
    }


def _assembled_cg_settings(rtol=1e-12, atol=1e-14):
    return {
        'interface_solver': 'cg',
        'interface_matvec_mode': 'tilewise',
        'interface_preconditioner': 'block_jacobi',
        'interface_cg_rtol': rtol,
        'interface_cg_atol': atol,
    }


class TestNetlistMultiTileSmoke:

    def _pkl_dir(self):
        pkl_dir = os.path.join(NETLIST_MULTI_TILE_DIR, 'distributed_pkl')
        if not os.path.isdir(pkl_dir):
            pytest.skip(
                f"{pkl_dir} not found -- run "
                f"`python -m distributed parse {NETLIST_MULTI_TILE_DIR} --net "
                f"VDD -o {pkl_dir}` first."
            )
        return pkl_dir

    def test_transient_never_assemble_vs_assembled_max_dv(self, tmp_path, caplog):
        from distributed.model import load_distributed_partitions, create_distributed_model
        from distributed.solver import DistributedDDMSolver

        pkl_dir = self._pkl_dir()

        def _run(drop_s_global, subdir):
            bundle = load_distributed_partitions(str(pkl_dir))
            model = create_distributed_model(bundle, backend='local')
            if model.island_detection_mode != 'summaries':
                pytest.skip(
                    f"model.island_detection_mode="
                    f"{model.island_detection_mode!r} (need 'summaries' -- "
                    f"re-parse with the Stage 1e parser)"
                )
            settings = (
                _never_assemble_settings() if drop_s_global
                else _assembled_cg_settings()
            )
            model.settings.update(settings)
            solver = DistributedDDMSolver(model)
            dc_ctx = solver.prepare()
            trans_ctx = solver.prepare_transient(dt=1e-10, method='be', verbose=True)
            try:
                assert (trans_ctx._S_global is None) == drop_s_global
                smoothed = solver.preprocess_sources(
                    time_step=1e-10, t_start=0.0, t_end=1e-9, smooth=False,
                    pkl_dir=str(tmp_path / subdir),
                )
                result = solver.solve_transient(
                    trans_ctx, dc_context=dc_ctx, t_end=1e-9,
                    smoothed_sources=smoothed,
                )
                return result.as_flat()
            finally:
                trans_ctx.release()
                dc_ctx.release()
                model.shutdown()

        v_assembled = _run(False, 'assembled')
        # Finding 5: caplog.records accumulates for the WHOLE test, not just
        # the block inside `at_level` -- the assembled _run(False, ...) call
        # above also runs with verbose=True and logs the exact 'nnz (density'
        # line at INFO. Under the default pytest capture level (WARNING) that
        # line is filtered out before it ever reaches caplog.records, so the
        # assertions below happen to pass today -- but that's an accident of
        # the ambient log-level config, not something this test controls: a
        # run with `--log-level=INFO` (or a future `log_level`/`log_cli_level`
        # added to pyproject.toml) would capture the assembled run's INFO
        # records too, and the "never-assemble path must not log nnz/density"
        # assertion would then spuriously fail on the ASSEMBLED run's own
        # (entirely expected) record. Clear the log BEFORE entering the
        # never-assemble run so caplog.records below only ever reflects that
        # run, regardless of ambient log-level configuration.
        caplog.clear()
        with caplog.at_level(logging.INFO, logger='distributed.result_factorization'):
            v_never_assemble = _run(True, 'never_assemble')

        # Gate 4: never-assemble path logs confirm no S_global assembly.
        # Both paths log a line starting "Transient interface: ..." (same
        # prefix, different content), so distinguish by the assembled
        # path's S_global-specific suffix ("nnz (density ...%)", from its
        # `_sparse_mem_bytes(S_global)`-derived stats line) rather than the
        # shared prefix -- absence of THAT is the no-assembly signal,
        # alongside trans_ctx._S_global is None (asserted inside _run
        # above) and the never-assemble path's own distinct verbose header.
        # caplog.records here is scoped to the never-assemble run only (see
        # the caplog.clear() above).
        assert not any(
            'nnz (density' in rec.message for rec in caplog.records
        ), (
            "never-assemble path must not log S_global's nnz/density stats "
            "(that line only exists on the assembled path, which computes "
            "it from a real S_global matrix)"
        )
        assert any(
            'never-assemble S_global' in rec.message for rec in caplog.records
        ), "expected the never-assemble path's own verbose header to log"

        common = set(v_assembled) & set(v_never_assemble)
        assert common, "fixture produced no comparable nodes"
        max_dv = 0.0
        for node in common:
            d_a, _ = v_assembled[node]
            d_n, _ = v_never_assemble[node]
            max_dv = max(max_dv, abs(d_a - d_n))
        assert max_dv <= 1e-9, f"max|dV|={max_dv:.3e} exceeds the 1e-9 V gate"
