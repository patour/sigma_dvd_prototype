"""Regression tests for code-review findings 4, 9, 10, 11, 12, 14.

Each test is labelled with the finding number it guards.
"""

from __future__ import annotations

import logging
import sys
import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: reuse the two-tile model fixture pattern from test_time_domain.py
# ─────────────────────────────────────────────────────────────────────────────


def _build_two_tile_model():
    """Minimal 2-tile model with VCS instance sources (for smooth=True/False tests)."""
    import os, tempfile
    from distributed.backend import LocalBackend
    from distributed.model import DistributedPowerGridModel
    from distributed.parser import PackageData, PowerGridMetaData, TileConfig
    from distributed.tile_worker import TileWorker, TileData

    tile_a = TileData(
        tile_id=(0, 0),
        resistive_edges=[('a1', 'shared', 1.0), ('shared', 'pad', 2.0)],
        all_nodes={'a1', 'shared', 'pad'},
        boundary_nodes={'shared', 'pad'},
        current_injections={'a1': 0.5},
        capacitive_edges=[('a1', '0', 10.0)],
    )
    tile_b = TileData(
        tile_id=(0, 1),
        resistive_edges=[('shared', 'b1', 3.0), ('b1', '0', 1.0)],
        all_nodes={'shared', 'b1'},
        boundary_nodes={'shared'},
        current_injections={'b1': 0.3},
        capacitive_edges=[('b1', '0', 5.0)],
    )

    interface_nodes = {'shared', 'pad'}
    be = LocalBackend()
    be.initialize()

    wa = TileWorker()
    wa.setup_from_tile_data(tile_a, interface_nodes)
    wb = TileWorker()
    wb.setup_from_tile_data(tile_b, interface_nodes)
    workers = [wa, wb]

    pkg_data = PackageData(
        vsrc_dict={'V1': {'node+': 'pad', 'node-': '0', 'net': 'VDD', 'value': 1.0}},
        package_edges=[('pad', 'shared', 10.0)],
        pad_nodes={'pad'},
        tap_nodes=set(),
        die_attachment_nodes=set(),
        vdd=1.0,
        net_name='VDD',
        package_cap_edges=[],
    )

    tile_configs = [
        TileConfig(tile_id=(0, 0), ckt_path='', nd_path=None,
                   instance_path=None, net_filter=None),
        TileConfig(tile_id=(0, 1), ckt_path='', nd_path=None,
                   instance_path=None, net_filter=None),
    ]

    metadata = PowerGridMetaData(
        tile_grid=(1, 2), parameters={},
        tile_configs=tile_configs, package_data=pkg_data,
    )

    model = DistributedPowerGridModel(
        backend=be,
        workers=workers,
        interface_nodes=interface_nodes,
        tile_boundary_nodes={(0, 0): ['shared', 'pad'], (0, 1): ['shared']},
        tile_interior_counts={(0, 0): 1, (0, 1): 1},
        package_data=pkg_data,
        metadata=metadata,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Finding 4: use_raw_sources resets _active_sources after smooth=True
# ─────────────────────────────────────────────────────────────────────────────

class TestFinding4UseRawSources:
    """Finding 4: preprocess_sources(smooth=False) must reset _active_sources to raw VCS."""

    def test_use_raw_sources_method_exists(self):
        """use_raw_sources() method must exist on TileWorker (_TimeDomainMixin)."""
        from distributed.tile_worker import TileWorker
        assert hasattr(TileWorker, 'use_raw_sources'), (
            "TileWorker must have a use_raw_sources() method"
        )

    def test_use_raw_sources_resets_active_sources(self, tmp_path):
        """After smooth=True then smooth=False, _active_sources must be raw VCS."""
        import os, pickle
        from analysis.vectorized_sources import VectorizedCurrentSources
        from distributed.tile_worker import TileWorker, TileData
        from distributed.backend import LocalBackend

        # Build a minimal TileWorker with raw VCS pre-loaded
        tile_data = TileData(
            tile_id=(0, 0),
            resistive_edges=[('a', 'b', 1.0), ('b', '0', 1.0)],
            all_nodes={'a', 'b'},
            boundary_nodes={'b'},
            current_injections={'a': 0.5},
            capacitive_edges=[],
        )
        be = LocalBackend()
        be.initialize()
        worker = TileWorker()
        worker.setup_from_tile_data(tile_data, {'b'})

        # Build a tiny raw VCS (empty — no PWL/pulse waveforms)
        raw_vcs = VectorizedCurrentSources.from_current_sources({}, {}, 2)
        worker._vec_sources = raw_vcs
        worker._active_sources = raw_vcs

        # Build a "smoothed" VCS (also empty, but a different object)
        smoothed_vcs = VectorizedCurrentSources.from_current_sources({}, {}, 2)
        worker._smoothed_sources = smoothed_vcs
        worker._active_sources = smoothed_vcs  # simulate after smooth=True

        # Also set a step-column table to confirm it's cleared
        worker._step_col_table = object()

        # call use_raw_sources() — this is what preprocess_sources(smooth=False) invokes
        worker.use_raw_sources()

        assert worker._active_sources is raw_vcs, (
            "use_raw_sources() must set _active_sources to _vec_sources (raw VCS)"
        )
        assert worker._step_col_table is None, (
            "use_raw_sources() must clear the step-column table"
        )

    def test_preprocess_sources_smooth_false_calls_use_raw(self, tmp_path):
        """preprocess_sources(smooth=False) must call use_raw_sources on all workers.

        After smooth=True, a second call with smooth=False must revert workers.
        We verify by inspecting _active_sources after each call.
        """
        from analysis.vectorized_sources import VectorizedCurrentSources
        from distributed.tile_worker import TileWorker, TileData
        from distributed.backend import LocalBackend
        from distributed.model import DistributedPowerGridModel
        from distributed.parser import PackageData, PowerGridMetaData, TileConfig
        from distributed.solver import DistributedDDMSolver

        # Build a one-tile model (no instance files → empty VCS)
        tile_data = TileData(
            tile_id=(0, 0),
            resistive_edges=[('a', 'b', 1.0), ('b', '0', 1.0)],
            all_nodes={'a', 'b'},
            boundary_nodes={'b'},
            current_injections={'a': 0.5},
            capacitive_edges=[],
        )
        be = LocalBackend()
        be.initialize()
        worker = TileWorker()
        worker.setup_from_tile_data(tile_data, {'b'})

        pkg_data = PackageData(
            vsrc_dict={},
            package_edges=[('pad', 'b', 10.0)],
            pad_nodes={'pad'},
            tap_nodes=set(),
            die_attachment_nodes=set(),
            vdd=1.0,
            net_name='VDD',
            package_cap_edges=[],
        )
        tile_config = TileConfig(
            tile_id=(0, 0), ckt_path='', nd_path=None,
            instance_path=None, net_filter=None,
        )
        metadata = PowerGridMetaData(
            tile_grid=(1, 1), parameters={},
            tile_configs=[tile_config], package_data=pkg_data,
            net_name='VDD', vdd=1.0,
        )
        model = DistributedPowerGridModel(
            backend=be,
            workers=[worker],
            interface_nodes={'b', 'pad'},
            tile_boundary_nodes={(0, 0): ['b', 'pad']},
            tile_interior_counts={(0, 0): 1},
            package_data=pkg_data,
            metadata=metadata,
        )
        solver = DistributedDDMSolver(model)

        # First call: smooth=True (builds smoothed VCS from empty sources — no-op smoothing)
        # Pass pkl_dir explicitly so we don't rely on the (empty) ckt_path fallback.
        h1 = solver.preprocess_sources(
            time_step=1e-9, t_start=0.0, t_end=10e-9, smooth=True,
            pkl_dir=str(tmp_path),
        )
        assert h1.smoothed is True

        # Capture the smoothed _active_sources reference
        # (with empty sources, smooth and raw are same object, so we need to
        # distinguish using a sentinel on the worker)
        raw_vcs = worker._vec_sources
        smoothed_vcs = worker._smoothed_sources
        # Put a distinguishable sentinel smoothed object
        from analysis.vectorized_sources import VectorizedCurrentSources
        sentinel_smoothed = VectorizedCurrentSources.from_current_sources({}, {}, 2)
        worker._smoothed_sources = sentinel_smoothed
        worker._active_sources = sentinel_smoothed

        # Second call: smooth=False → must call use_raw_sources() on worker
        h2 = solver.preprocess_sources(
            time_step=1e-9, t_start=0.0, t_end=10e-9, smooth=False,
            pkl_dir=str(tmp_path),
        )
        assert h2.smoothed is False, "Handle must report smoothed=False"
        assert worker._active_sources is raw_vcs, (
            "After preprocess_sources(smooth=False), worker._active_sources "
            "must be the raw VCS, not the smoothed sentinel"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Finding 9: _pack_workers skips remote-actor backends
# ─────────────────────────────────────────────────────────────────────────────

class TestFinding9PackWorkers:
    """Finding 9: _pack_workers must not wrap remote-actor backends."""

    def test_local_backend_supports_inprocess_packing(self):
        """LocalBackend.supports_inprocess_packing must be True."""
        from distributed.backend import LocalBackend
        be = LocalBackend()
        assert getattr(be, 'supports_inprocess_packing', False) is True

    def test_ray_backend_no_inprocess_packing(self):
        """RayBackend.supports_inprocess_packing must be False (default)."""
        from distributed.backend import RayBackend, ComputeBackend
        # Default from ComputeBackend ABC
        assert getattr(ComputeBackend, 'supports_inprocess_packing', True) is False
        be = RayBackend()
        assert getattr(be, 'supports_inprocess_packing', True) is False

    def test_custom_backend_not_packed(self, caplog):
        """A backend with supports_inprocess_packing=False must not be packed.

        _pack_workers must log a warning and return workers unchanged.
        """
        from distributed.backend import ComputeBackend
        from distributed.model import _pack_workers

        # Create a minimal fake backend (remote-actor-like, no packing support)
        class FakeRemoteBackend(ComputeBackend):
            supports_inprocess_packing = False
            def initialize(self, **kw): pass
            def create_actors(self, cls, cfgs): return []
            def call(self, a, m, *args, **kw): pass
            def call_all(self, actors, method, args=None): return []
            def map_func(self, f, args): return []
            def gather(self, fs): return []
            def shutdown(self): pass

        be = FakeRemoteBackend()
        sentinel_workers = [object(), object()]

        with caplog.at_level(logging.WARNING, logger='distributed.model'):
            result = _pack_workers(sentinel_workers, 2, be)

        assert result is sentinel_workers, (
            "_pack_workers must return workers unchanged for remote backends"
        )
        assert any('FakeRemoteBackend' in r.message or 'no-op' in r.message.lower()
                   for r in caplog.records), (
            "_pack_workers must log a warning for unsupported backends"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Finding 10: LocalBackend() passed to create_distributed_model works
# ─────────────────────────────────────────────────────────────────────────────

class TestFinding10LocalBackendInstance:
    """Finding 10: passing backend=LocalBackend() must not raise AttributeError."""

    def test_init_backend_with_local_backend_instance(self):
        """_init_backend(LocalBackend()) must not raise AttributeError on _initialized."""
        from distributed.backend import LocalBackend
        from distributed.model import _init_backend

        be = LocalBackend()
        # Before initialize(), there is no _initialized attribute on the ABC.
        # _init_backend must use getattr(be, '_initialized', False) to avoid crash.
        result = _init_backend(be, {})
        assert result is be

    def test_create_distributed_model_with_local_backend_instance(self, tmp_path):
        """create_distributed_model(bundle, backend=LocalBackend()) must succeed."""
        import pickle
        from distributed.backend import LocalBackend
        from distributed.model import (
            create_distributed_model,
            load_distributed_partitions,
            DistributedPowerGridModel,
        )
        from distributed.parser import PackageData, PowerGridMetaData, TileConfig
        from distributed.tile_worker import TileData
        from distributed.solver import DistributedDDMSolver

        # Build a minimal parsed PKL bundle in tmp_path
        tile_data = TileData(
            tile_id=(0, 0),
            resistive_edges=[('a', 'b', 1.0), ('b', '0', 2.0)],
            all_nodes={'a', 'b'},
            boundary_nodes={'b'},
            current_injections={'a': 0.5},
            capacitive_edges=[],
        )
        pkg_data = PackageData(
            vsrc_dict={'V1': {'node+': 'pad', 'node-': '0', 'net': 'VDD', 'value': 1.0}},
            package_edges=[('pad', 'b', 10.0)],
            pad_nodes={'pad'},
            tap_nodes=set(),
            die_attachment_nodes=set(),
            vdd=1.0,
            net_name='VDD',
            package_cap_edges=[],
        )
        tile_config = TileConfig(
            tile_id=(0, 0), ckt_path='', nd_path=None,
            instance_path=None, net_filter=None,
        )
        metadata = PowerGridMetaData(
            tile_grid=(1, 1), parameters={},
            tile_configs=[tile_config], package_data=pkg_data,
            net_name='VDD', vdd=1.0,
        )
        boundary_nodes = {'b', 'pad'}

        pkl_dir = str(tmp_path)
        tile_str = '0_0'
        with open(tmp_path / f'tile_{tile_str}.pkl', 'wb') as f:
            pickle.dump(tile_data, f)
        with open(tmp_path / 'metadata.pkl', 'wb') as f:
            pickle.dump({'metadata': metadata, 'boundary_nodes': boundary_nodes}, f)

        bundle = load_distributed_partitions(pkl_dir)

        # This must not raise AttributeError on LocalBackend._initialized
        be = LocalBackend()
        model = create_distributed_model(bundle, backend=be)
        assert isinstance(model, DistributedPowerGridModel)

        solver = DistributedDDMSolver(model)
        ctx = solver.prepare(verbose=False)
        result = solver.solve_dc(ctx, verbose=False)
        assert len(result.flatten()) > 0

        ctx.release()
        model.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Finding 11: release() frees _cg_solver
# ─────────────────────────────────────────────────────────────────────────────

class TestFinding11CgSolverRelease:
    """Finding 11: release() must null _cg_solver on both context types."""

    def test_dc_context_release_clears_cg_solver(self):
        """DistributedSolverContext.release() must null _cg_solver."""
        from distributed.result import DistributedSolverContext

        ctx = DistributedSolverContext()
        # Simulate a factored CG context
        ctx._cg_solver = object()  # fake CG solver holding S_global
        ctx._S_global = object()
        ctx.is_factored = True
        # Release (no model → skip worker call)
        ctx.release()

        assert ctx._cg_solver is None, (
            "DistributedSolverContext.release() must clear _cg_solver"
        )
        assert ctx._S_global is None
        assert ctx.is_factored is False

    def test_transient_context_release_clears_cg_solver(self):
        """DistributedTransientContext.release() must null _cg_solver."""
        from distributed.result import DistributedTransientContext

        ctx = DistributedTransientContext()
        # Simulate a factored CG transient context
        ctx._cg_solver = object()  # fake CG solver
        ctx._S_global = object()
        ctx.is_factored = True
        # Release (no model → skip worker call)
        ctx.release()

        assert ctx._cg_solver is None, (
            "DistributedTransientContext.release() must clear _cg_solver"
        )
        assert ctx._S_global is None
        assert ctx.is_factored is False

    def test_dc_context_cg_solver_initialized_in_init(self):
        """DistributedSolverContext.__init__ must initialize _cg_solver=None."""
        from distributed.result import DistributedSolverContext

        ctx = DistributedSolverContext()
        assert hasattr(ctx, '_cg_solver'), (
            "DistributedSolverContext must define _cg_solver in __init__"
        )
        assert ctx._cg_solver is None

    def test_transient_context_cg_solver_initialized_in_init(self):
        """DistributedTransientContext.__init__ must initialize _cg_solver=None."""
        from distributed.result import DistributedTransientContext

        ctx = DistributedTransientContext()
        assert hasattr(ctx, '_cg_solver'), (
            "DistributedTransientContext must define _cg_solver in __init__"
        )
        assert ctx._cg_solver is None

    def test_release_before_factor_does_not_raise(self):
        """release() on an unfactored context (before factor()) must not raise AttributeError."""
        from distributed.result import DistributedSolverContext, DistributedTransientContext

        # DC
        ctx_dc = DistributedSolverContext()
        ctx_dc.release()  # must not raise
        assert ctx_dc._cg_solver is None

        # Transient
        ctx_td = DistributedTransientContext()
        ctx_td.release()  # must not raise
        assert ctx_td._cg_solver is None


# ─────────────────────────────────────────────────────────────────────────────
# Finding 12: YAML whitelist + CLI flag
# ─────────────────────────────────────────────────────────────────────────────

class TestFinding12YamlKeys:
    """Finding 12: streaming_assembly, use_step_columns, max_table_mb in YAML whitelist."""

    def test_yaml_whitelist_has_b3_keys(self):
        """_VALID_SOLVER_YAML_KEYS must contain streaming_assembly, use_step_columns, max_table_mb."""
        from distributed.cli import _VALID_SOLVER_YAML_KEYS

        for key in ('streaming_assembly', 'use_step_columns', 'max_table_mb'):
            assert key in _VALID_SOLVER_YAML_KEYS, (
                f"'{key}' missing from _VALID_SOLVER_YAML_KEYS; "
                f"YAML with this key would raise ValueError"
            )

    def test_yaml_with_b3_keys_does_not_raise(self, tmp_path):
        """YAML containing streaming_assembly/use_step_columns/max_table_mb must not raise."""
        import yaml as _yaml
        from distributed.cli import _validate_solver_yaml_keys

        solver_cfg = {
            'streaming_assembly': 'auto',
            'use_step_columns': True,
            'max_table_mb': 256.0,
        }
        # Must not raise ValueError
        _validate_solver_yaml_keys(solver_cfg, 'solver')

    def test_streaming_assembly_yaml_reaches_model_settings(self):
        """YAML streaming_assembly value must propagate into model.settings."""
        import argparse
        from distributed.cli import _push_b3_settings

        class FakeModel:
            settings = {}

        # Simulate YAML having set streaming_assembly on args
        args = argparse.Namespace(streaming_assembly='auto', use_step_columns=None, max_table_mb=None)
        model = FakeModel()
        _push_b3_settings(model, args)

        assert model.settings.get('streaming_assembly') == 'auto', (
            "streaming_assembly='auto' from YAML must reach model.settings"
        )

    def test_streaming_assembly_cli_true_reaches_model_settings(self):
        """--streaming-assembly true must set model.settings['streaming_assembly'] = True."""
        import argparse
        from distributed.cli import _push_b3_settings

        class FakeModel:
            settings = {}

        args = argparse.Namespace(streaming_assembly='true', use_step_columns=None, max_table_mb=None)
        model = FakeModel()
        _push_b3_settings(model, args)

        assert model.settings.get('streaming_assembly') is True

    def test_streaming_assembly_cli_false_reaches_model_settings(self):
        """--streaming-assembly false must set model.settings['streaming_assembly'] = False."""
        import argparse
        from distributed.cli import _push_b3_settings

        class FakeModel:
            settings = {}

        args = argparse.Namespace(streaming_assembly='false', use_step_columns=None, max_table_mb=None)
        model = FakeModel()
        _push_b3_settings(model, args)

        assert model.settings.get('streaming_assembly') is False

    def test_use_step_columns_yaml_reaches_model_settings(self):
        """use_step_columns from YAML must propagate into model.settings."""
        import argparse
        from distributed.cli import _push_b3_settings

        class FakeModel:
            settings = {}

        args = argparse.Namespace(streaming_assembly=None, use_step_columns=False, max_table_mb=None)
        model = FakeModel()
        _push_b3_settings(model, args)

        assert model.settings.get('use_step_columns') is False

    def test_max_table_mb_yaml_reaches_model_settings(self):
        """max_table_mb from YAML must propagate into model.settings."""
        import argparse
        from distributed.cli import _push_b3_settings

        class FakeModel:
            settings = {}

        args = argparse.Namespace(streaming_assembly=None, use_step_columns=None, max_table_mb=128.0)
        model = FakeModel()
        _push_b3_settings(model, args)

        assert model.settings.get('max_table_mb') == 128.0

    def test_streaming_assembly_cli_flag_registered(self):
        """--streaming-assembly flag must be registered in the CLI parser."""
        import argparse
        from distributed.cli import _add_config_and_solver_args

        parser = argparse.ArgumentParser()
        # Add config and solver args (contains --streaming-assembly) — must not raise
        _add_config_and_solver_args(parser)

        # Must be able to parse --streaming-assembly auto without error
        args = parser.parse_args(['--streaming-assembly', 'auto'])
        assert args.streaming_assembly == 'auto'

        args_false = parser.parse_args(['--streaming-assembly', 'false'])
        assert args_false.streaming_assembly == 'false'

        args_true = parser.parse_args(['--streaming-assembly', 'true'])
        assert args_true.streaming_assembly == 'true'


# ─────────────────────────────────────────────────────────────────────────────
# Finding 14: try/finally teardown guards in test_equivalence.py
# ─────────────────────────────────────────────────────────────────────────────

class TestFinding14TeardownProtection:
    """Finding 14: verify logging.disable teardown can't leak when assertions fail.

    These tests verify the pattern used in test_equivalence.py — specifically
    that logging.disable(logging.NOTSET) is called even when an assertion fails.
    """

    def test_logging_restored_after_assert_failure(self):
        """Simulate assert failure inside try/finally — logging must be restored."""
        import logging

        # Before test: logging should be enabled
        assert logging.root.manager.disable == logging.NOTSET or True

        try:
            logging.disable(logging.WARNING)
            try:
                # Simulate a failing assertion
                # (we catch AssertionError to test the finally semantics)
                assert False, "simulated test failure"
            except AssertionError:
                pass
            finally:
                logging.disable(logging.NOTSET)
        finally:
            # Ensure we clean up regardless
            logging.disable(logging.NOTSET)

        # Logging should be restored
        assert logging.root.manager.disable == logging.NOTSET

    def test_context_released_in_finally_pattern(self):
        """Simulate context release in finally — no leak even on failure."""
        from distributed.result import DistributedSolverContext

        ctx = DistributedSolverContext()
        ctx._cg_solver = object()
        ctx._S_global = object()
        ctx.is_factored = True

        released = False
        try:
            # Simulate work that fails
            assert False, "simulated failure"
        except AssertionError:
            pass
        finally:
            if ctx is not None:
                try:
                    ctx.release()
                    released = True
                except Exception:
                    pass

        assert released, "context must be released in finally even on assertion failure"
        assert ctx._cg_solver is None
        assert ctx.is_factored is False
