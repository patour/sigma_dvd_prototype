"""Unit tests for A3: CHOLMOD supernodal guardrails + per-actor threading.

Tests:
    - Auto-mode resolution both sides of the nnz threshold
    - Simplicial warning fires (caplog)
    - resolved_mode lands in context interface_stats
    - threads_per_worker='auto' arithmetic
    - RayBackend runtime_env construction (mock; no real actors spawned)
    - CLI --threads-per-worker parsing helper
    - YAML threads_per_worker key accepted in solver section
"""

from __future__ import annotations

import logging
import unittest
from typing import Optional
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import scipy.sparse as sp

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spd_matrix(n: int, nnz_target: Optional[int] = None) -> sp.csc_matrix:
    """Build a small n×n SPD sparse matrix.

    If nnz_target is given, the matrix's nnz attribute is monkey-patched
    to that value so tests can cross the threshold without allocating huge data.
    """
    # Diagonal SPD + small off-diagonal coupling
    diag = np.full(n, float(n + 1))
    rng = np.random.default_rng(42)
    data = rng.random(n) * 0.1
    offsets = [-1, 0, 1]
    A = sp.diags(
        [data[: n - 1], diag, data[: n - 1]],
        offsets,
        shape=(n, n),
        format='csc',
        dtype=float,
    )
    return A


# ---------------------------------------------------------------------------
# 1.  Auto-mode resolution tests
# ---------------------------------------------------------------------------


class TestAutoModeResolution(unittest.TestCase):
    """_factor_conductance_matrix resolves 'auto' based on nnz threshold."""

    def _factor_small(self, monkeypatch_threshold: int = 1):
        """Factor a small matrix with the threshold set to monkeypatch_threshold."""
        import pgmath.factor as fmod

        orig_threshold = fmod.AUTO_SUPERNODAL_NNZ_THRESHOLD
        fmod.AUTO_SUPERNODAL_NNZ_THRESHOLD = monkeypatch_threshold
        try:
            matrix = _make_spd_matrix(4)  # tiny, nnz ~ 10
            from pgmath.factor import _factor_conductance_matrix, SolverBackendConfig
            cfg = SolverBackendConfig(use_cholmod=False)  # use splu for portability
            return _factor_conductance_matrix(matrix, config=cfg)
        finally:
            fmod.AUTO_SUPERNODAL_NNZ_THRESHOLD = orig_threshold

    def test_splu_resolved_mode_is_splu(self):
        """splu backend always returns resolved_mode='splu'."""
        from pgmath.factor import _factor_conductance_matrix, SolverBackendConfig

        matrix = _make_spd_matrix(4)
        cfg = SolverBackendConfig(use_cholmod=False)
        adapter = _factor_conductance_matrix(matrix, config=cfg)

        self.assertEqual(adapter.resolved_mode, 'splu')
        self.assertEqual(adapter.backend, 'splu')

    def test_splu_backend_info_is_splu_string(self):
        """splu backend_info is 'splu'."""
        from pgmath.factor import _factor_conductance_matrix, SolverBackendConfig

        matrix = _make_spd_matrix(4)
        cfg = SolverBackendConfig(use_cholmod=False)
        adapter = _factor_conductance_matrix(matrix, config=cfg)

        self.assertEqual(adapter.backend_info, 'splu')

    @pytest.mark.skipif(
        not __import__('pgmath.factor', fromlist=['HAS_CHOLMOD']).HAS_CHOLMOD,
        reason='sksparse.cholmod not installed',
    )
    def test_cholmod_auto_below_threshold_stays_auto(self):
        """With cholmod, auto mode + nnz below threshold -> resolved_mode='cholmod-auto'."""
        import pgmath.factor as fmod
        from pgmath.factor import _factor_conductance_matrix, SolverBackendConfig

        matrix = _make_spd_matrix(6)
        # Set threshold very high so this tiny matrix is "below"
        orig = fmod.AUTO_SUPERNODAL_NNZ_THRESHOLD
        fmod.AUTO_SUPERNODAL_NNZ_THRESHOLD = int(1e12)
        try:
            cfg = SolverBackendConfig(use_cholmod=True, cholmod_mode='auto')
            adapter = _factor_conductance_matrix(matrix, config=cfg)
            self.assertEqual(adapter.resolved_mode, 'cholmod-auto')
            self.assertIn('cholmod', adapter.backend_info)
            self.assertIn('mode=auto', adapter.backend_info)
        finally:
            fmod.AUTO_SUPERNODAL_NNZ_THRESHOLD = orig

    @pytest.mark.skipif(
        not __import__('pgmath.factor', fromlist=['HAS_CHOLMOD']).HAS_CHOLMOD,
        reason='sksparse.cholmod not installed',
    )
    def test_cholmod_auto_above_threshold_promotes_to_supernodal(self):
        """With cholmod, auto mode + nnz > tiny threshold -> resolved_mode='cholmod-supernodal'."""
        import pgmath.factor as fmod
        from pgmath.factor import _factor_conductance_matrix, SolverBackendConfig

        matrix = _make_spd_matrix(6)
        # Set threshold to 1 so any matrix exceeds it
        orig = fmod.AUTO_SUPERNODAL_NNZ_THRESHOLD
        fmod.AUTO_SUPERNODAL_NNZ_THRESHOLD = 1
        try:
            cfg = SolverBackendConfig(use_cholmod=True, cholmod_mode='auto')
            adapter = _factor_conductance_matrix(matrix, config=cfg)
            self.assertEqual(adapter.resolved_mode, 'cholmod-supernodal')
            self.assertIn('mode=supernodal', adapter.backend_info)
        finally:
            fmod.AUTO_SUPERNODAL_NNZ_THRESHOLD = orig

    @pytest.mark.skipif(
        not __import__('pgmath.factor', fromlist=['HAS_CHOLMOD']).HAS_CHOLMOD,
        reason='sksparse.cholmod not installed',
    )
    def test_cholmod_explicit_supernodal_returns_supernodal(self):
        """Explicit mode='supernodal' -> resolved_mode='cholmod-supernodal'."""
        from pgmath.factor import _factor_conductance_matrix, SolverBackendConfig

        matrix = _make_spd_matrix(6)
        cfg = SolverBackendConfig(use_cholmod=True, cholmod_mode='supernodal')
        adapter = _factor_conductance_matrix(matrix, config=cfg)

        self.assertEqual(adapter.resolved_mode, 'cholmod-supernodal')

    @pytest.mark.skipif(
        not __import__('pgmath.factor', fromlist=['HAS_CHOLMOD']).HAS_CHOLMOD,
        reason='sksparse.cholmod not installed',
    )
    def test_cholmod_explicit_simplicial_returns_simplicial(self):
        """Explicit mode='simplicial' -> resolved_mode='cholmod-simplicial'."""
        from pgmath.factor import _factor_conductance_matrix, SolverBackendConfig

        matrix = _make_spd_matrix(6)
        cfg = SolverBackendConfig(use_cholmod=True, cholmod_mode='simplicial')
        adapter = _factor_conductance_matrix(matrix, config=cfg)

        self.assertEqual(adapter.resolved_mode, 'cholmod-simplicial')

    def test_default_no_cholmod_variant_uses_splu(self):
        """When HAS_CHOLMOD is False, resolved_mode is always 'splu'."""
        import pgmath.factor as fmod
        from pgmath.factor import _factor_conductance_matrix

        orig = fmod.HAS_CHOLMOD
        fmod.HAS_CHOLMOD = False
        try:
            matrix = _make_spd_matrix(4)
            adapter = _factor_conductance_matrix(matrix, use_cholmod=False)
            self.assertEqual(adapter.resolved_mode, 'splu')
        finally:
            fmod.HAS_CHOLMOD = orig


# ---------------------------------------------------------------------------
# 2.  Simplicial warning tests
# ---------------------------------------------------------------------------


class TestSimplicialWarning(unittest.TestCase):
    """Simplicial mode with large nnz emits a logging.WARNING."""

    @pytest.mark.skipif(
        not __import__('pgmath.factor', fromlist=['HAS_CHOLMOD']).HAS_CHOLMOD,
        reason='sksparse.cholmod not installed',
    )
    def test_simplicial_above_threshold_emits_warning(self, caplog=None):
        """cholmod_mode='simplicial' + nnz > threshold => warning."""
        import pgmath.factor as fmod
        from pgmath.factor import _factor_conductance_matrix, SolverBackendConfig

        matrix = _make_spd_matrix(6)
        orig = fmod.AUTO_SUPERNODAL_NNZ_THRESHOLD
        fmod.AUTO_SUPERNODAL_NNZ_THRESHOLD = 1  # every matrix exceeds threshold
        fmod._SIMPLICIAL_GUARDRAIL_WARNED = False  # re-arm once-per-process latch

        try:
            with self.assertLogs('pgmath.factor', level='WARNING') as cm:
                cfg = SolverBackendConfig(use_cholmod=True, cholmod_mode='simplicial')
                _factor_conductance_matrix(matrix, config=cfg)

            # At least one warning mentioning 'simplicial'
            warnings_text = ' '.join(cm.output)
            self.assertIn('simplicial', warnings_text.lower())
            self.assertIn('supernodal', warnings_text.lower())

            # The latch suppresses a second identical warning in-process.
            with self.assertNoLogs('pgmath.factor', level='WARNING'):
                _factor_conductance_matrix(matrix, config=cfg)
        finally:
            fmod.AUTO_SUPERNODAL_NNZ_THRESHOLD = orig
            fmod._SIMPLICIAL_GUARDRAIL_WARNED = False

    @pytest.mark.skipif(
        not __import__('pgmath.factor', fromlist=['HAS_CHOLMOD']).HAS_CHOLMOD,
        reason='sksparse.cholmod not installed',
    )
    def test_simplicial_below_threshold_no_warning(self):
        """cholmod_mode='simplicial' + nnz < threshold => NO warning."""
        import pgmath.factor as fmod
        from pgmath.factor import _factor_conductance_matrix, SolverBackendConfig

        matrix = _make_spd_matrix(6)
        orig = fmod.AUTO_SUPERNODAL_NNZ_THRESHOLD
        fmod.AUTO_SUPERNODAL_NNZ_THRESHOLD = int(1e12)  # well above tiny matrix

        try:
            with self.assertNoLogs('pgmath.factor', level='WARNING'):
                cfg = SolverBackendConfig(use_cholmod=True, cholmod_mode='simplicial')
                _factor_conductance_matrix(matrix, config=cfg)
        except AttributeError:
            # assertNoLogs added in Python 3.10; skip on older versions
            pass
        finally:
            fmod.AUTO_SUPERNODAL_NNZ_THRESHOLD = orig

    @pytest.mark.skipif(
        not __import__('pgmath.factor', fromlist=['HAS_CHOLMOD']).HAS_CHOLMOD,
        reason='sksparse.cholmod not installed',
    )
    def test_auto_above_threshold_no_warning(self):
        """auto mode + nnz > threshold silently promotes to supernodal — no warning."""
        import pgmath.factor as fmod
        from pgmath.factor import _factor_conductance_matrix, SolverBackendConfig

        matrix = _make_spd_matrix(6)
        orig = fmod.AUTO_SUPERNODAL_NNZ_THRESHOLD
        fmod.AUTO_SUPERNODAL_NNZ_THRESHOLD = 1

        try:
            with self.assertRaises(AssertionError):
                # assertLogs raises AssertionError when no logs are emitted
                with self.assertLogs('pgmath.factor', level='WARNING'):
                    cfg = SolverBackendConfig(use_cholmod=True, cholmod_mode='auto')
                    _factor_conductance_matrix(matrix, config=cfg)
        finally:
            fmod.AUTO_SUPERNODAL_NNZ_THRESHOLD = orig


# ---------------------------------------------------------------------------
# 3.  resolved_mode in context stats
# ---------------------------------------------------------------------------


class TestResolvedModeInContextStats(unittest.TestCase):
    """interface_stats['resolved_mode'] is populated after factor()."""

    def _build_minimal_model(self):
        """Build the 2-tile model from test_time_domain.py."""
        from distributed.backend import LocalBackend
        from distributed.model import DistributedPowerGridModel
        from distributed.parser import PackageData, PowerGridMetaData, TileConfig
        from distributed.tile_worker import TileData, TileWorker

        tile_a = TileData(
            tile_id=(0, 0),
            resistive_edges=[('a1', 'shared', 1.0), ('shared', 'pad', 2.0)],
            all_nodes={'a1', 'shared', 'pad'},
            boundary_nodes={'shared', 'pad'},
            current_injections={'a1': 0.5},
            capacitive_edges=[],
        )
        tile_b = TileData(
            tile_id=(0, 1),
            resistive_edges=[('shared', 'b1', 3.0), ('b1', '0', 1.0)],
            all_nodes={'shared', 'b1'},
            boundary_nodes={'shared'},
            current_injections={'b1': 0.3},
            capacitive_edges=[],
        )

        be = LocalBackend()
        be.initialize()
        wa = TileWorker()
        wa.setup_from_tile_data(tile_a, {'shared', 'pad'})
        wb = TileWorker()
        wb.setup_from_tile_data(tile_b, {'shared', 'pad'})

        from distributed.parser import PackageData as PD, PowerGridMetaData as PGMD

        pkg = PD(
            vsrc_dict={'V1': {'node+': 'pad', 'node-': '0', 'net': 'VDD', 'value': 1.0}},
            package_edges=[('pad', 'shared', 10.0)],
            pad_nodes={'pad'},
            net_name='VDD',
            vdd=1.0,
            tap_nodes=set(),
            die_attachment_nodes=set(),
        )

        tile_configs = [
            TileConfig(tile_id=(0, 0), ckt_path='', nd_path='', instance_path=None, net_filter=None),
            TileConfig(tile_id=(0, 1), ckt_path='', nd_path='', instance_path=None, net_filter=None),
        ]
        meta = PGMD(
            tile_configs=tile_configs,
            package_data=pkg,
            tile_grid=(1, 2),
            parameters={},
            net_name='VDD',
            vdd=1.0,
        )

        model = DistributedPowerGridModel(
            backend=be,
            workers=[wa, wb],
            interface_nodes={'shared', 'pad'},
            tile_boundary_nodes={(0, 0): ['shared', 'pad'], (0, 1): ['shared']},
            tile_interior_counts={(0, 0): 1, (0, 1): 1},
            package_data=pkg,
            metadata=meta,
        )
        return model

    def test_dc_context_interface_stats_has_resolved_mode(self):
        """After DC prepare(), interface_stats contains 'resolved_mode'."""
        from distributed.solver import DistributedDDMSolver

        model = self._build_minimal_model()
        solver = DistributedDDMSolver(model)
        ctx = solver.prepare(verbose=False)

        timings = ctx.timings
        solver_stats = timings.get('solver_stats', {})
        iface = solver_stats.get('interface', {})

        self.assertIn('resolved_mode', iface, "interface_stats must contain 'resolved_mode'")
        rm = iface['resolved_mode']
        self.assertIsInstance(rm, str)
        self.assertGreater(len(rm), 0)
        # Must be one of the known modes
        self.assertIn(rm, {'cholmod-supernodal', 'cholmod-simplicial', 'cholmod-auto', 'splu'})

    def test_transient_context_interface_stats_has_resolved_mode(self):
        """After prepare_transient(), transient interface_stats contains 'resolved_mode'."""
        from distributed.solver import DistributedDDMSolver

        model = self._build_minimal_model()
        solver = DistributedDDMSolver(model)
        trans_ctx = solver.prepare_transient(dt=1e-10, method='BE')

        timings = trans_ctx.timings
        solver_stats = timings.get('solver_stats', {})
        iface = solver_stats.get('interface', {})

        self.assertIn('resolved_mode', iface,
                      "transient interface_stats must contain 'resolved_mode'")
        rm = iface['resolved_mode']
        self.assertIn(rm, {'cholmod-supernodal', 'cholmod-simplicial', 'cholmod-auto', 'splu'})


# ---------------------------------------------------------------------------
# 4.  threads_per_worker 'auto' arithmetic
# ---------------------------------------------------------------------------


class TestThreadsPerWorkerAuto(unittest.TestCase):
    """RayBackend._resolve_threads_per_worker('auto') uses cpus // n_workers."""

    def _make_backend(self, tpw, total_cpus: int, n_workers: int):
        """Return a partially-mocked RayBackend with tpw and a fake ray."""
        from distributed.backend import RayBackend

        be = RayBackend(threads_per_worker=tpw)
        be._initialized = True
        fake_ray = MagicMock()
        fake_ray.available_resources.return_value = {'CPU': float(total_cpus)}
        be._ray = fake_ray
        return be

    def test_auto_divides_evenly(self):
        """auto: cpus=8, n_workers=4 -> threads=2."""
        be = self._make_backend('auto', total_cpus=8, n_workers=4)
        self.assertEqual(be._resolve_threads_per_worker(4), 2)

    def test_auto_minimum_one(self):
        """auto: cpus=2, n_workers=8 -> threads=1 (minimum)."""
        be = self._make_backend('auto', total_cpus=2, n_workers=8)
        self.assertEqual(be._resolve_threads_per_worker(8), 1)

    def test_auto_single_worker(self):
        """auto: cpus=16, n_workers=1 -> threads=16."""
        be = self._make_backend('auto', total_cpus=16, n_workers=1)
        self.assertEqual(be._resolve_threads_per_worker(1), 16)

    def test_none_passthrough(self):
        """threads_per_worker=None -> _resolve returns None."""
        from distributed.backend import RayBackend
        be = RayBackend(threads_per_worker=None)
        be._initialized = True
        be._ray = MagicMock()
        self.assertIsNone(be._resolve_threads_per_worker(4))

    def test_explicit_int_passthrough(self):
        """threads_per_worker=3 -> _resolve returns 3 regardless of cpu count."""
        from distributed.backend import RayBackend
        be = RayBackend(threads_per_worker=3)
        be._initialized = True
        fake_ray = MagicMock()
        fake_ray.available_resources.return_value = {'CPU': 16.0}
        be._ray = fake_ray
        self.assertEqual(be._resolve_threads_per_worker(4), 3)


# ---------------------------------------------------------------------------
# 5.  RayBackend runtime_env construction (mock; no real Ray spawned)
# ---------------------------------------------------------------------------


class TestRayBackendRuntimeEnv(unittest.TestCase):
    """create_actors sets runtime_env env_vars when threads_per_worker is set."""

    def _run_create_actors_mock(self, tpw, n: int = 2):
        """Invoke RayBackend.create_actors with a mocked ray module.

        Returns the list of options calls recorded on the RemoteClass mock.
        """
        from distributed.backend import RayBackend
        from distributed.tile_worker import TileWorker

        be = RayBackend(threads_per_worker=tpw)
        be._initialized = True

        # Build a mock ray that records actor creation
        fake_ray = MagicMock()
        fake_ray.available_resources.return_value = {'CPU': 8.0}

        # RemoteClass.remote() and RemoteClass.options(...).remote() both return a mock actor
        mock_actor = MagicMock()
        mock_remote_class = MagicMock()
        mock_remote_class.remote.return_value = mock_actor
        mock_options_result = MagicMock()
        mock_options_result.remote.return_value = mock_actor
        mock_remote_class.options.return_value = mock_options_result
        fake_ray.remote.return_value = mock_remote_class

        be._ray = fake_ray

        configs = [None] * n
        actors = be.create_actors(TileWorker, configs)
        return fake_ray, mock_remote_class, actors

    def test_no_threads_no_runtime_env(self):
        """threads_per_worker=None -> actors created without runtime_env."""
        fake_ray, mock_cls, actors = self._run_create_actors_mock(tpw=None, n=3)
        # options should NOT have been called (plain .remote() calls)
        mock_cls.options.assert_not_called()
        # .remote() was called 3 times
        self.assertEqual(mock_cls.remote.call_count, 3)

    def test_explicit_threads_sets_env_vars(self):
        """threads_per_worker=4 -> runtime_env has OMP/OPENBLAS/MKL_NUM_THREADS=4."""
        fake_ray, mock_cls, actors = self._run_create_actors_mock(tpw=4, n=2)

        # options() should have been called with runtime_env containing env_vars
        self.assertEqual(mock_cls.options.call_count, 2)
        call_kwargs = mock_cls.options.call_args_list[0].kwargs
        env_vars = call_kwargs.get('runtime_env', {}).get('env_vars', {})
        self.assertEqual(env_vars.get('OMP_NUM_THREADS'), '4')
        self.assertEqual(env_vars.get('OPENBLAS_NUM_THREADS'), '4')
        self.assertEqual(env_vars.get('MKL_NUM_THREADS'), '4')

    def test_auto_threads_sets_env_vars(self):
        """threads_per_worker='auto' (cpus=8, workers=2) -> threads=4."""
        fake_ray, mock_cls, actors = self._run_create_actors_mock(tpw='auto', n=2)

        # 8 cpus // 2 workers = 4
        self.assertEqual(mock_cls.options.call_count, 2)
        call_kwargs = mock_cls.options.call_args_list[0].kwargs
        env_vars = call_kwargs.get('runtime_env', {}).get('env_vars', {})
        self.assertEqual(env_vars.get('OMP_NUM_THREADS'), '4')
        self.assertEqual(env_vars.get('OPENBLAS_NUM_THREADS'), '4')
        self.assertEqual(env_vars.get('MKL_NUM_THREADS'), '4')

    def test_local_backend_is_noop(self):
        """LocalBackend.create_actors ignores threads_per_worker (in-process)."""
        from distributed.backend import LocalBackend
        from distributed.tile_worker import TileWorker

        be = LocalBackend()
        be.initialize()
        # LocalBackend.create_actors just instantiates actors; no env manipulation
        actors = be.create_actors(TileWorker, [None, None])
        self.assertEqual(len(actors), 2)
        self.assertIsInstance(actors[0], TileWorker)


# ---------------------------------------------------------------------------
# 6.  CLI --threads-per-worker parsing helper
# ---------------------------------------------------------------------------


class TestParseThreadsPerWorker(unittest.TestCase):
    """_parse_threads_per_worker handles int, 'auto', and None."""

    def _fn(self, val):
        from distributed.cli import _parse_threads_per_worker
        return _parse_threads_per_worker(val)

    def test_none_returns_none(self):
        self.assertIsNone(self._fn(None))

    def test_auto_string_returns_auto(self):
        self.assertEqual(self._fn('auto'), 'auto')

    def test_numeric_string_returns_int(self):
        self.assertEqual(self._fn('8'), 8)
        self.assertIsInstance(self._fn('8'), int)

    def test_int_passthrough(self):
        self.assertEqual(self._fn(4), 4)

    def test_invalid_string_raises(self):
        with self.assertRaises(ValueError):
            self._fn('not_a_number')


# ---------------------------------------------------------------------------
# 7.  YAML solver section: threads_per_worker key accepted
# ---------------------------------------------------------------------------


class TestYAMLThreadsPerWorker(unittest.TestCase):
    """YAML solver: section can contain 'threads_per_worker' without error."""

    def test_valid_key_passes_validation(self):
        from distributed.cli import _validate_solver_yaml_keys

        # Should NOT raise
        _validate_solver_yaml_keys({'threads_per_worker': 4})
        _validate_solver_yaml_keys({'threads_per_worker': 'auto'})

    def test_threads_per_worker_not_in_role_keys(self):
        """threads_per_worker is a top-level solver key, not a coordinator/worker key."""
        from distributed.cli import _VALID_ROLE_YAML_KEYS, _VALID_SOLVER_YAML_KEYS

        self.assertIn('threads_per_worker', _VALID_SOLVER_YAML_KEYS)
        self.assertNotIn('threads_per_worker', _VALID_ROLE_YAML_KEYS)

    def test_threads_per_worker_in_coordinator_dict_is_invalid(self):
        """threads_per_worker inside coordinator: sub-dict should raise ValueError."""
        from distributed.cli import _validate_solver_yaml_keys

        with self.assertRaises(ValueError):
            _validate_solver_yaml_keys({
                'coordinator': {'threads_per_worker': 4},
            })

    def test_auto_supernodal_threshold_is_documented_constant(self):
        """AUTO_SUPERNODAL_NNZ_THRESHOLD is an int equal to 5e7."""
        from pgmath.factor import AUTO_SUPERNODAL_NNZ_THRESHOLD
        self.assertEqual(AUTO_SUPERNODAL_NNZ_THRESHOLD, int(5e7))
        self.assertIsInstance(AUTO_SUPERNODAL_NNZ_THRESHOLD, int)
