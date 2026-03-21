"""Unit tests for distributed CLI argument parsing and mode dispatch.

Tests that the argument parser correctly handles the time-domain flags
and that the mode dispatch logic selects the right code path.
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from distributed.cli import build_parser

pytestmark = pytest.mark.unit


class TestBuildParser:
    """Tests for build_parser() argument parsing."""

    def test_solve_defaults(self):
        """Default solve args should match DC mode with standard defaults."""
        parser = build_parser()
        args = parser.parse_args(['solve', '/tmp/pkl'])
        assert args.mode == 'dc'
        assert args.t_start == 0.0
        assert args.t_end == 100e-9
        assert args.dt == 0.1e-9
        assert args.n_points == 101
        assert args.method == 'be'
        assert args.smooth is True

    def test_solve_mode_dc(self):
        """Explicit --mode dc should be accepted."""
        parser = build_parser()
        args = parser.parse_args(['solve', '/tmp/pkl', '--mode', 'dc'])
        assert args.mode == 'dc'

    def test_solve_mode_quasi_static(self):
        """--mode quasi-static should be accepted with time params."""
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl',
            '--mode', 'quasi-static',
            '--t-start', '1e-9',
            '--t-end', '50e-9',
            '--n-points', '51',
        ])
        assert args.mode == 'quasi-static'
        assert args.t_start == 1e-9
        assert args.t_end == 50e-9
        assert args.n_points == 51

    def test_solve_mode_transient(self):
        """--mode transient should be accepted with time params."""
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl',
            '--mode', 'transient',
            '--t-end', '200e-9',
            '--dt', '0.05e-9',
            '--method', 'trap',
        ])
        assert args.mode == 'transient'
        assert args.t_end == 200e-9
        assert args.dt == 0.05e-9
        assert args.method == 'trap'

    def test_solve_invalid_mode(self):
        """Invalid mode should be rejected by argparse."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['solve', '/tmp/pkl', '--mode', 'invalid'])

    def test_solve_invalid_method(self):
        """Invalid integration method should be rejected by argparse."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['solve', '/tmp/pkl', '--method', 'rk4'])

    def test_solve_no_smooth(self):
        """--no-smooth should disable smoothing."""
        parser = build_parser()
        args = parser.parse_args(['solve', '/tmp/pkl', '--no-smooth'])
        assert args.smooth is False

    def test_solve_smooth_default(self):
        """Smooth should be True by default."""
        parser = build_parser()
        args = parser.parse_args(['solve', '/tmp/pkl'])
        assert args.smooth is True

    def test_run_has_time_domain_args(self):
        """The run subcommand should also have time-domain args."""
        parser = build_parser()
        args = parser.parse_args([
            'run', '/tmp/netlist',
            '--mode', 'transient',
            '--dt', '0.2e-9',
            '--method', 'be',
        ])
        assert args.mode == 'transient'
        assert args.dt == 0.2e-9
        assert args.method == 'be'

    def test_solve_existing_args_preserved(self):
        """Existing solve args (backend, plot, etc.) should still work."""
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl',
            '--backend', 'local',
            '--plot',
            '--verbose',
            '--top-k', '50',
        ])
        assert args.backend == 'local'
        assert args.plot is True
        assert args.verbose is True
        assert args.top_k == 50
        # Time-domain defaults should be present alongside
        assert args.mode == 'dc'

    def test_parse_subcommand_unaffected(self):
        """The parse subcommand should NOT have time-domain args."""
        parser = build_parser()
        args = parser.parse_args(['parse', '/tmp/netlist'])
        assert not hasattr(args, 'mode')
        assert not hasattr(args, 'dt')


class TestSolveDispatch:
    """Tests that cmd_solve dispatches to the correct mode handler."""

    @patch('distributed.cli._solve_dc')
    @patch('distributed.solver.DistributedDDMSolver', autospec=True)
    @patch('distributed.model.create_distributed_model')
    @patch('distributed.model.load_distributed_partitions')
    @patch('distributed.cli._load_and_apply_config', side_effect=lambda a: a)
    @patch('distributed.cli._setup_logging')
    def test_dc_mode_calls_solve_dc(
        self, mock_log, mock_config, mock_load, mock_create,
        mock_solver_cls, mock_solve_dc,
    ):
        """cmd_solve with mode=dc should call _solve_dc."""
        from distributed.cli import cmd_solve

        mock_model = MagicMock()
        mock_create.return_value = mock_model
        mock_solver = mock_solver_cls.return_value
        mock_ctx = MagicMock()
        mock_solver.prepare.return_value = mock_ctx

        args = argparse.Namespace(
            pkl_dir='/tmp/pkl', backend='local', verbose=False,
            mode='dc', plot=False, output=None,
        )

        cmd_solve(args)
        mock_solve_dc.assert_called_once()
        mock_model.shutdown.assert_called_once()

    @patch('distributed.cli._solve_quasi_static')
    @patch('distributed.solver.DistributedDDMSolver', autospec=True)
    @patch('distributed.model.create_distributed_model')
    @patch('distributed.model.load_distributed_partitions')
    @patch('distributed.cli._load_and_apply_config', side_effect=lambda a: a)
    @patch('distributed.cli._setup_logging')
    def test_quasi_static_mode_dispatch(
        self, mock_log, mock_config, mock_load, mock_create,
        mock_solver_cls, mock_qs,
    ):
        """cmd_solve with mode=quasi-static should call _solve_quasi_static."""
        from distributed.cli import cmd_solve

        mock_model = MagicMock()
        mock_create.return_value = mock_model
        mock_solver = mock_solver_cls.return_value
        mock_ctx = MagicMock()
        mock_solver.prepare.return_value = mock_ctx

        args = argparse.Namespace(
            pkl_dir='/tmp/pkl', backend='local', verbose=False,
            mode='quasi-static', plot=False, output=None,
            t_start=0.0, t_end=100e-9, n_points=101,
        )

        cmd_solve(args)
        mock_qs.assert_called_once()
        mock_model.shutdown.assert_called_once()

    @patch('distributed.cli._solve_transient')
    @patch('distributed.solver.DistributedDDMSolver', autospec=True)
    @patch('distributed.model.create_distributed_model')
    @patch('distributed.model.load_distributed_partitions')
    @patch('distributed.cli._load_and_apply_config', side_effect=lambda a: a)
    @patch('distributed.cli._setup_logging')
    def test_transient_mode_dispatch(
        self, mock_log, mock_config, mock_load, mock_create,
        mock_solver_cls, mock_trans,
    ):
        """cmd_solve with mode=transient should call _solve_transient."""
        from distributed.cli import cmd_solve

        mock_model = MagicMock()
        mock_create.return_value = mock_model
        mock_solver = mock_solver_cls.return_value
        mock_ctx = MagicMock()
        mock_solver.prepare.return_value = mock_ctx

        args = argparse.Namespace(
            pkl_dir='/tmp/pkl', backend='local', verbose=False,
            mode='transient', plot=False, output=None,
            t_start=0.0, t_end=100e-9, dt=0.1e-9,
            method='be',
        )

        cmd_solve(args)
        mock_trans.assert_called_once()
        mock_model.shutdown.assert_called_once()

    @patch('distributed.solver.DistributedDDMSolver', autospec=True)
    @patch('distributed.model.create_distributed_model')
    @patch('distributed.model.load_distributed_partitions')
    @patch('distributed.cli._load_and_apply_config', side_effect=lambda a: a)
    @patch('distributed.cli._setup_logging')
    def test_model_shutdown_on_error(
        self, mock_log, mock_config, mock_load, mock_create,
        mock_solver_cls,
    ):
        """model.shutdown() should be called even if solve raises."""
        from distributed.cli import cmd_solve

        mock_model = MagicMock()
        mock_create.return_value = mock_model
        mock_solver = mock_solver_cls.return_value
        mock_solver.prepare.side_effect = RuntimeError("test error")

        args = argparse.Namespace(
            pkl_dir='/tmp/pkl', backend='local', verbose=False,
            mode='dc', plot=False, output=None,
        )

        with pytest.raises(RuntimeError, match="test error"):
            cmd_solve(args)
        mock_model.shutdown.assert_called_once()


class TestTimeDomainReport:
    """Tests for _report_time_domain_result helper."""

    def test_report_logs_summary(self, caplog):
        """Time-domain report should log peak IR-drop and timing."""
        import time
        import numpy as np
        from distributed.cli import _report_time_domain_result

        mock_result = MagicMock()
        mock_result.t_array = np.linspace(0, 100e-9, 101)
        mock_result.peak_ir_drop = 0.005  # 5 mV
        mock_result.peak_ir_drop_time = 50e-9

        args = argparse.Namespace(output=None, plot=False)

        import logging
        with caplog.at_level(logging.INFO):
            _report_time_domain_result(mock_result, args, time.perf_counter(), 'quasi-static')

        assert 'quasi-static' in caplog.text
        assert '5.000 mV' in caplog.text

    def test_report_plot_without_solver_warns(self, caplog):
        """With --plot but no solver, should log a warning."""
        import time
        import numpy as np
        from distributed.cli import _report_time_domain_result

        mock_result = MagicMock()
        mock_result.t_array = np.linspace(0, 100e-9, 10)
        mock_result.peak_ir_drop = 0.001
        mock_result.peak_ir_drop_time = 10e-9

        args = argparse.Namespace(output=None, plot=True)

        import logging
        with caplog.at_level(logging.WARNING):
            _report_time_domain_result(mock_result, args, time.perf_counter(), 'transient')

        assert 'solver reference not available' in caplog.text

    def test_report_plot_calls_generate_td_reports(self):
        """With --plot and solver, should call solver.generate_td_reports()."""
        import time
        import numpy as np
        from distributed.cli import _report_time_domain_result

        mock_result = MagicMock()
        mock_result.t_array = np.linspace(0, 100e-9, 10)
        mock_result.peak_ir_drop = 0.001
        mock_result.peak_ir_drop_time = 10e-9

        mock_solver = MagicMock()

        args = argparse.Namespace(
            output='./results', plot=True,
            plot_layers='M1,M2', max_stripes=500,
            stripe_bin_size=100, top_k=50, verbose=True,
        )

        _report_time_domain_result(
            mock_result, args, time.perf_counter(), 'transient',
            solver=mock_solver,
        )

        mock_solver.generate_td_reports.assert_called_once_with(
            mock_result,
            output_dir='./results',
            plot_layers=['M1', 'M2'],
            max_stripes=500,
            stripe_bin_size=100,
            top_k=50,
            verbose=True,
        )

    def test_report_saves_result(self, tmp_path):
        """With --output, should call result.dump()."""
        import time
        import numpy as np
        from distributed.cli import _report_time_domain_result

        mock_result = MagicMock()
        mock_result.t_array = np.linspace(0, 100e-9, 10)
        mock_result.peak_ir_drop = 0.002
        mock_result.peak_ir_drop_time = 20e-9

        args = argparse.Namespace(output=str(tmp_path), plot=False)

        _report_time_domain_result(mock_result, args, time.perf_counter(), 'quasi-static')

        mock_result.dump.assert_called_once()
        call_path = mock_result.dump.call_args[0][0]
        assert 'quasi_static' in call_path
