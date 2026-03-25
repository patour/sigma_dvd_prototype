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


class TestDecomposeParser:
    """Tests for decompose subcommand argument parsing."""

    def test_decompose_defaults(self):
        """Default decompose args should have sensible defaults."""
        parser = build_parser()
        args = parser.parse_args(['decompose', '/tmp/pkl'])
        assert args.command == 'decompose'
        assert args.netlist_dir == '/tmp/pkl'
        assert args.backend == 'local'
        assert args.output == './irdrop_decomp_results'
        assert args.verbose is False
        assert args.no_plot is False
        assert args.top_k == 5  # Overridden default for decompose
        assert args.window_percent == 10.0
        assert args.instances is None
        assert args.aggressor_top_k == 0
        assert args.adjoint_method == 'dynamic'
        assert args.adjoint_memory_window == 20
        assert args.plot_layers is None
        assert args.max_stripes == 500

    def test_decompose_has_time_domain_args(self):
        """Decompose should have relevant time-domain args only."""
        parser = build_parser()
        args = parser.parse_args([
            'decompose', '/tmp/pkl',
            '--t-end', '50e-9',
            '--dt', '0.05e-9',
            '--method', 'trap',
        ])
        assert args.t_end == 50e-9
        assert args.dt == 0.05e-9
        assert args.method == 'trap'
        assert args.t_start == 0.0
        # Smooth defaults to None (sentinel) for decompose; resolved to
        # True later in cmd_decompose when no config overrides it.
        assert args.smooth is None

    def test_decompose_rejects_mode_and_npoints(self):
        """Decompose should NOT accept --mode or --n-points (H1)."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['decompose', '/tmp/pkl', '--mode', 'dc'])
        with pytest.raises(SystemExit):
            parser.parse_args(['decompose', '/tmp/pkl', '--n-points', '51'])

    def test_decompose_no_smooth(self):
        """--no-smooth should disable source smoothing."""
        parser = build_parser()
        args = parser.parse_args(['decompose', '/tmp/pkl', '--no-smooth'])
        assert args.smooth is False

    def test_decompose_custom_window(self):
        """--window-percent should accept custom values."""
        parser = build_parser()
        args = parser.parse_args([
            'decompose', '/tmp/pkl', '--window-percent', '15.5',
        ])
        assert args.window_percent == 15.5

    def test_decompose_instances(self):
        """--instances should accept comma-separated node names."""
        parser = build_parser()
        args = parser.parse_args([
            'decompose', '/tmp/pkl',
            '--instances', '1000_2000_M1,3000_4000_M1',
        ])
        assert args.instances == '1000_2000_M1,3000_4000_M1'

    def test_decompose_aggressor_args(self):
        """Aggressor analysis args should be parsed correctly."""
        parser = build_parser()
        args = parser.parse_args([
            'decompose', '/tmp/pkl',
            '--aggressor-top-k', '10',
            '--adjoint-method', 'static',
            '--adjoint-memory-window', '30',
        ])
        assert args.aggressor_top_k == 10
        assert args.adjoint_method == 'static'
        assert args.adjoint_memory_window == 30

    def test_decompose_invalid_adjoint_method(self):
        """Invalid adjoint method should be rejected."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                'decompose', '/tmp/pkl', '--adjoint-method', 'invalid',
            ])

    def test_decompose_plot_args(self):
        """Plot-related args should be parsed correctly."""
        parser = build_parser()
        args = parser.parse_args([
            'decompose', '/tmp/pkl',
            '--plot-layers', 'M1,M2',
            '--max-stripes', '1000',
            '--no-plot',
        ])
        assert args.plot_layers == 'M1,M2'
        assert args.max_stripes == 1000
        assert args.no_plot is True

    def test_decompose_output_dir(self):
        """--output should override default output directory."""
        parser = build_parser()
        args = parser.parse_args([
            'decompose', '/tmp/pkl', '-o', '/my/output',
        ])
        assert args.output == '/my/output'

    def test_decompose_top_k_override(self):
        """--top-k should override the decompose-specific default of 5."""
        parser = build_parser()
        args = parser.parse_args([
            'decompose', '/tmp/pkl', '--top-k', '20',
        ])
        assert args.top_k == 20

    def test_decompose_has_config_args(self):
        """Decompose should have solver config args from shared group."""
        parser = build_parser()
        args = parser.parse_args([
            'decompose', '/tmp/pkl', '--use-cholmod',
        ])
        assert args.use_cholmod is True

    def test_decompose_func_set(self):
        """Decompose subcommand should have cmd_decompose as its handler."""
        from distributed.cli import cmd_decompose
        parser = build_parser()
        args = parser.parse_args(['decompose', '/tmp/pkl'])
        assert args.func is cmd_decompose


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


class TestDecomposeDispatch:
    """Tests that cmd_decompose calls the right functions."""

    @patch('analysis.dynamic_irdrop_decomposition.generate_plots')
    @patch('analysis.dynamic_irdrop_decomposition.print_results')
    @patch('analysis.dynamic_irdrop_decomposition.Logger')
    @patch('distributed.decomposition.analyze_distributed_decomposition')
    @patch('distributed.cli._load_and_apply_config', side_effect=lambda a: a)
    @patch('distributed.cli._setup_logging')
    def test_decompose_calls_analyze(
        self, mock_log, mock_config, mock_analyze,
        mock_logger_cls, mock_print, mock_gen_plots, tmp_path,
    ):
        """cmd_decompose should call analyze_distributed_decomposition."""
        from distributed.cli import cmd_decompose

        mock_result = MagicMock()
        mock_result.worst_instances = [MagicMock()]
        mock_solver = MagicMock()
        mock_model = MagicMock()
        mock_analyze.return_value = (mock_result, mock_solver, mock_model)

        # Mock the Logger context manager
        mock_logger_inst = MagicMock()
        mock_logger_cls.return_value = mock_logger_inst

        # Create fake tile pkl so the glob check in cmd_decompose passes
        pkl_subdir = tmp_path / 'distributed_pkl'
        pkl_subdir.mkdir()
        (pkl_subdir / 'tile_0_0.pkl').touch()

        args = argparse.Namespace(
            netlist_dir=str(tmp_path), net=None,
            backend='local', verbose=False,
            output=str(tmp_path / 'out'), no_plot=True,
            t_start=0.0, t_end=100e-9, dt=0.1e-9,
            top_k=5, window_percent=10.0, instances=None,
            method='be', smooth=True,
            aggressor_top_k=0, adjoint_method='dynamic',
            adjoint_memory_window=20,
            plot_layers=None, max_stripes=500,
        )

        cmd_decompose(args)

        mock_analyze.assert_called_once_with(
            pkl_dir=str(pkl_subdir),
            backend='local',
            t_start=0.0,
            t_end=100e-9,
            dt=0.1e-9,
            top_k=5,
            window_percent=10.0,
            integration_method='be',
            instances=None,
            smooth_sources=True,
            aggressor_top_k=0,
            adjoint_method='dynamic',
            adjoint_memory_window=20,
            verbose=False,
        )
        mock_print.assert_called_once()
        mock_result.save_json.assert_called_once()
        mock_model.shutdown.assert_called_once()

    @patch('analysis.dynamic_irdrop_decomposition.generate_plots')
    @patch('analysis.dynamic_irdrop_decomposition.print_results')
    @patch('analysis.dynamic_irdrop_decomposition.Logger')
    @patch('distributed.decomposition.analyze_distributed_decomposition')
    @patch('distributed.cli._load_and_apply_config', side_effect=lambda a: a)
    @patch('distributed.cli._setup_logging')
    def test_decompose_parses_instances(
        self, mock_log, mock_config, mock_analyze,
        mock_logger_cls, mock_print, mock_gen_plots, tmp_path,
    ):
        """--instances should be split and passed as list."""
        from distributed.cli import cmd_decompose

        mock_result = MagicMock()
        mock_result.worst_instances = []
        mock_analyze.return_value = (mock_result, MagicMock(), MagicMock())
        mock_logger_cls.return_value = MagicMock()

        # Create fake tile pkl so the glob check in cmd_decompose passes
        pkl_subdir = tmp_path / 'distributed_pkl'
        pkl_subdir.mkdir()
        (pkl_subdir / 'tile_0_0.pkl').touch()

        args = argparse.Namespace(
            netlist_dir=str(tmp_path), net=None,
            backend='local', verbose=False,
            output=str(tmp_path / 'out'), no_plot=True,
            t_start=0.0, t_end=100e-9, dt=0.1e-9,
            top_k=5, window_percent=10.0,
            instances='1000_2000_M1, 3000_4000_M1',
            method='be', smooth=True,
            aggressor_top_k=0, adjoint_method='dynamic',
            adjoint_memory_window=20,
            plot_layers=None, max_stripes=500,
        )

        cmd_decompose(args)

        call_kwargs = mock_analyze.call_args[1]
        assert call_kwargs['instances'] == ['1000_2000_M1', '3000_4000_M1']

    @patch('analysis.dynamic_irdrop_decomposition.generate_plots')
    @patch('analysis.dynamic_irdrop_decomposition.print_results')
    @patch('analysis.dynamic_irdrop_decomposition.Logger')
    @patch('distributed.decomposition.analyze_distributed_decomposition')
    @patch('distributed.cli._load_and_apply_config', side_effect=lambda a: a)
    @patch('distributed.cli._setup_logging')
    def test_decompose_generates_plots_when_enabled(
        self, mock_log, mock_config, mock_analyze,
        mock_logger_cls, mock_print, mock_gen_plots, tmp_path,
    ):
        """With --no-plot=False, should call generate_plots."""
        from distributed.cli import cmd_decompose

        mock_result = MagicMock()
        mock_result.worst_instances = []
        mock_analyze.return_value = (mock_result, MagicMock(), MagicMock())
        mock_logger_cls.return_value = MagicMock()

        # Create fake tile pkl so the glob check in cmd_decompose passes
        pkl_subdir = tmp_path / 'distributed_pkl'
        pkl_subdir.mkdir()
        (pkl_subdir / 'tile_0_0.pkl').touch()

        args = argparse.Namespace(
            netlist_dir=str(tmp_path), net=None,
            backend='local', verbose=True,
            output=str(tmp_path / 'out'), no_plot=False,
            t_start=0.0, t_end=100e-9, dt=0.1e-9,
            top_k=5, window_percent=10.0, instances=None,
            method='be', smooth=True,
            aggressor_top_k=0, adjoint_method='dynamic',
            adjoint_memory_window=20,
            plot_layers='M1,M2', max_stripes=800,
        )

        cmd_decompose(args)

        mock_gen_plots.assert_called_once()
        call_kwargs = mock_gen_plots.call_args[1]
        assert call_kwargs['show'] is False
        assert call_kwargs['heatmap_layers'] == ['M1', 'M2']
        assert call_kwargs['max_stripes'] == 800

    @patch('distributed.decomposition.analyze_distributed_decomposition')
    @patch('distributed.cli._load_and_apply_config', side_effect=lambda a: a)
    @patch('distributed.cli._setup_logging')
    def test_decompose_model_shutdown_on_analyze_error(
        self, mock_log, mock_config, mock_analyze, tmp_path,
    ):
        """When analyze raises before returning, cmd_decompose should not
        crash on shutdown (model is None).  B1 ensures the model is shut
        down inside analyze_distributed_decomposition itself."""
        from distributed.cli import cmd_decompose

        mock_analyze.side_effect = RuntimeError("analysis failed")

        # Create fake tile pkl so the glob check in cmd_decompose passes
        pkl_subdir = tmp_path / 'distributed_pkl'
        pkl_subdir.mkdir()
        (pkl_subdir / 'tile_0_0.pkl').touch()

        args = argparse.Namespace(
            netlist_dir=str(tmp_path), net=None,
            backend='local', verbose=False,
            output=str(tmp_path / 'out'), no_plot=True,
            t_start=0.0, t_end=100e-9, dt=0.1e-9,
            top_k=5, window_percent=10.0, instances=None,
            method='be', smooth=True,
            aggressor_top_k=0, adjoint_method='dynamic',
            adjoint_memory_window=20,
            plot_layers=None, max_stripes=500,
        )

        with pytest.raises(RuntimeError, match="analysis failed"):
            cmd_decompose(args)
        # model stays None in cmd_decompose because analyze raised
        # before returning -- the finally block correctly skips shutdown.

    @patch('analysis.dynamic_irdrop_decomposition.generate_plots')
    @patch('analysis.dynamic_irdrop_decomposition.print_results')
    @patch('analysis.dynamic_irdrop_decomposition.Logger')
    @patch('distributed.decomposition.analyze_distributed_decomposition')
    @patch('distributed.cli._load_and_apply_config', side_effect=lambda a: a)
    @patch('distributed.cli._setup_logging')
    def test_decompose_model_shutdown_on_post_analyze_error(
        self, mock_log, mock_config, mock_analyze,
        mock_logger_cls, mock_print, mock_gen_plots, tmp_path,
    ):
        """model.shutdown() should be called even if post-analysis
        processing (print_results, save_json, generate_plots) raises."""
        from distributed.cli import cmd_decompose

        mock_result = MagicMock()
        mock_result.worst_instances = [MagicMock()]
        mock_solver = MagicMock()
        mock_model = MagicMock()
        mock_analyze.return_value = (mock_result, mock_solver, mock_model)

        mock_logger_inst = MagicMock()
        mock_logger_cls.return_value = mock_logger_inst

        # Make print_results raise after analyze succeeds
        mock_print.side_effect = RuntimeError("print failed")

        # Create fake tile pkl so the glob check in cmd_decompose passes
        pkl_subdir = tmp_path / 'distributed_pkl'
        pkl_subdir.mkdir()
        (pkl_subdir / 'tile_0_0.pkl').touch()

        args = argparse.Namespace(
            netlist_dir=str(tmp_path), net=None,
            backend='local', verbose=False,
            output=str(tmp_path / 'out'), no_plot=True,
            t_start=0.0, t_end=100e-9, dt=0.1e-9,
            top_k=5, window_percent=10.0, instances=None,
            method='be', smooth=True,
            aggressor_top_k=0, adjoint_method='dynamic',
            adjoint_memory_window=20,
            plot_layers=None, max_stripes=500,
        )

        with pytest.raises(RuntimeError, match="print failed"):
            cmd_decompose(args)
        mock_model.shutdown.assert_called_once()

    @patch('analysis.dynamic_irdrop_decomposition.generate_plots')
    @patch('analysis.dynamic_irdrop_decomposition.print_results')
    @patch('analysis.dynamic_irdrop_decomposition.Logger')
    @patch('distributed.decomposition.analyze_distributed_decomposition')
    @patch('distributed.cli._load_and_apply_config', side_effect=lambda a: a)
    @patch('distributed.cli._setup_logging')
    def test_decompose_skips_plots_with_no_plot(
        self, mock_log, mock_config, mock_analyze,
        mock_logger_cls, mock_print, mock_gen_plots, tmp_path,
    ):
        """With --no-plot, should NOT call generate_plots."""
        from distributed.cli import cmd_decompose

        mock_result = MagicMock()
        mock_result.worst_instances = []
        mock_analyze.return_value = (mock_result, MagicMock(), MagicMock())
        mock_logger_cls.return_value = MagicMock()

        # Create fake tile pkl so the glob check in cmd_decompose passes
        pkl_subdir = tmp_path / 'distributed_pkl'
        pkl_subdir.mkdir()
        (pkl_subdir / 'tile_0_0.pkl').touch()

        args = argparse.Namespace(
            netlist_dir=str(tmp_path), net=None,
            backend='local', verbose=False,
            output=str(tmp_path / 'out'), no_plot=True,
            t_start=0.0, t_end=100e-9, dt=0.1e-9,
            top_k=5, window_percent=10.0, instances=None,
            method='be', smooth=True,
            aggressor_top_k=0, adjoint_method='dynamic',
            adjoint_memory_window=20,
            plot_layers=None, max_stripes=500,
        )

        cmd_decompose(args)

        mock_gen_plots.assert_not_called()


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


class TestDecomposeConfigMerge:
    """Tests for _merge_decompose_config and YAML config support."""

    def _default_decompose_args(self, **overrides):
        """Create a Namespace with decompose defaults, applying overrides."""
        defaults = dict(
            netlist_dir='/tmp/pkl', net=None, backend='local', verbose=False,
            output='./irdrop_decomp_results', no_plot=False,
            t_start=0.0, t_end=100e-9, dt=0.1e-9,
            top_k=5, window_percent=10.0, instances=None,
            method='be', smooth=None,
            aggressor_top_k=0, adjoint_method='dynamic',
            adjoint_memory_window=20,
            plot_layers=None, max_stripes=500,
            config=None, use_cholmod=None, use_splu=False,
            cholmod_mode='auto', cholmod_ordering='default',
            cholmod_use_long=None,
            profile_memory=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_defaults_dict_matches_argparse(self):
        """Ensure _DEFAULTS in _merge_decompose_config stays in sync with argparse."""
        from distributed.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(['decompose', '/tmp/pkl'])

        # These are the defaults that _merge_decompose_config relies on.
        # If any change in build_parser(), this test will catch the drift.
        expected = {
            't_start': 0.0, 't_end': 100e-9, 'dt': 0.1e-9, 'method': 'be',
            'top_k': 5, 'window_percent': 10.0, 'aggressor_top_k': 0,
            'adjoint_method': 'dynamic', 'adjoint_memory_window': 20,
            'output': './irdrop_decomp_results', 'no_plot': False,
            'plot_layers': None, 'max_stripes': 500, 'verbose': False,
            'instances': None,
        }
        for key, val in expected.items():
            actual = getattr(args, key)
            assert actual == val, (
                f"_DEFAULTS[{key!r}] = {val!r} but argparse default is {actual!r}. "
                f"Update _DEFAULTS in _merge_decompose_config."
            )

    def test_empty_config_no_change(self):
        """Empty config dict should not modify any args."""
        from distributed.cli import _merge_decompose_config

        args = self._default_decompose_args()
        result = _merge_decompose_config({}, args)

        assert result.t_end == 100e-9
        assert result.dt == 0.1e-9
        assert result.top_k == 5
        assert result.smooth is True  # resolved from None

    def test_time_section_applied(self):
        """Config time section should override defaults."""
        from distributed.cli import _merge_decompose_config

        config = {
            'time': {
                'start': '5ns',
                'end': '50ns',
                'dt': '50ps',
            }
        }
        args = self._default_decompose_args()
        result = _merge_decompose_config(config, args)

        assert result.t_start == pytest.approx(5e-9)
        assert result.t_end == pytest.approx(50e-9)
        assert result.dt == pytest.approx(50e-12)

    def test_cli_overrides_config_time(self):
        """CLI time args should take precedence over config."""
        from distributed.cli import _merge_decompose_config

        config = {
            'time': {
                'end': '50ns',
                'dt': '50ps',
            }
        }
        # Simulate user passing --t-end 200e-9 on CLI (differs from default)
        args = self._default_decompose_args(t_end=200e-9)
        result = _merge_decompose_config(config, args)

        # CLI wins for t_end, config wins for dt
        assert result.t_end == pytest.approx(200e-9)
        assert result.dt == pytest.approx(50e-12)

    def test_analysis_section_applied(self):
        """Config analysis section should set top_k, window_percent, method."""
        from distributed.cli import _merge_decompose_config

        config = {
            'analysis': {
                'top_k': 10,
                'window_percent': 15.0,
                'integration': 'trap',
            }
        }
        args = self._default_decompose_args()
        result = _merge_decompose_config(config, args)

        assert result.top_k == 10
        assert result.window_percent == 15.0
        assert result.method == 'trap'

    def test_cli_overrides_config_analysis(self):
        """CLI analysis args should take precedence over config."""
        from distributed.cli import _merge_decompose_config

        config = {
            'analysis': {
                'top_k': 10,
                'window_percent': 15.0,
            }
        }
        # User passed --top-k 20 on CLI
        args = self._default_decompose_args(top_k=20)
        result = _merge_decompose_config(config, args)

        assert result.top_k == 20  # CLI wins
        assert result.window_percent == 15.0  # config wins

    def test_smooth_default_when_unset(self):
        """Smooth defaults to True when neither CLI nor config set it."""
        from distributed.cli import _merge_decompose_config

        config = {}
        args = self._default_decompose_args(smooth=None)
        result = _merge_decompose_config(config, args)

        assert result.smooth is True

    def test_smooth_from_config(self):
        """Config smooth_sources should be used when CLI is unset."""
        from distributed.cli import _merge_decompose_config

        config = {'analysis': {'smooth_sources': False}}
        args = self._default_decompose_args(smooth=None)
        result = _merge_decompose_config(config, args)

        assert result.smooth is False

    def test_smooth_cli_overrides_config(self):
        """Explicit --smooth on CLI should override config."""
        from distributed.cli import _merge_decompose_config

        config = {'analysis': {'smooth_sources': False}}
        # User passed --smooth explicitly (argparse sets True)
        args = self._default_decompose_args(smooth=True)
        result = _merge_decompose_config(config, args)

        assert result.smooth is True

    def test_smooth_no_smooth_cli_overrides_config(self):
        """Explicit --no-smooth on CLI should override config."""
        from distributed.cli import _merge_decompose_config

        config = {'analysis': {'smooth_sources': True}}
        # User passed --no-smooth explicitly (argparse sets False)
        args = self._default_decompose_args(smooth=False)
        result = _merge_decompose_config(config, args)

        assert result.smooth is False

    def test_instances_from_config_list(self):
        """Config instances list should be joined as comma-separated string."""
        from distributed.cli import _merge_decompose_config

        config = {
            'analysis': {
                'instances': ['1000_2000_M0', '3000_4000_M0'],
            }
        }
        args = self._default_decompose_args()
        result = _merge_decompose_config(config, args)

        assert result.instances == '1000_2000_M0,3000_4000_M0'

    def test_instances_cli_overrides_config(self):
        """CLI --instances should take precedence over config list."""
        from distributed.cli import _merge_decompose_config

        config = {
            'analysis': {
                'instances': ['1000_2000_M0', '3000_4000_M0'],
            }
        }
        args = self._default_decompose_args(
            instances='5000_6000_M1,7000_8000_M1'
        )
        result = _merge_decompose_config(config, args)

        assert result.instances == '5000_6000_M1,7000_8000_M1'

    def test_instances_from_file(self, tmp_path):
        """Config instances_file should be read and joined."""
        from distributed.cli import _merge_decompose_config

        victims_file = tmp_path / 'victims.txt'
        victims_file.write_text('1000_2000_M0\n3000_4000_M0\n\n')

        config = {
            'analysis': {
                'instances_file': str(victims_file),
            }
        }
        args = self._default_decompose_args()
        result = _merge_decompose_config(config, args)

        assert result.instances == '1000_2000_M0,3000_4000_M0'

    def test_instances_list_takes_priority_over_file(self, tmp_path):
        """Config instances list should take priority over instances_file."""
        from distributed.cli import _merge_decompose_config

        victims_file = tmp_path / 'victims.txt'
        victims_file.write_text('file_node_1\nfile_node_2\n')

        config = {
            'analysis': {
                'instances': ['list_node_1'],
                'instances_file': str(victims_file),
            }
        }
        args = self._default_decompose_args()
        result = _merge_decompose_config(config, args)

        assert result.instances == 'list_node_1'

    def test_aggressor_section_applied(self):
        """Config aggressor section should set aggressor params."""
        from distributed.cli import _merge_decompose_config

        config = {
            'aggressor': {
                'top_k': 15,
                'method': 'static',
                'memory_window': 30,
            }
        }
        args = self._default_decompose_args()
        result = _merge_decompose_config(config, args)

        assert result.aggressor_top_k == 15
        assert result.adjoint_method == 'static'


    def test_output_section_applied(self):
        """Config output section should set output params."""
        from distributed.cli import _merge_decompose_config

        config = {
            'output': {
                'output_dir': '/my/results',
                'no_plot': True,
                'plot_layers': ['M0', 'M1', 'M2'],
                'max_stripes': 1000,
                'verbose': True,
            }
        }
        args = self._default_decompose_args()
        result = _merge_decompose_config(config, args)

        assert result.output == '/my/results'
        assert result.no_plot is True
        assert result.plot_layers == 'M0,M1,M2'
        assert result.max_stripes == 1000
        assert result.verbose is True

    def test_solver_section_applied(self):
        """Config solver section should set cholmod params."""
        from distributed.cli import _merge_decompose_config

        config = {
            'solver': {
                'use_cholmod': True,
                'ordering': 'metis',
                'mode': 'supernodal',
            }
        }
        args = self._default_decompose_args()
        result = _merge_decompose_config(config, args)

        assert result.use_cholmod is True
        assert result.cholmod_ordering == 'metis'
        assert result.cholmod_mode == 'supernodal'

    def test_backend_from_config(self):
        """Config backend should override default 'local'."""
        from distributed.cli import _merge_decompose_config

        config = {'backend': 'ray'}
        args = self._default_decompose_args()
        result = _merge_decompose_config(config, args)

        assert result.backend == 'ray'

    def test_cli_backend_overrides_config(self):
        """CLI backend should take precedence over config."""
        from distributed.cli import _merge_decompose_config

        config = {'backend': 'local'}
        args = self._default_decompose_args(backend='ray')
        result = _merge_decompose_config(config, args)

        assert result.backend == 'ray'

    def test_full_config_round_trip(self):
        """A realistic full config should merge correctly."""
        from distributed.cli import _merge_decompose_config

        config = {
            'backend': 'ray',
            'time': {'start': '1ns', 'end': '10ns', 'dt': '100ps'},
            'analysis': {
                'top_k': 3,
                'window_percent': 8.0,
                'integration': 'trap',
                'smooth_sources': False,
                'instances': ['node_A', 'node_B'],
            },
            'aggressor': {'top_k': 5, 'method': 'static', 'memory_window': 10},
            'output': {
                'output_dir': '/out',
                'verbose': True,
                'max_stripes': 250,
            },
        }
        args = self._default_decompose_args()
        result = _merge_decompose_config(config, args)

        assert result.backend == 'ray'
        assert result.t_start == pytest.approx(1e-9)
        assert result.t_end == pytest.approx(10e-9)
        assert result.dt == pytest.approx(100e-12)
        assert result.top_k == 3
        assert result.window_percent == 8.0
        assert result.method == 'trap'
        assert result.smooth is False
        assert result.instances == 'node_A,node_B'
        assert result.aggressor_top_k == 5
        assert result.adjoint_method == 'static'
        assert result.adjoint_memory_window == 10
        assert result.output == '/out'
        assert result.verbose is True
        assert result.max_stripes == 250

    def test_backward_compat_no_config(self):
        """Without --config, decompose should work exactly as before."""
        parser = build_parser()
        args = parser.parse_args([
            'decompose', '/tmp/pkl',
            '--t-end', '50e-9',
            '--top-k', '3',
            '--no-smooth',
        ])
        assert args.t_end == 50e-9
        assert args.top_k == 3
        assert args.smooth is False
        assert args.config is None

    def test_smooth_none_default_in_decompose(self):
        """Decompose smooth should default to None (sentinel)."""
        parser = build_parser()
        args = parser.parse_args(['decompose', '/tmp/pkl'])
        assert args.smooth is None


class TestLoadDecomposeConfig:
    """Tests for _load_decompose_config."""

    def test_load_valid_yaml(self, tmp_path):
        """Should load a valid YAML file."""
        from distributed.cli import _load_decompose_config

        cfg_file = tmp_path / 'test.yaml'
        cfg_file.write_text('time:\n  end: 10ns\n')

        config = _load_decompose_config(str(cfg_file))
        assert config['time']['end'] == '10ns'

    def test_load_empty_yaml(self, tmp_path):
        """Empty YAML file should return empty dict."""
        from distributed.cli import _load_decompose_config

        cfg_file = tmp_path / 'empty.yaml'
        cfg_file.write_text('')

        config = _load_decompose_config(str(cfg_file))
        assert config == {}

    def test_load_missing_file(self):
        """Missing file should raise SystemExit."""
        from distributed.cli import _load_decompose_config

        with pytest.raises(SystemExit):
            _load_decompose_config('/nonexistent/path.yaml')


class TestDecomposeConfigIntegration:
    """Integration tests for config file in cmd_decompose flow."""

    @patch('analysis.dynamic_irdrop_decomposition.generate_plots')
    @patch('analysis.dynamic_irdrop_decomposition.print_results')
    @patch('analysis.dynamic_irdrop_decomposition.Logger')
    @patch('distributed.decomposition.analyze_distributed_decomposition')
    @patch('distributed.cli._load_and_apply_config', side_effect=lambda a: a)
    @patch('distributed.cli._setup_logging')
    def test_cmd_decompose_with_config(
        self, mock_log, mock_config, mock_analyze,
        mock_logger_cls, mock_print, mock_gen_plots, tmp_path,
    ):
        """cmd_decompose should merge config and pass correct values."""
        from distributed.cli import cmd_decompose

        mock_result = MagicMock()
        mock_result.worst_instances = []
        mock_analyze.return_value = (mock_result, MagicMock(), MagicMock())
        mock_logger_cls.return_value = MagicMock()

        # Create fake tile pkl so the glob check in cmd_decompose passes
        netlist_dir = tmp_path / 'netlist'
        netlist_dir.mkdir()
        pkl_subdir = netlist_dir / 'distributed_pkl'
        pkl_subdir.mkdir()
        (pkl_subdir / 'tile_0_0.pkl').touch()

        # Write a config file
        cfg_file = tmp_path / 'decompose.yaml'
        cfg_file.write_text(
            'time:\n'
            '  end: 5ns\n'
            '  dt: 50ps\n'
            'analysis:\n'
            '  top_k: 8\n'
            '  window_percent: 12.0\n'
            '  smooth_sources: false\n'
            '  instances:\n'
            '    - victim_node_1\n'
            '    - victim_node_2\n'
            'aggressor:\n'
            '  top_k: 7\n'
            'output:\n'
            '  verbose: true\n'
        )

        args = argparse.Namespace(
            netlist_dir=str(netlist_dir), net=None,
            backend='local', verbose=False,
            output='./irdrop_decomp_results', no_plot=True,
            t_start=0.0, t_end=100e-9, dt=0.1e-9,
            top_k=5, window_percent=10.0, instances=None,
            method='be', smooth=None,
            aggressor_top_k=0, adjoint_method='dynamic',
            adjoint_memory_window=20,
            plot_layers=None, max_stripes=500,
            config=str(cfg_file),
            use_cholmod=None, use_splu=False,
            cholmod_mode='auto', cholmod_ordering='default',
            cholmod_use_long=None,
            profile_memory=False,
        )

        cmd_decompose(args)

        call_kwargs = mock_analyze.call_args[1]
        assert call_kwargs['t_end'] == pytest.approx(5e-9)
        assert call_kwargs['dt'] == pytest.approx(50e-12)
        assert call_kwargs['top_k'] == 8
        assert call_kwargs['window_percent'] == 12.0
        assert call_kwargs['smooth_sources'] is False
        assert call_kwargs['instances'] == ['victim_node_1', 'victim_node_2']
        assert call_kwargs['aggressor_top_k'] == 7

    @patch('analysis.dynamic_irdrop_decomposition.generate_plots')
    @patch('analysis.dynamic_irdrop_decomposition.print_results')
    @patch('analysis.dynamic_irdrop_decomposition.Logger')
    @patch('distributed.decomposition.analyze_distributed_decomposition')
    @patch('distributed.cli._load_and_apply_config', side_effect=lambda a: a)
    @patch('distributed.cli._setup_logging')
    def test_cmd_decompose_cli_overrides_config(
        self, mock_log, mock_config, mock_analyze,
        mock_logger_cls, mock_print, mock_gen_plots, tmp_path,
    ):
        """CLI args should take precedence over config values."""
        from distributed.cli import cmd_decompose

        mock_result = MagicMock()
        mock_result.worst_instances = []
        mock_analyze.return_value = (mock_result, MagicMock(), MagicMock())
        mock_logger_cls.return_value = MagicMock()

        # Create fake tile pkl so the glob check in cmd_decompose passes
        netlist_dir = tmp_path / 'netlist'
        netlist_dir.mkdir()
        pkl_subdir = netlist_dir / 'distributed_pkl'
        pkl_subdir.mkdir()
        (pkl_subdir / 'tile_0_0.pkl').touch()

        # Config sets t_end=5ns, top_k=8
        cfg_file = tmp_path / 'decompose.yaml'
        cfg_file.write_text(
            'time:\n'
            '  end: 5ns\n'
            'analysis:\n'
            '  top_k: 8\n'
        )

        # CLI overrides t_end to 200e-9 (differs from default 100e-9)
        args = argparse.Namespace(
            netlist_dir=str(netlist_dir), net=None,
            backend='local', verbose=False,
            output='./irdrop_decomp_results', no_plot=True,
            t_start=0.0, t_end=200e-9, dt=0.1e-9,
            top_k=5, window_percent=10.0, instances=None,
            method='be', smooth=None,
            aggressor_top_k=0, adjoint_method='dynamic',
            adjoint_memory_window=20,
            plot_layers=None, max_stripes=500,
            config=str(cfg_file),
            use_cholmod=None, use_splu=False,
            cholmod_mode='auto', cholmod_ordering='default',
            cholmod_use_long=None,
            profile_memory=False,
        )

        cmd_decompose(args)

        call_kwargs = mock_analyze.call_args[1]
        # CLI wins for t_end, config wins for top_k
        assert call_kwargs['t_end'] == pytest.approx(200e-9)
        assert call_kwargs['top_k'] == 8

    @patch('analysis.dynamic_irdrop_decomposition.generate_plots')
    @patch('analysis.dynamic_irdrop_decomposition.print_results')
    @patch('analysis.dynamic_irdrop_decomposition.Logger')
    @patch('distributed.decomposition.analyze_distributed_decomposition')
    @patch('distributed.cli._load_and_apply_config', side_effect=lambda a: a)
    @patch('distributed.cli._setup_logging')
    def test_cmd_decompose_no_config_backward_compat(
        self, mock_log, mock_config, mock_analyze,
        mock_logger_cls, mock_print, mock_gen_plots, tmp_path,
    ):
        """Without config, cmd_decompose should work exactly as before."""
        from distributed.cli import cmd_decompose

        mock_result = MagicMock()
        mock_result.worst_instances = []
        mock_analyze.return_value = (mock_result, MagicMock(), MagicMock())
        mock_logger_cls.return_value = MagicMock()

        # Create fake tile pkl so the glob check in cmd_decompose passes
        pkl_subdir = tmp_path / 'distributed_pkl'
        pkl_subdir.mkdir()
        (pkl_subdir / 'tile_0_0.pkl').touch()

        args = argparse.Namespace(
            netlist_dir=str(tmp_path), net=None,
            backend='local', verbose=False,
            output=str(tmp_path / 'out'), no_plot=True,
            t_start=0.0, t_end=100e-9, dt=0.1e-9,
            top_k=5, window_percent=10.0, instances=None,
            method='be', smooth=None,  # None is the new default
            aggressor_top_k=0, adjoint_method='dynamic',
            adjoint_memory_window=20,
            plot_layers=None, max_stripes=500,
            config=None,
            use_cholmod=None, use_splu=False,
            cholmod_mode='auto', cholmod_ordering='default',
            cholmod_use_long=None,
            profile_memory=False,
        )

        cmd_decompose(args)

        call_kwargs = mock_analyze.call_args[1]
        assert call_kwargs['smooth_sources'] is True  # resolved from None
        assert call_kwargs['t_end'] == 100e-9
        assert call_kwargs['top_k'] == 5
