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

    def test_interface_cg_rtol_argparse_level_default_is_none(self):
        """Finding 2: bare argparse defaults are None (unresolved sentinel);
        the real default (1e-8) is only resolved by _load_and_apply_config()
        AFTER the YAML merge, so explicit-CLI-vs-YAML precedence is
        unambiguous (see TestInterfaceCGPrecedence below)."""
        parser = build_parser()
        args = parser.parse_args(['solve', '/tmp/pkl'])
        assert args.interface_cg_rtol is None

    def test_interface_cg_rtol_default_is_1e8(self):
        """Stage 1b: --interface-cg-rtol resolves to 1e-8 once
        _load_and_apply_config() has applied the built-in default."""
        from distributed.cli import _load_and_apply_config

        parser = build_parser()
        args = parser.parse_args(['solve', '/tmp/pkl'])
        args = _load_and_apply_config(args)
        assert args.interface_cg_rtol == 1e-8

    def test_interface_cg_rtol_explicit_override(self):
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--interface-cg-rtol', '1e-10',
        ])
        assert args.interface_cg_rtol == 1e-10

    def test_interface_cg_new_flag_defaults(self):
        """Stage 1b/1c: new interface CG flags resolve to their documented
        defaults once _load_and_apply_config() has run."""
        from distributed.cli import _load_and_apply_config

        parser = build_parser()
        args = parser.parse_args(['solve', '/tmp/pkl'])
        args = _load_and_apply_config(args)
        assert args.interface_cg_atol == 1e-14
        assert args.interface_cg_maxiter is None
        assert args.interface_cg_strict is True
        assert args.interface_factor_memory_budget == 'auto'
        assert args.interface_block_jacobi_max_bytes == 'auto'

    def test_interface_cg_new_flags_explicit(self):
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl',
            '--interface-cg-atol', '1e-12',
            '--interface-cg-maxiter', '500',
            '--interface-cg-no-strict',
            '--interface-factor-memory-budget', '1073741824',
            '--interface-block-jacobi-max-bytes', '2147483648',
        ])
        assert args.interface_cg_atol == 1e-12
        assert args.interface_cg_maxiter == 500
        assert args.interface_cg_strict is False
        assert args.interface_factor_memory_budget == '1073741824'
        assert args.interface_block_jacobi_max_bytes == '2147483648'

    def test_interface_preconditioner_accepts_explicit_auto(self):
        """Finding 11 regression: 'auto' is the new documented default for
        --interface-preconditioner (Stage 3) but was missing from the
        argparse choices list, so a user could not explicitly pass 'auto'
        on the command line (e.g. to override an explicit value set in a
        YAML config -- explicit CLI flags always win per _load_and_apply_
        config's precedence). Before the fix this raised
        `argparse.ArgumentTypeError`/SystemExit ("invalid choice: 'auto'")."""
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--interface-preconditioner', 'auto',
        ])
        assert args.interface_preconditioner == 'auto'

    def test_interface_coarse_max_bytes_default_and_explicit(self):
        """Finding 5: new interface_coarse_max_bytes setting resolves to
        'auto' by default and accepts an explicit byte count."""
        from distributed.cli import _load_and_apply_config

        parser = build_parser()
        args = parser.parse_args(['solve', '/tmp/pkl'])
        args = _load_and_apply_config(args)
        assert args.interface_coarse_max_bytes == 'auto'

        parser2 = build_parser()
        args2 = parser2.parse_args([
            'solve', '/tmp/pkl', '--interface-coarse-max-bytes', '1073741824',
        ])
        assert args2.interface_coarse_max_bytes == '1073741824'

    def test_interface_cg_new_flags_present_on_run(self):
        """The run subcommand shares the same interface-solver flag group,
        resolving to the same defaults once _load_and_apply_config() runs."""
        from distributed.cli import _load_and_apply_config

        parser = build_parser()
        args = parser.parse_args(['run', '/tmp/netlist'])
        args = _load_and_apply_config(args)
        assert args.interface_cg_rtol == 1e-8
        assert args.interface_cg_atol == 1e-14
        assert args.interface_cg_strict is True
        assert args.interface_factor_memory_budget == 'auto'
        assert args.interface_block_jacobi_max_bytes == 'auto'


class TestInterfaceCGYamlConfig:
    """Stage 1: `solver:` YAML section plumbs the new interface CG settings
    through `_load_and_apply_config` (used by `cmd_solve` / `cmd_run`)."""

    def test_yaml_interface_cg_settings_applied(self, tmp_path):
        from distributed.cli import build_parser, _load_and_apply_config

        config_path = tmp_path / 'solver.yaml'
        # Note: PyYAML only parses bare-mantissa exponents (e.g. "1.0e-10")
        # as floats -- "1e-10" (no decimal point) parses as a string.
        config_path.write_text(
            "solver:\n"
            "  interface_solver: cg\n"
            "  interface_matvec_mode: tilewise\n"
            "  interface_preconditioner: jacobi\n"
            "  interface_cg_rtol: 1.0e-10\n"
            "  interface_cg_atol: 1.0e-11\n"
            "  interface_cg_maxiter: 250\n"
            "  interface_cg_strict: false\n"
            "  interface_factor_memory_budget: 1073741824\n"
            "  interface_block_jacobi_max_bytes: 2147483648\n"
        )
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--config', str(config_path),
        ])
        args = _load_and_apply_config(args)

        assert args.interface_solver == 'cg'
        assert args.interface_matvec_mode == 'tilewise'
        assert args.interface_preconditioner == 'jacobi'
        assert args.interface_cg_rtol == 1e-10
        assert args.interface_cg_atol == 1e-11
        assert args.interface_cg_maxiter == 250
        assert args.interface_cg_strict is False
        assert args.interface_factor_memory_budget == 1073741824
        assert args.interface_block_jacobi_max_bytes == 2147483648

    def test_yaml_interface_cg_cli_override_takes_precedence(self, tmp_path):
        from distributed.cli import build_parser, _load_and_apply_config

        config_path = tmp_path / 'solver.yaml'
        config_path.write_text(
            "solver:\n"
            "  interface_cg_rtol: 1.0e-10\n"
        )
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--config', str(config_path),
            '--interface-cg-rtol', '1e-9',
        ])
        args = _load_and_apply_config(args)
        assert args.interface_cg_rtol == 1e-9


class TestInterfaceCGPrecedence:
    """Finding 2: explicit-CLI > YAML > built-in-default precedence, tested
    per the four scenarios that broke under the old shared-sentinel-tuple
    check (a single tuple of "any key's default" values, so an explicit CLI
    value equal to ANY key's default -- not just its own -- was misread as
    "unset")."""

    def test_explicit_flag_equal_to_own_default_beats_yaml(self, tmp_path):
        """--interface-cg-strict (value True, which is also the built-in
        default) must beat a YAML interface_cg_strict: false -- previously
        it could not, because True was in the shared default-sentinel set."""
        from distributed.cli import build_parser, _load_and_apply_config

        config_path = tmp_path / 'solver.yaml'
        config_path.write_text("solver:\n  interface_cg_strict: false\n")
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--config', str(config_path),
            '--interface-cg-strict',
        ])
        args = _load_and_apply_config(args)
        assert args.interface_cg_strict is True

    def test_explicit_flag_equal_to_another_keys_default_beats_yaml(self, tmp_path):
        """--interface-cg-maxiter 1 (1 == True in Python, and True is the
        default for the UNRELATED interface_cg_strict flag) must not be
        misread as "unset" and overridden by a YAML interface_cg_maxiter."""
        from distributed.cli import build_parser, _load_and_apply_config

        config_path = tmp_path / 'solver.yaml'
        config_path.write_text("solver:\n  interface_cg_maxiter: 999\n")
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--config', str(config_path),
            '--interface-cg-maxiter', '1',
        ])
        args = _load_and_apply_config(args)
        assert args.interface_cg_maxiter == 1

    def test_unset_flag_yields_yaml_value(self, tmp_path):
        """A flag left unset on the CLI picks up the YAML value."""
        from distributed.cli import build_parser, _load_and_apply_config

        config_path = tmp_path / 'solver.yaml'
        config_path.write_text("solver:\n  interface_cg_atol: 1.0e-11\n")
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--config', str(config_path),
        ])
        args = _load_and_apply_config(args)
        assert args.interface_cg_atol == 1e-11

    def test_stage2_settings_yaml_precedence(self, tmp_path):
        """Stage 2 additions (matvec_threads, interface_matvec_dtype,
        interface_strict_dtype_rtol, interface_drop_s_global) follow the
        same explicit-CLI > YAML > built-in-default precedence as the
        Stage 1 keys."""
        from distributed.cli import build_parser, _load_and_apply_config

        config_path = tmp_path / 'solver.yaml'
        config_path.write_text(
            "solver:\n"
            "  matvec_threads: 16\n"
            "  interface_matvec_dtype: float32\n"
            "  interface_strict_dtype_rtol: false\n"
            "  interface_drop_s_global: true\n"
        )
        parser = build_parser()
        # matvec_threads left unset on CLI -> picks up YAML.
        # interface_matvec_dtype explicit on CLI -> beats YAML.
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--config', str(config_path),
            '--interface-matvec-dtype', 'float64',
        ])
        args = _load_and_apply_config(args)
        assert int(args.matvec_threads) == 16
        assert args.interface_matvec_dtype == 'float64'
        assert args.interface_strict_dtype_rtol is False
        assert args.interface_drop_s_global is True

    def test_interface_no_drop_s_global_overrides_yaml_true(self, tmp_path):
        """Finding 10: --interface-no-drop-s-global must produce an
        explicit False that beats a YAML interface_drop_s_global: true --
        pre-fix, no CLI flag could express False for this key at all, so
        explicit-CLI > YAML precedence was unexpressable in that direction
        (unlike every other paired boolean flag in this group, e.g.
        --interface-cg-strict/--interface-cg-no-strict)."""
        from distributed.cli import build_parser, _load_and_apply_config

        config_path = tmp_path / 'solver.yaml'
        config_path.write_text(
            "solver:\n"
            "  interface_drop_s_global: true\n"
        )
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--config', str(config_path),
            '--interface-no-drop-s-global',
        ])
        args = _load_and_apply_config(args)
        assert args.interface_drop_s_global is False

    def test_interface_no_drop_s_global_parses_to_false(self):
        """Sanity: the negation flag alone (no YAML) parses to False."""
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--interface-no-drop-s-global',
        ])
        assert args.interface_drop_s_global is False

    def test_nothing_set_yields_builtin_default(self):
        """Neither CLI nor YAML set -> the built-in default is used."""
        from distributed.cli import build_parser, _load_and_apply_config

        parser = build_parser()
        args = parser.parse_args(['solve', '/tmp/pkl'])
        args = _load_and_apply_config(args)
        assert args.interface_solver == 'auto'
        # Stage 2 item 8: default changed 'assembled' -> 'auto' (tilewise
        # whenever per-tile Schur blocks are available).
        assert args.interface_matvec_mode == 'auto'
        # Stage 3: default changed 'block_jacobi' -> 'auto' (resolves to
        # 'two_level' for CG+tilewise, else 'block_jacobi' -- see
        # interface_iterative.resolve_preconditioner).
        assert args.interface_preconditioner == 'auto'
        assert args.interface_cg_rtol == 1e-8
        assert args.interface_cg_atol == 1e-14
        assert args.interface_cg_maxiter is None
        assert args.interface_cg_strict is True
        assert args.interface_factor_memory_budget == 'auto'
        assert args.interface_block_jacobi_max_bytes == 'auto'
        # Stage 2 additions
        assert args.matvec_threads == 'auto'
        assert args.interface_matvec_dtype == 'float64'
        assert args.interface_strict_dtype_rtol is True
        assert args.interface_drop_s_global is False
        # Stage 3 additions
        # Measurement-driven flip (2026-07-20): interface_coarse.
        # DEFAULT_GENEO_K changed 4 -> 0 (GenEO measured zero iteration
        # benefit on mi200k_v2, see interface_deflation_notes.md); GenEO
        # itself is unchanged and fully opt-in via geneo_k > 0.
        assert args.interface_coarse_geneo_k == 0
        assert args.interface_coarse_geneo_tol == 1e-6
        assert args.interface_coarse_eps_rank == 1e-12
        assert args.interface_coarse_max_cols == 4096
        # Finding 5 (Stage 3): new byte-based coarse-build guard.
        assert args.interface_coarse_max_bytes == 'auto'
        # A-DEF2 work package additions.
        # Measurement-driven flip (2026-07-20): interface_coarse.
        # DEFAULT_APPLY_MODE changed 'additive' -> 'deflated' ('deflated'
        # beat 'additive' in every cell of the mi200k_v2 head-to-head
        # matrix, see interface_deflation_notes.md's "Defaults flipped by
        # measurement" section). 'additive' remains fully supported and
        # selectable explicitly.
        assert args.interface_coarse_apply_mode == 'deflated'
        assert args.interface_deflated_reproject_every == 50
        assert args.interface_warm_start_extrapolation is False


class TestADef2CLIWiring:
    """A-DEF2 work package: CLI flags + YAML + precedence for
    interface_coarse_apply_mode / interface_deflated_reproject_every /
    interface_warm_start_extrapolation -- follows the exact pattern of
    TestInterfaceCGYamlConfig/TestInterfaceCGPrecedence above."""

    def test_explicit_cli_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl',
            '--interface-coarse-apply-mode', 'deflated',
            '--interface-deflated-reproject-every', '10',
            '--interface-warm-start-extrapolation',
        ])
        assert args.interface_coarse_apply_mode == 'deflated'
        assert args.interface_deflated_reproject_every == 10
        assert args.interface_warm_start_extrapolation is True

    def test_yaml_settings_applied(self, tmp_path):
        from distributed.cli import build_parser, _load_and_apply_config

        config_path = tmp_path / 'solver.yaml'
        config_path.write_text(
            "solver:\n"
            "  interface_coarse_apply_mode: deflated\n"
            "  interface_deflated_reproject_every: 25\n"
            "  interface_warm_start_extrapolation: true\n"
        )
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--config', str(config_path),
        ])
        args = _load_and_apply_config(args)
        assert args.interface_coarse_apply_mode == 'deflated'
        assert args.interface_deflated_reproject_every == 25
        assert args.interface_warm_start_extrapolation is True

    def test_explicit_cli_beats_yaml(self, tmp_path):
        from distributed.cli import build_parser, _load_and_apply_config

        config_path = tmp_path / 'solver.yaml'
        config_path.write_text(
            "solver:\n"
            "  interface_coarse_apply_mode: deflated\n"
        )
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--config', str(config_path),
            '--interface-coarse-apply-mode', 'additive',
        ])
        args = _load_and_apply_config(args)
        assert args.interface_coarse_apply_mode == 'additive'

    def test_no_warm_start_extrapolation_overrides_yaml_true(self, tmp_path):
        """Finding-10-style paired negation flag: an explicit CLI False
        must beat a YAML true (matches --interface-no-drop-s-global's
        established pattern)."""
        from distributed.cli import build_parser, _load_and_apply_config

        config_path = tmp_path / 'solver.yaml'
        config_path.write_text(
            "solver:\n  interface_warm_start_extrapolation: true\n"
        )
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--config', str(config_path),
            '--interface-no-warm-start-extrapolation',
        ])
        args = _load_and_apply_config(args)
        assert args.interface_warm_start_extrapolation is False

    def test_invalid_apply_mode_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                'solve', '/tmp/pkl',
                '--interface-coarse-apply-mode', 'bogus',
            ])

    def test_build_interface_settings_includes_deflated_keys(self):
        from distributed.cli import (
            build_parser, _load_and_apply_config, _build_interface_settings,
        )

        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl',
            '--interface-coarse-apply-mode', 'deflated',
            '--interface-deflated-reproject-every', '5',
            '--interface-warm-start-extrapolation',
        ])
        args = _load_and_apply_config(args)
        settings = _build_interface_settings(args)
        assert settings['interface_coarse_apply_mode'] == 'deflated'
        assert settings['interface_deflated_reproject_every'] == 5
        assert settings['interface_warm_start_extrapolation'] is True

    def test_warm_start_extrapolation_default_resolves_dynamically(self, monkeypatch):
        """Finding 6 (round-1 code review) regression: the CLI/YAML default
        for interface_warm_start_extrapolation must read
        interface_iterative.DEFAULT_WARM_START_EXTRAPOLATION dynamically
        (same _iface_default pattern as interface_coarse_apply_mode/
        interface_coarse_geneo_k/etc. -- see
        test_interface_iterative_stage2.py's analogous
        monkeypatch.setattr(interface_coarse, 'DEFAULT_*', ...) tests),
        NOT a def-time-bound literal snapshot -- a hardcoded default would
        make this knob permanently unreachable for CLI-driven runs even
        after the coordinator flips the canonical module constant."""
        import distributed.interface_iterative as interface_iterative
        from distributed.cli import _load_and_apply_config, _build_interface_settings

        monkeypatch.setattr(
            interface_iterative, 'DEFAULT_WARM_START_EXTRAPOLATION', True,
        )
        parser = build_parser()
        args = parser.parse_args(['solve', '/tmp/pkl'])
        args = _load_and_apply_config(args)
        assert args.interface_warm_start_extrapolation is True
        settings = _build_interface_settings(args)
        assert settings['interface_warm_start_extrapolation'] is True


class TestTwoLevelBaseCliYaml:
    """NN/BDD work package: --interface-two-level-base CLI flag + YAML
    plumbing for interface_two_level_base and the neumann knobs
    (interface_neumann_{weight,reg,max_bytes} are YAML/settings-only --
    no CLI flags).  Same explicit-CLI > YAML > built-in-default precedence
    machinery as every other interface key."""

    def test_argparse_level_default_is_none(self):
        parser = build_parser()
        args = parser.parse_args(['solve', '/tmp/pkl'])
        assert args.interface_two_level_base is None

    def test_explicit_flag_all_choices(self):
        for base in ('auto', 'block_jacobi', 'jacobi', 'neumann'):
            parser = build_parser()
            args = parser.parse_args([
                'solve', '/tmp/pkl', '--interface-two-level-base', base,
            ])
            assert args.interface_two_level_base == base

    def test_invalid_base_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                'solve', '/tmp/pkl', '--interface-two-level-base', 'bogus',
            ])

    def test_defaults_resolve(self):
        """Unset -> 'auto' base; neumann knobs resolve to the canonical
        interface_iterative defaults (stiffness / 0.0 / 'auto')."""
        from distributed.cli import _load_and_apply_config

        parser = build_parser()
        args = parser.parse_args(['solve', '/tmp/pkl'])
        args = _load_and_apply_config(args)
        assert args.interface_two_level_base == 'auto'
        assert args.interface_neumann_weight == 'stiffness'
        assert args.interface_neumann_reg == 0.0
        assert args.interface_neumann_max_bytes == 'auto'

    def test_yaml_settings_applied(self, tmp_path):
        from distributed.cli import build_parser, _load_and_apply_config

        config_path = tmp_path / 'solver.yaml'
        # PyYAML float caveat: bare-mantissa exponents ("1e-3") parse as
        # strings; use "1.0e-3".
        config_path.write_text(
            "solver:\n"
            "  interface_two_level_base: neumann\n"
            "  interface_neumann_weight: multiplicity\n"
            "  interface_neumann_reg: 1.0e-3\n"
            "  interface_neumann_max_bytes: 1073741824\n"
        )
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--config', str(config_path),
        ])
        args = _load_and_apply_config(args)
        assert args.interface_two_level_base == 'neumann'
        assert args.interface_neumann_weight == 'multiplicity'
        assert args.interface_neumann_reg == 1e-3
        assert args.interface_neumann_max_bytes == 1073741824

    def test_explicit_cli_beats_yaml(self, tmp_path):
        from distributed.cli import build_parser, _load_and_apply_config

        config_path = tmp_path / 'solver.yaml'
        config_path.write_text("solver:\n  interface_two_level_base: auto\n")
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--config', str(config_path),
            '--interface-two-level-base', 'jacobi',
        ])
        args = _load_and_apply_config(args)
        assert args.interface_two_level_base == 'jacobi'

    def test_build_interface_settings_includes_nn_keys(self):
        from distributed.cli import (
            build_parser, _load_and_apply_config, _build_interface_settings,
        )

        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--interface-two-level-base', 'jacobi',
        ])
        args = _load_and_apply_config(args)
        settings = _build_interface_settings(args)
        assert settings['interface_two_level_base'] == 'jacobi'
        assert settings['interface_neumann_weight'] == 'stiffness'
        assert settings['interface_neumann_reg'] == 0.0
        assert settings['interface_neumann_max_bytes'] == 'auto'

    def test_neumann_defaults_resolve_dynamically(self, monkeypatch):
        """Same Finding-6 lazy-resolution contract as
        interface_warm_start_extrapolation: the CLI/YAML defaults for the
        neumann knobs must read interface_iterative.DEFAULT_NEUMANN_WEIGHT/
        _REG at resolution time, not a def-time-bound literal copy."""
        import distributed.interface_iterative as interface_iterative
        from distributed.cli import _load_and_apply_config

        monkeypatch.setattr(
            interface_iterative, 'DEFAULT_NEUMANN_WEIGHT', 'multiplicity',
        )
        monkeypatch.setattr(
            interface_iterative, 'DEFAULT_NEUMANN_REG', 1e-3,
        )
        parser = build_parser()
        args = parser.parse_args(['solve', '/tmp/pkl'])
        args = _load_and_apply_config(args)
        assert args.interface_neumann_weight == 'multiplicity'
        assert args.interface_neumann_reg == 1e-3

    def test_run_subcommand_has_flag(self):
        """run shares the interface flag group with solve."""
        parser = build_parser()
        args = parser.parse_args([
            'run', '/tmp/netlist', '--interface-two-level-base', 'jacobi',
        ])
        assert args.interface_two_level_base == 'jacobi'


class TestProgressEveryCliYaml:
    """Docs Sec 7.13 recommended change 2: --interface-cg-progress-every
    CLI flag + interface_cg_progress_every YAML key (0 = disabled default),
    same precedence machinery as the other interface keys."""

    def test_argparse_level_default_is_none(self):
        parser = build_parser()
        args = parser.parse_args(['solve', '/tmp/pkl'])
        assert args.interface_cg_progress_every is None

    def test_default_resolves_to_zero(self):
        from distributed.cli import _load_and_apply_config

        parser = build_parser()
        args = parser.parse_args(['solve', '/tmp/pkl'])
        args = _load_and_apply_config(args)
        assert args.interface_cg_progress_every == 0

    def test_explicit_flag(self):
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--interface-cg-progress-every', '50',
        ])
        assert args.interface_cg_progress_every == 50

    def test_yaml_applied(self, tmp_path):
        from distributed.cli import build_parser, _load_and_apply_config

        config_path = tmp_path / 'solver.yaml'
        config_path.write_text(
            "solver:\n  interface_cg_progress_every: 25\n"
        )
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--config', str(config_path),
        ])
        args = _load_and_apply_config(args)
        assert args.interface_cg_progress_every == 25

    def test_explicit_zero_beats_yaml(self, tmp_path):
        """--interface-cg-progress-every 0 (equal to the built-in default)
        must still beat a YAML value -- the Finding-2 None-sentinel
        contract."""
        from distributed.cli import build_parser, _load_and_apply_config

        config_path = tmp_path / 'solver.yaml'
        config_path.write_text(
            "solver:\n  interface_cg_progress_every: 25\n"
        )
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--config', str(config_path),
            '--interface-cg-progress-every', '0',
        ])
        args = _load_and_apply_config(args)
        assert args.interface_cg_progress_every == 0

    def test_build_interface_settings_includes_key(self):
        from distributed.cli import (
            build_parser, _load_and_apply_config, _build_interface_settings,
        )

        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--interface-cg-progress-every', '50',
        ])
        args = _load_and_apply_config(args)
        settings = _build_interface_settings(args)
        assert settings['interface_cg_progress_every'] == 50


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
            qs_candidate_factor=3000,
            max_qs_candidates=10000,
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
            qs_candidate_factor=3000,
            max_qs_candidates=10000,
            verbose=False,
            coordinator_solver_config=None,
            worker_solver_config=None,
            threads_per_worker=None,
            # Finding 3: cmd_decompose now pushes the same interface_*
            # settings dict as cmd_solve/cmd_run (built-in defaults here
            # since `args` carries no interface_* attributes).
            interface_settings={
                'interface_solver': 'auto',
                'interface_matvec_mode': 'auto',
                # Stage 3: default changed 'block_jacobi' -> 'auto'.
                'interface_preconditioner': 'auto',
                'interface_cg_rtol': 1e-8,
                'interface_cg_atol': 1e-14,
                'interface_cg_maxiter': None,
                'interface_cg_progress_every': 0,
                'interface_cg_strict': True,
                'interface_factor_memory_budget': 'auto',
                'interface_block_jacobi_max_bytes': 'auto',
                'matvec_threads': 'auto',
                'interface_matvec_dtype': 'float64',
                'interface_strict_dtype_rtol': True,
                'interface_drop_s_global': False,
                # Measurement-driven flip (2026-07-20): DEFAULT_GENEO_K
                # 4 -> 0 (see interface_deflation_notes.md).
                'interface_coarse_geneo_k': 0,
                'interface_coarse_geneo_tol': 1e-6,
                'interface_coarse_eps_rank': 1e-12,
                'interface_coarse_max_cols': 4096,
                # Finding 5 (Stage 3): new byte-based coarse-build guard.
                'interface_coarse_max_bytes': 'auto',
                # A-DEF2 work package. Measurement-driven flip
                # (2026-07-20): DEFAULT_APPLY_MODE 'additive' -> 'deflated'
                # (see interface_deflation_notes.md's "Defaults flipped by
                # measurement" section).
                'interface_coarse_apply_mode': 'deflated',
                'interface_deflated_reproject_every': 50,
                'interface_warm_start_extrapolation': False,
                # NN/BDD work package: two_level base + neumann knobs.
                'interface_two_level_base': 'auto',
                'interface_neumann_weight': 'stiffness',
                'interface_neumann_reg': 0.0,
                'interface_neumann_max_bytes': 'auto',
            },
            island_detection='auto',
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
    def test_decompose_propagates_interface_settings(
        self, mock_log, mock_config, mock_analyze,
        mock_logger_cls, mock_print, mock_gen_plots, tmp_path,
    ):
        """Finding 3: a representative --interface-cg-* flag reaches
        model.settings via the interface_settings= kwarg passed into
        analyze_distributed_decomposition -- previously these flags were
        accepted by the decompose subparser and silently dropped."""
        from distributed.cli import cmd_decompose

        mock_result = MagicMock()
        mock_result.worst_instances = []
        mock_analyze.return_value = (mock_result, MagicMock(), MagicMock())
        mock_logger_cls.return_value = MagicMock()

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
            qs_candidate_factor=3000,
            max_qs_candidates=10000,
            plot_layers=None, max_stripes=500,
            # Representative interface_* overrides (as if the user passed
            # --interface-cg-no-strict --interface-cg-maxiter 200)
            interface_cg_strict=False,
            interface_cg_maxiter=200,
        )

        cmd_decompose(args)

        call_kwargs = mock_analyze.call_args[1]
        assert 'interface_settings' in call_kwargs
        assert call_kwargs['interface_settings']['interface_cg_strict'] is False
        assert call_kwargs['interface_settings']['interface_cg_maxiter'] == 200

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
            qs_candidate_factor=3000,
            max_qs_candidates=10000,
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
            'qs_candidate_factor': 3000,
            'max_qs_candidates': 10000,
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


class TestFileLogging:
    """Tests for _add_file_logging() and _close_file_logging()."""

    def test_none_output_dir_returns_none(self):
        """Falsy output_dir should return None without creating a handler."""
        from distributed.cli import _add_file_logging
        assert _add_file_logging(None, 'dc') is None
        assert _add_file_logging('', 'dc') is None

    def test_creates_log_file(self, tmp_path):
        """Should create a log file matching the mode slug pattern."""
        from distributed.cli import _add_file_logging, _close_file_logging

        fh = _add_file_logging(str(tmp_path), 'dc')
        try:
            assert fh is not None
            log_files = list(tmp_path.glob('dc_*.log'))
            assert len(log_files) == 1
        finally:
            _close_file_logging(fh)

    def test_mode_slug_sanitization(self, tmp_path):
        """Hyphens in mode should be replaced with underscores in filename."""
        from distributed.cli import _add_file_logging, _close_file_logging

        fh = _add_file_logging(str(tmp_path), 'quasi-static')
        try:
            log_files = list(tmp_path.glob('quasi_static_*.log'))
            assert len(log_files) == 1
            assert 'quasi-static' not in log_files[0].name
        finally:
            _close_file_logging(fh)

    def test_handler_removed_after_close(self, tmp_path):
        """_close_file_logging should remove the handler from root logger."""
        import logging
        from distributed.cli import _add_file_logging, _close_file_logging

        root = logging.getLogger()
        n_before = len(root.handlers)
        fh = _add_file_logging(str(tmp_path), 'transient')
        assert len(root.handlers) == n_before + 1
        _close_file_logging(fh)
        assert len(root.handlers) == n_before

    def test_creates_output_dir_if_missing(self, tmp_path):
        """Should create nested output directories."""
        from distributed.cli import _add_file_logging, _close_file_logging

        nested = tmp_path / 'sub' / 'dir'
        assert not nested.exists()
        fh = _add_file_logging(str(nested), 'parse')
        try:
            assert nested.is_dir()
            assert len(list(nested.glob('parse_*.log'))) == 1
        finally:
            _close_file_logging(fh)


# ---------------------------------------------------------------------------
# Stage 1e island_detection CLI/YAML plumbing: findings F6, F7, F12
# ---------------------------------------------------------------------------

class TestIslandDetectionYamlPrecedence:
    """Finding F7 (plus the pre-existing solve/run precedence already wired
    through _load_and_apply_config): explicit CLI > YAML > 'auto', and a
    falsy-but-non-None YAML value (e.g. PyYAML parsing `off`/`no` to
    ``False``) must NOT be silently coerced to 'auto' -- it must survive so
    model._resolve_island_detection's loud ValueError can fire."""

    def test_yaml_value_applied_when_cli_unset(self, tmp_path):
        from distributed.cli import build_parser, _load_and_apply_config

        config_path = tmp_path / 'solver.yaml'
        config_path.write_text("solver:\n  island_detection: schur_bfs\n")
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--config', str(config_path),
        ])
        args = _load_and_apply_config(args)
        assert args.island_detection == 'schur_bfs'

    def test_cli_flag_beats_yaml(self, tmp_path):
        from distributed.cli import build_parser, _load_and_apply_config

        config_path = tmp_path / 'solver.yaml'
        config_path.write_text("solver:\n  island_detection: schur_bfs\n")
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--config', str(config_path),
            '--island-detection', 'summaries',
        ])
        args = _load_and_apply_config(args)
        assert args.island_detection == 'summaries'

    def test_nothing_set_yields_auto(self):
        from distributed.cli import build_parser, _load_and_apply_config

        parser = build_parser()
        args = parser.parse_args(['solve', '/tmp/pkl'])
        args = _load_and_apply_config(args)
        assert args.island_detection == 'auto'

    def test_falsy_yaml_value_not_coerced_to_auto(self, tmp_path):
        """PyYAML 1.1 parses `island_detection: off` to the bool False --
        _load_and_apply_config must preserve it as-is (only an explicit
        `is None` check gates the 'auto' default), and
        _resolve_island_detection_arg (the F7 fix at the cmd_solve/cmd_run/
        cmd_decompose call sites) must NOT silently coerce it to 'auto'
        either."""
        from distributed.cli import (
            build_parser, _load_and_apply_config, _resolve_island_detection_arg,
        )

        config_path = tmp_path / 'solver.yaml'
        config_path.write_text("solver:\n  island_detection: off\n")
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--config', str(config_path),
        ])
        args = _load_and_apply_config(args)
        assert args.island_detection is False, (
            f"Expected PyYAML to parse 'off' as bool False; "
            f"got {args.island_detection!r}"
        )
        assert _resolve_island_detection_arg(args) is False, (
            "The old `getattr(args, 'island_detection', None) or 'auto'` "
            "pattern would have silently coerced False to 'auto' here"
        )

    def test_falsy_value_reaches_model_resolve_error(self, tmp_path):
        """End-to-end: the falsy YAML value survives all the way to
        model._resolve_island_detection, which raises ValueError for it
        (exactly as it would for any other invalid string) instead of the
        silently-substituted 'auto' engaging the fast path unannounced."""
        from distributed.cli import (
            build_parser, _load_and_apply_config, _resolve_island_detection_arg,
        )
        from distributed.model import ParsedTileBundle, _resolve_island_detection
        from distributed.parser import PackageData, PowerGridMetaData

        config_path = tmp_path / 'solver.yaml'
        config_path.write_text("solver:\n  island_detection: off\n")
        parser = build_parser()
        args = parser.parse_args([
            'solve', '/tmp/pkl', '--config', str(config_path),
        ])
        args = _load_and_apply_config(args)
        resolved = _resolve_island_detection_arg(args)

        pkg = PackageData(
            vsrc_dict={}, package_edges=[], pad_nodes=set(),
            tap_nodes=set(), die_attachment_nodes=set(), vdd=1.0, net_name='VDD',
        )
        metadata = PowerGridMetaData(
            tile_grid=(1, 1), parameters={}, tile_configs=[], package_data=pkg,
            net_name='VDD', vdd=1.0,
        )
        bundle = ParsedTileBundle(
            metadata=metadata, shared_boundary_nodes=set(), pkl_dir='/tmp/irrelevant',
        )
        with pytest.raises(ValueError):
            _resolve_island_detection(bundle, set(), resolved)

    def test_resolve_island_detection_arg_default_when_truly_unset(self):
        """A bare Namespace missing island_detection entirely (e.g. a test
        or caller that skipped _load_and_apply_config) still degrades to
        'auto' -- only a genuine None falls back, not any other falsy value."""
        from distributed.cli import _resolve_island_detection_arg

        assert _resolve_island_detection_arg(argparse.Namespace()) == 'auto'
        assert _resolve_island_detection_arg(
            argparse.Namespace(island_detection=None)
        ) == 'auto'
        assert _resolve_island_detection_arg(
            argparse.Namespace(island_detection='schur_bfs')
        ) == 'schur_bfs'


class TestDecomposeIslandDetectionConfig:
    """Finding F6: cmd_decompose nulls args.config before
    _load_and_apply_config (to avoid re-loading the file through the wrong
    pdn_solver schema), which means _load_and_apply_config's own
    island_detection resolution never sees the decompose YAML's solver
    section -- _merge_decompose_config must resolve it itself."""

    def _default_decompose_args(self, **overrides):
        defaults = dict(
            netlist_dir='/tmp/pkl', net=None, backend='local', verbose=False,
            output='./irdrop_decomp_results', no_plot=False,
            t_start=0.0, t_end=100e-9, dt=0.1e-9,
            top_k=5, window_percent=10.0, instances=None,
            method='be', smooth=None,
            aggressor_top_k=0, adjoint_method='dynamic',
            adjoint_memory_window=20,
            qs_candidate_factor=3000,
            max_qs_candidates=10000,
            plot_layers=None, max_stripes=500,
            config=None, use_cholmod=None, use_splu=False,
            cholmod_mode='auto', cholmod_ordering='default',
            cholmod_use_long=None,
            profile_memory=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_solver_island_detection_applied_from_yaml(self):
        from distributed.cli import _merge_decompose_config

        config = {'solver': {'island_detection': 'schur_bfs'}}
        args = self._default_decompose_args()
        result = _merge_decompose_config(config, args)
        assert result.island_detection == 'schur_bfs'

    def test_cli_island_detection_beats_yaml(self):
        from distributed.cli import _merge_decompose_config

        config = {'solver': {'island_detection': 'schur_bfs'}}
        args = self._default_decompose_args(island_detection='summaries')
        result = _merge_decompose_config(config, args)
        assert result.island_detection == 'summaries'

    def test_unset_without_yaml_stays_none_until_load_and_apply_config(self):
        """_merge_decompose_config leaves it unset (None) when the YAML
        doesn't set it either -- _load_and_apply_config's own fallback
        (called afterward by cmd_decompose) resolves it to 'auto'."""
        from distributed.cli import _merge_decompose_config, _load_and_apply_config

        args = self._default_decompose_args()
        result = _merge_decompose_config({}, args)
        assert getattr(result, 'island_detection', None) is None

        # cmd_decompose nulls args.config before this call (finding F6
        # comment); simulate that here.
        result.config = None
        result = _load_and_apply_config(result)
        assert result.island_detection == 'auto'

    def test_full_cmd_decompose_flow_resolves_island_detection_from_yaml(self, tmp_path):
        """End-to-end simulation of cmd_decompose's actual sequence:
        _load_decompose_config -> args.config = None -> _merge_decompose_config
        -> _load_and_apply_config. YAML's solver.island_detection must
        survive to the final resolved args.island_detection."""
        from distributed.cli import (
            _load_decompose_config, _merge_decompose_config, _load_and_apply_config,
        )

        config_path = tmp_path / 'decompose.yaml'
        config_path.write_text(
            "solver:\n"
            "  island_detection: schur_bfs\n"
        )
        args = self._default_decompose_args(config=str(config_path))
        decompose_config = _load_decompose_config(str(config_path))
        args.config = None  # mirrors cmd_decompose's own line
        args = _merge_decompose_config(decompose_config, args)
        args = _load_and_apply_config(args)

        assert args.island_detection == 'schur_bfs'


class TestIslandDetectionRoleYamlValidation:
    """Finding F12: island_detection is a top-level-solver-only setting; it
    must not silently validate inside a coordinator:/worker: role sub-dict
    (which _load_and_apply_config never reads it from)."""

    def test_top_level_island_detection_is_valid(self):
        from distributed.cli import _validate_solver_yaml_keys

        # Must not raise.
        _validate_solver_yaml_keys({'island_detection': 'schur_bfs'}, 'solver')

    def test_island_detection_in_role_subdict_raises(self):
        from distributed.cli import _validate_solver_yaml_keys

        with pytest.raises(ValueError, match='island_detection'):
            _validate_solver_yaml_keys(
                {'coordinator': {'island_detection': 'schur_bfs'}}, 'solver',
            )

    def test_island_detection_excluded_from_valid_role_keys(self):
        from distributed.cli import _VALID_ROLE_YAML_KEYS

        assert 'island_detection' not in _VALID_ROLE_YAML_KEYS

    def test_island_detection_in_worker_subdict_raises(self):
        from distributed.cli import _validate_solver_yaml_keys

        with pytest.raises(ValueError, match='island_detection'):
            _validate_solver_yaml_keys(
                {'worker': {'island_detection': 'summaries'}}, 'solver',
            )
