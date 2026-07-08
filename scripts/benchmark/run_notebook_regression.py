#!/usr/bin/env python3
"""Notebook regression harness: execute notebooks headless, extract metrics, compare.

Executes a set of four IR-drop analysis notebooks and extracts key numeric
metrics from their output cells by regex.  In ``--capture`` mode the extracted
metrics are written as baseline JSONs; in ``--compare`` mode the notebooks are
re-executed and metrics are compared against the stored baselines (exact match
on extracted values, which are already quantized to ~1 e-3 mV precision by the
print statements).

Notebooks executed (all relative to project root)::

    notebooks/dynamic_irdrop_decomposition.ipynb
    notebooks/transient_analysis_validation.ipynb
    notebooks/distributed_transient_analysis_validation.ipynb
    notebooks/distributed_dynamic_irdrop_decomposition.ipynb

Usage::

    # Capture baselines (executes all notebooks, writes JSONs)
    python scripts/benchmark/run_notebook_regression.py \\
        --capture scripts/benchmark/baselines/

    # Smoke-test a single fast notebook
    python scripts/benchmark/run_notebook_regression.py \\
        --capture scripts/benchmark/baselines/ \\
        --notebooks transient_analysis_validation

    # Compare (re-execute, extract, diff against baselines)
    python scripts/benchmark/run_notebook_regression.py \\
        --compare scripts/benchmark/baselines/

    # Compare a single notebook
    python scripts/benchmark/run_notebook_regression.py \\
        --compare scripts/benchmark/baselines/ \\
        --notebooks transient_analysis_validation
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root / Python path bootstrap
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_NOTEBOOKS_DIR = _PROJECT_ROOT / 'notebooks'

# Workers (Ray) and nbconvert both inherit the driver cwd; set it to the
# project root so that relative paths like 'netlist/...' resolve correctly
# (see root CLAUDE.md "Notebook cwd for Ray" pitfall).
os.chdir(_PROJECT_ROOT)

if str(_PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / 'src'))


# ---------------------------------------------------------------------------
# Notebook registry
# ---------------------------------------------------------------------------

#: Short name → (notebook path relative to project root, kernel cwd)
# Kernel cwd matters: notebooks using Path('../netlist/...') need cwd=notebooks/;
# notebooks using Path('netlist/...') need cwd=project root.
# distributed notebooks os.chdir to project root themselves; we set it here
# so Ray workers also inherit the right cwd from the start.
NOTEBOOK_REGISTRY: dict[str, tuple[str, str]] = {
    'dynamic_irdrop_decomposition': (
        'notebooks/dynamic_irdrop_decomposition.ipynb',
        str(_PROJECT_ROOT),      # uses Path.cwd() walk-up to find pyproject.toml
    ),
    'transient_analysis_validation': (
        'notebooks/transient_analysis_validation.ipynb',
        str(_NOTEBOOKS_DIR),     # uses Path('../netlist/netlist_sampled')
    ),
    'distributed_transient_analysis_validation': (
        'notebooks/distributed_transient_analysis_validation.ipynb',
        # kernel_cwd must be notebooks/ so the fallback os.chdir(Path('..'))
        # (used when __file__ is undefined under nbclient) lands on the project
        # root rather than the project's parent directory.
        str(_NOTEBOOKS_DIR),
    ),
    'distributed_dynamic_irdrop_decomposition': (
        'notebooks/distributed_dynamic_irdrop_decomposition.ipynb',
        # Same reason: Path('..') relative to notebooks/ → project root.
        str(_NOTEBOOKS_DIR),
    ),
}

_DEFAULT_NOTEBOOKS = list(NOTEBOOK_REGISTRY.keys())
_TIMEOUT_SECONDS   = 3600  # per-notebook execution timeout


# ---------------------------------------------------------------------------
# Metric extraction: regex rules per notebook
# ---------------------------------------------------------------------------

# Each rule is (metric_name, pattern, converter).
# The pattern is matched against the *full text* of the notebook's stdout
# output cells (joined).  The first capturing group is the extracted value.

_FLOAT = float
_INT   = int


def _mk_rules(rules):
    """Return list of (name, compiled_re, converter) triples.

    Uses MULTILINE + DOTALL so ``.*?`` spans newlines (needed for multi-line
    output blocks like "Transient (BE) Results\\n  Peak IR-drop: ...").
    """
    return [(name, re.compile(pat, re.MULTILINE | re.DOTALL), conv)
            for name, pat, conv in rules]


def _mk_line_rules(rules):
    """Return list of (name, compiled_re, converter) triples, anchored to one line.

    Compiled with MULTILINE only (no DOTALL).  Patterns must use ``[ \\t]+``
    rather than ``\\s+`` between columns so whitespace cannot cross a newline.
    This prevents the column-count bleed where a short row's trailing ``\\s+``
    consumes the newline and matches the next row's leading rank digit as an
    extra numeric column.
    """
    return [(name, re.compile(pat, re.MULTILINE), conv)
            for name, pat, conv in rules]


# --- transient_analysis_validation ---
_RULES_TRANSIENT = _mk_rules([
    # "Quasi-Static Results ...\n  Peak IR-drop: 62.5698 mV"
    ('qs_peak_ir_drop_mV',  r'Quasi-Static Results.*?Peak IR-drop:\s*([\d.]+)\s*mV',   _FLOAT),
    ('qs_peak_time_ns',     r'Quasi-Static Results.*?Peak time:\s*([\d.]+)\s*ns',       _FLOAT),
    # "Transient (BE) Results ...\n  Peak IR-drop: 50.9123 mV"
    ('be_peak_ir_drop_mV',  r'Transient \(BE\) Results.*?Peak IR-drop:\s*([\d.]+)\s*mV', _FLOAT),
    ('be_peak_time_ns',     r'Transient \(BE\) Results.*?Peak time:\s*([\d.]+)\s*ns',    _FLOAT),
    # "Transient (Trap) Results ...\n  Peak IR-drop: 78.5103 mV"
    ('trap_peak_ir_drop_mV',r'Transient \(Trap\) Results.*?Peak IR-drop:\s*([\d.]+)\s*mV', _FLOAT),
    ('trap_peak_time_ns',   r'Transient \(Trap\) Results.*?Peak time:\s*([\d.]+)\s*ns',    _FLOAT),
    # Comparison block printed at 6 dp: "Quasi-Static:     62.569770 mV"
    ('comparison_qs_mV',    r'Quasi-Static:\s+([\d.]+)\s+mV',       _FLOAT),
    ('comparison_be_mV',    r'Transient \(BE\):\s+([\d.]+)\s+mV',   _FLOAT),
    ('comparison_trap_mV',  r'Transient \(Trap\):\s+([\d.]+)\s+mV', _FLOAT),
    # Diff lines: "  Transient (BE):   11657.4509 uV"
    ('diff_be_uV',          r'Transient \(BE\):\s+([\d.]+)\s+uV',   _FLOAT),
    ('diff_trap_uV',        r'Transient \(Trap\):\s+([\d.]+)\s+uV', _FLOAT),
])

# --- dynamic_irdrop_decomposition ---
# Output format (from existing cell output):
#   "Analyzing instance 1/2: i_Inst037447:...\n  Window: ...\n  Near sources: ...\n
#    Peak IR-drop: 32.170 mV = 22.282 (near) + 9.889 (far)\n  Near contribution: 69.3%"
_RULES_DYNAMIC = _mk_rules([
    ('instance1_peak_mV',  r'Analyzing instance 1/\d+.*?Peak IR-drop:\s*([\d.]+)\s*mV',      _FLOAT),
    ('instance1_near_mV',  r'Analyzing instance 1/\d+.*?Peak IR-drop.*?=\s*([\d.]+)\s*\(near\)', _FLOAT),
    ('instance1_far_mV',   r'Analyzing instance 1/\d+.*?Peak IR-drop.*?\+\s*([\d.]+)\s*\(far\)', _FLOAT),
    ('instance1_near_pct', r'Analyzing instance 1/\d+.*?Near contribution:\s*([\d.]+)%',      _FLOAT),
    ('instance2_peak_mV',  r'Analyzing instance 2/\d+.*?Peak IR-drop:\s*([\d.]+)\s*mV',      _FLOAT),
    ('instance2_near_pct', r'Analyzing instance 2/\d+.*?Near contribution:\s*([\d.]+)%',      _FLOAT),
    ('avg_near_pct',       r'Average near contribution:\s*([\d.]+)%',                         _FLOAT),
    # Adjoint block (instance 1):
    #   "Running static adjoint...\n  Done in ...\n  IR-drop at T: 36.474 mV\n  Self-contribution: 6.742 mV (18.5%)"
    ('adj_static_irdrop_mV',       r'Running static adjoint.*?IR-drop at T:\s*([\d.]+)\s*mV',       _FLOAT),
    ('adj_static_self_mV',         r'Running static adjoint.*?Self-contribution:\s*([\d.]+)\s*mV',   _FLOAT),
    ('adj_static_self_pct',        r'Running static adjoint.*?Self-contribution:.*?\(([\d.]+)%\)',    _FLOAT),
    ('adj_static_attribution_eff', r'Running static adjoint.*?Attribution efficiency:\s*([\d.]+)',    _FLOAT),
    ('adj_dynamic_irdrop_mV',      r'Running dynamic adjoint.*?IR-drop at T:\s*([\d.]+)\s*mV',      _FLOAT),
    ('adj_dynamic_self_mV',        r'Running dynamic adjoint.*?Self-contribution:\s*([\d.]+)\s*mV',  _FLOAT),
    ('adj_dynamic_self_pct',       r'Running dynamic adjoint.*?Self-contribution:.*?\(([\d.]+)%\)',   _FLOAT),
    ('adj_dynamic_attribution_eff',r'Running dynamic adjoint.*?Attribution efficiency:\s*([\d.]+)',   _FLOAT),
    # Top-aggressor node names (rank-1 static and dynamic) from the first adjoint
    # comparison block.  Format (cell 32 output):
    #   "TOP-10 AGGRESSORS\n  Rank  Static Node ...  Dynamic Node ...\n  ---\n  1  <node>  mV  %  <node>  mV  %"
    # The lazy .*? with DOTALL stops at the first dashes line, so the first match
    # is always from the victim-1 adjoint block.
    ('adj_rank1_static_aggressor',
     r'TOP-10 AGGRESSORS.*?[-]{20,}\n\s*1\s+(\S+)',                           str),
    ('adj_rank1_dynamic_aggressor',
     r'TOP-10 AGGRESSORS.*?[-]{20,}\n\s*1\s+\S+\s+[\d.]+\s+[\d.]+\s+(\S+)', str),
]) + _mk_line_rules([
    # Table row (one per line): "1   i_Inst037447..  32.170  22.282  9.889  69.3  30.7"
    # Uses [ \t]+ (not \s+) so whitespace cannot bleed across lines.
    # The table has exactly 5 numeric columns: peak near far near% far%.
    ('table_rank1_peak_mV', r'^[ \t]*1[ \t]+\S+[ \t]+([\d.]+)[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+[\d.]+', _FLOAT),
    ('table_rank1_near_mV', r'^[ \t]*1[ \t]+\S+[ \t]+[\d.]+[ \t]+([\d.]+)[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+[\d.]+', _FLOAT),
    ('table_rank1_far_mV',  r'^[ \t]*1[ \t]+\S+[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+([\d.]+)[ \t]+[\d.]+[ \t]+[\d.]+', _FLOAT),
    ('table_rank1_near_pct',r'^[ \t]*1[ \t]+\S+[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+([\d.]+)[ \t]+[\d.]+', _FLOAT),
    # Table row rank 2:  "2   i_Inst011524..  25.916  23.376  2.540  90.2  9.8"
    ('table_rank2_peak_mV', r'^[ \t]*2[ \t]+\S+[ \t]+([\d.]+)[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+[\d.]+', _FLOAT),
    ('table_rank2_near_mV', r'^[ \t]*2[ \t]+\S+[ \t]+[\d.]+[ \t]+([\d.]+)[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+[\d.]+', _FLOAT),
    ('table_rank2_far_mV',  r'^[ \t]*2[ \t]+\S+[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+([\d.]+)[ \t]+[\d.]+[ \t]+[\d.]+', _FLOAT),
    ('table_rank2_near_pct',r'^[ \t]*2[ \t]+\S+[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+([\d.]+)[ \t]+[\d.]+', _FLOAT),
])

# --- distributed_transient_analysis_validation ---
_RULES_DIST_TRANSIENT = _mk_rules([
    # Method comparison table:
    # "Quasi-Static          62.5698       2.28"
    ('qs_peak_ir_drop_mV',   r'Quasi-Static\s+([\d.]+)\s+[\d.]+',   _FLOAT),
    ('qs_peak_time_ns',      r'Quasi-Static\s+[\d.]+\s+([\d.]+)',    _FLOAT),
    ('be_peak_ir_drop_mV',   r'Transient \(BE\)\s+([\d.]+)\s+[\d.]+', _FLOAT),
    ('be_peak_time_ns',      r'Transient \(BE\)\s+[\d.]+\s+([\d.]+)', _FLOAT),
    ('trap_peak_ir_drop_mV', r'Transient \(Trap\)\s+([\d.]+)\s+[\d.]+', _FLOAT),
    ('trap_peak_time_ns',    r'Transient \(Trap\)\s+[\d.]+\s+([\d.]+)', _FLOAT),
    # "Max:::::::::  78.5102  78.5102  78.5102"
    ('peak_max_qs_mV',  r'Max[:]+\s+([\d.]+)\s+[\d.]+\s+[\d.]+', _FLOAT),
    ('peak_max_be_mV',  r'Max[:]+\s+[\d.]+\s+([\d.]+)\s+[\d.]+', _FLOAT),
    ('peak_max_trap_mV',r'Max[:]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)', _FLOAT),
    # "QS vs BE:   11657.45 uV"
    ('diff_qs_vs_be_uV',   r'QS vs BE:\s+([\d.]+)\s+uV',   _FLOAT),
    ('diff_qs_vs_trap_uV', r'QS vs Trap:\s+([\d.]+)\s+uV', _FLOAT),
    ('diff_be_vs_trap_uV', r'BE vs Trap:\s+([\d.]+)\s+uV', _FLOAT),
])

# --- distributed_dynamic_irdrop_decomposition ---
_RULES_DIST_DYNAMIC = _mk_rules([
    # Summary table (same format as dynamic_irdrop_decomposition)
    ('peak_ir_drop_mV',     r'Peak IR-drop:\s*([\d.]+)\s*mV',       _FLOAT),
    ('avg_near_pct',        r'Average near contribution:\s*([\d.]+)%', _FLOAT),
    # Adjoint block (victim #1):
    # "  Static:  IR-drop=36.474 mV, Self=6.742 mV (18.5%), eff=1.0000"
    ('adj_static_irdrop_mV',        r'Static:\s+IR-drop=([\d.]+)\s+mV', _FLOAT),
    ('adj_static_self_mV',          r'Static:.*?Self=([\d.]+)\s+mV',    _FLOAT),
    ('adj_static_self_pct',         r'Static:.*?Self=[\d.]+\s+mV\s+\(([\d.]+)%\)', _FLOAT),
    ('adj_static_attribution_eff',  r'Static:.*?eff=([\d.]+)',           _FLOAT),
    ('adj_dynamic_irdrop_mV',       r'Dynamic:\s+IR-drop=([\d.]+)\s+mV', _FLOAT),
    ('adj_dynamic_self_mV',         r'Dynamic:.*?Self=([\d.]+)\s+mV',   _FLOAT),
    ('adj_dynamic_self_pct',        r'Dynamic:.*?Self=[\d.]+\s+mV\s+\(([\d.]+)%\)', _FLOAT),
    ('adj_dynamic_attribution_eff', r'Dynamic:.*?eff=([\d.]+)',          _FLOAT),
    # Top-aggressor node names from the adjoint comparison table (cell 14 output).
    # Format: "TOP-10 AGGRESSORS\n  Rank  Static Node ...\n  ---\n  1  <node>  mV  %  <node>  mV  %"
    ('adj_rank1_static_aggressor',
     r'TOP-10 AGGRESSORS.*?[-]{20,}\n\s*1\s+(\S+)',                           str),
    ('adj_rank1_dynamic_aggressor',
     r'TOP-10 AGGRESSORS.*?[-]{20,}\n\s*1\s+\S+\s+[\d.]+\s+[\d.]+\s+(\S+)', str),
]) + _mk_line_rules([
    # Table rows (one per line): "1  <node>  32.170  22.282  9.889  69.3  30.7"
    # Uses [ \t]+ (not \s+) so whitespace cannot bleed across lines.
    # The summary table has exactly 5 numeric columns: peak near far near% far%.
    # Any earlier worst-nodes table (Rank Node X Y Peak Time = 4 numerics) does
    # NOT match because it lacks the 5th column required here.
    ('table_rank1_peak_mV', r'^[ \t]*1[ \t]+\S+[ \t]+([\d.]+)[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+[\d.]+', _FLOAT),
    ('table_rank1_near_mV', r'^[ \t]*1[ \t]+\S+[ \t]+[\d.]+[ \t]+([\d.]+)[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+[\d.]+', _FLOAT),
    ('table_rank1_far_mV',  r'^[ \t]*1[ \t]+\S+[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+([\d.]+)[ \t]+[\d.]+[ \t]+[\d.]+', _FLOAT),
    ('table_rank1_near_pct',r'^[ \t]*1[ \t]+\S+[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+([\d.]+)[ \t]+[\d.]+', _FLOAT),
    # Table row rank 2:  "2  <node>  25.916  23.376  2.540  90.2  9.8"
    ('table_rank2_peak_mV', r'^[ \t]*2[ \t]+\S+[ \t]+([\d.]+)[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+[\d.]+', _FLOAT),
    ('table_rank2_near_mV', r'^[ \t]*2[ \t]+\S+[ \t]+[\d.]+[ \t]+([\d.]+)[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+[\d.]+', _FLOAT),
    ('table_rank2_far_mV',  r'^[ \t]*2[ \t]+\S+[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+([\d.]+)[ \t]+[\d.]+[ \t]+[\d.]+', _FLOAT),
    ('table_rank2_near_pct',r'^[ \t]*2[ \t]+\S+[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+[\d.]+[ \t]+([\d.]+)[ \t]+[\d.]+', _FLOAT),
])

_NOTEBOOK_RULES: dict[str, list] = {
    'dynamic_irdrop_decomposition':              _RULES_DYNAMIC,
    'transient_analysis_validation':             _RULES_TRANSIENT,
    'distributed_transient_analysis_validation': _RULES_DIST_TRANSIENT,
    'distributed_dynamic_irdrop_decomposition':  _RULES_DIST_DYNAMIC,
}


# ---------------------------------------------------------------------------
# Notebook execution
# ---------------------------------------------------------------------------

def _collect_stdout(nb) -> str:
    """Return all stream/text output from executed notebook cells as one string."""
    import nbformat as _nbf
    parts = []
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        for output in cell.get('outputs', []):
            if output.output_type == 'stream':
                text = output.get('text', '')
                parts.append(text if isinstance(text, str) else ''.join(text))
            elif output.output_type in ('execute_result', 'display_data'):
                td = output.get('data', {})
                text = td.get('text/plain', '')
                parts.append(text if isinstance(text, str) else ''.join(text))
    return '\n'.join(parts)


def execute_notebook(
    nb_path: Path,
    scratch_dir: Path,
    kernel_cwd: str | None = None,
    timeout: int = _TIMEOUT_SECONDS,
    verbose: bool = False,
) -> tuple[Any, str]:
    """Execute a notebook and return (executed_nb, stdout_text).

    Raises on cell error with the offending cell's traceback.
    Writes the executed copy to ``scratch_dir`` (NOT over the original).

    Args:
        nb_path: Absolute path to the notebook.
        scratch_dir: Directory for the executed copy.
        kernel_cwd: Working directory for the kernel process.  If None,
            defaults to the project root.  Matters for notebooks that use
            relative paths (e.g. ``Path('../netlist/...')`` needs cwd=notebooks/).
        timeout: Per-notebook timeout in seconds.
        verbose: Print progress messages.
    """
    import nbformat
    import nbclient
    from nbclient.exceptions import CellExecutionError

    if kernel_cwd is None:
        kernel_cwd = str(_PROJECT_ROOT)

    out_path = scratch_dir / nb_path.name.replace('.ipynb', '_executed.ipynb')

    if verbose:
        print(f"  Executing {nb_path.name} (kernel cwd={kernel_cwd}) ...", flush=True)

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # nbclient executes in a subprocess kernel; set metadata.path so the
    # kernel process starts in the right cwd (Ray workers inherit it).
    client = nbclient.NotebookClient(
        nb,
        timeout=timeout,
        kernel_name='python3',
        resources={'metadata': {'path': kernel_cwd}},
        allow_errors=False,     # raise CellExecutionError on cell error
    )

    try:
        client.execute()
    except CellExecutionError as exc:
        # CellExecutionError in nbclient 0.11 has .traceback / .ename / .evalue
        # (no .cell_source attribute).  str(exc) contains the full formatted
        # traceback including the cell source.
        raise RuntimeError(
            f"Notebook {nb_path.name} failed at a cell.\n"
            f"{str(exc)}"
        ) from exc
    finally:
        # Always save executed copy so partial output is inspectable
        with open(out_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        if verbose:
            print(f"  Executed copy saved to {out_path}", flush=True)

    stdout = _collect_stdout(nb)
    return nb, stdout


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def extract_metrics(nb_name: str, stdout: str) -> dict[str, Any]:
    """Apply regex rules for *nb_name* against *stdout*.  Returns {name: value}.

    Raises ``RuntimeError`` if fewer than half the rules matched (guard against
    silent regex drift or unexecuted notebooks producing no output).
    """
    rules = _NOTEBOOK_RULES.get(nb_name, [])
    metrics: dict[str, Any] = {}
    for name, pattern, conv in rules:
        m = pattern.search(stdout)
        if m:
            try:
                metrics[name] = conv(m.group(1))
            except (ValueError, IndexError):
                pass  # partial match — skip

    if rules:
        min_expected = max(1, len(rules) // 2)
        if len(metrics) < min_expected:
            missed = [name for name, _pat, _conv in rules if name not in metrics]
            raise RuntimeError(
                f"extract_metrics({nb_name!r}): only {len(metrics)}/{len(rules)} rules "
                f"matched (minimum {min_expected} required).  "
                f"Rules that did not match: {missed}.  "
                f"The notebook may not have been executed, its output format may have "
                f"changed, or the regex patterns need updating."
            )

    return metrics


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

#: Per-metric-type absolute tolerances for float comparisons.
#
# Values are extracted from notebook print statements, so the implicit
# precision is determined by the format string (e.g. "32.170 mV" → 0.001 mV
# resolution).  BLAS/summation-order nondeterminism across builds can flip the
# last printed digit, so we allow 2× the last digit as tolerance instead of
# requiring exact string equality.  String metrics (node names, aggressor IDs)
# still use exact match.
#
# Key is a suffix of the metric name; first matching suffix wins.
_METRIC_ABSTOL: list[tuple[str, float]] = [
    # suffix              abstol
    ('_mV',              0.002),    # 3 dp in mV → last digit 0.001
    ('_ns',              0.02),     # 2 dp in ns → last digit 0.01
    ('_uV',              0.05),     # 2 dp in µV → last digit 0.01 (allow 5× for sums)
    ('_pct',             0.2),      # 1 dp in % → last digit 0.1
    ('_eff',             0.0005),   # 4 dp attribution efficiency
]
_METRIC_DEFAULT_RELTOL = 1e-3       # fallback for unrecognised numeric keys


def _metric_close(key: str, bval: Any, cval: Any) -> bool:
    """Return True if *bval* and *cval* are within tolerance for metric *key*.

    String values use exact equality.  Numeric values use absolute tolerance
    looked up from ``_METRIC_ABSTOL`` by metric-name suffix; unrecognised keys
    fall back to a relative tolerance of ``_METRIC_DEFAULT_RELTOL``.
    """
    if not isinstance(bval, (int, float)):
        return bval == cval
    if not isinstance(cval, (int, float)):
        return False

    for suffix, abstol in _METRIC_ABSTOL:
        if key.endswith(suffix):
            return abs(float(bval) - float(cval)) <= abstol

    # Fallback: relative tolerance on unrecognised numeric keys
    scale = max(abs(float(bval)), abs(float(cval)), 1e-12)
    return abs(float(bval) - float(cval)) <= _METRIC_DEFAULT_RELTOL * scale


def compare_metrics(
    nb_name: str,
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Compare current metrics against baseline.  Returns (all_pass, diff_lines).

    Numeric values are compared with per-metric-type absolute tolerances (see
    ``_METRIC_ABSTOL``) rather than exact equality so that last-digit
    nondeterminism from BLAS/parallel reductions does not produce spurious
    failures.  String values (aggressor node names) still use exact equality.
    """
    all_pass  = True
    diff_lines: list[str] = []

    all_keys = sorted(set(baseline) | set(current))
    for k in all_keys:
        bval = baseline.get(k)
        cval = current.get(k)

        if bval is None:
            diff_lines.append(f"  NEW     {k} = {cval}")
        elif cval is None:
            diff_lines.append(f"  MISSING {k} (baseline={bval})")
            all_pass = False
        elif not _metric_close(k, bval, cval):
            diff_lines.append(f"  DIFF    {k}: baseline={bval!r} current={cval!r}")
            all_pass = False
        else:
            diff_lines.append(f"  OK      {k} = {cval!r}")

    return all_pass, diff_lines


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------

def baseline_path(baseline_dir: Path, nb_name: str) -> Path:
    return baseline_dir / f'{nb_name}_metrics.json'


def write_baseline(baseline_dir: Path, nb_name: str, metrics: dict) -> None:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    p = baseline_path(baseline_dir, nb_name)
    with open(p, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Baseline written: {p}")


def read_baseline(baseline_dir: Path, nb_name: str) -> dict:
    p = baseline_path(baseline_dir, nb_name)
    if not p.exists():
        raise FileNotFoundError(f"No baseline found at {p} — run --capture first")
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main modes
# ---------------------------------------------------------------------------

def run_capture(
    nb_names: list[str],
    baseline_dir: Path,
    scratch_dir: Path,
    timeout: int,
    verbose: bool,
) -> int:
    """Execute notebooks and capture baseline metrics.  Returns exit code."""
    failures = 0
    for name in nb_names:
        nb_rel, kernel_cwd = NOTEBOOK_REGISTRY[name]
        nb_path = _PROJECT_ROOT / nb_rel
        print(f"\n=== Capturing {name} ===", flush=True)
        try:
            _nb, stdout = execute_notebook(
                nb_path, scratch_dir,
                kernel_cwd=kernel_cwd,
                timeout=timeout,
                verbose=verbose,
            )
            metrics = extract_metrics(name, stdout)
            print(f"  Extracted {len(metrics)} metrics: {list(metrics.keys())}")
            for k, v in metrics.items():
                print(f"    {k} = {v!r}")
            write_baseline(baseline_dir, name, metrics)
        except Exception:
            print(f"  FAILED: {name}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            failures += 1
    return 0 if failures == 0 else 1


def run_compare(
    nb_names: list[str],
    baseline_dir: Path,
    scratch_dir: Path,
    timeout: int,
    verbose: bool,
) -> int:
    """Execute notebooks, extract metrics, compare against baselines.  Returns exit code."""
    failures = 0
    for name in nb_names:
        nb_rel, kernel_cwd = NOTEBOOK_REGISTRY[name]
        nb_path = _PROJECT_ROOT / nb_rel
        print(f"\n=== Comparing {name} ===", flush=True)
        try:
            baseline = read_baseline(baseline_dir, name)
        except FileNotFoundError as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            failures += 1
            continue

        try:
            _nb, stdout = execute_notebook(
                nb_path, scratch_dir,
                kernel_cwd=kernel_cwd,
                timeout=timeout,
                verbose=verbose,
            )
        except Exception:
            print(f"  FAILED (execution): {name}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            failures += 1
            continue

        try:
            metrics = extract_metrics(name, stdout)
        except RuntimeError as exc:
            print(f"  FAILED (extraction): {exc}", file=sys.stderr)
            failures += 1
            continue
        print(f"  Extracted {len(metrics)} metrics")

        passed, diff_lines = compare_metrics(name, metrics, baseline)
        for line in diff_lines:
            print(line)

        if passed:
            print(f"  PASS: all {len(diff_lines)} metrics match baseline")
        else:
            print(f"  FAIL: metric mismatch(es) in {name}", file=sys.stderr)
            failures += 1

    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Notebook regression harness: execute + compare key numeric metrics.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '--capture', metavar='BASELINE_DIR',
        help='Execute notebooks and write baseline JSONs to this directory',
    )
    mode.add_argument(
        '--compare', metavar='BASELINE_DIR',
        help='Re-execute notebooks and compare against baselines in this directory',
    )

    p.add_argument(
        '--notebooks', nargs='+',
        choices=list(NOTEBOOK_REGISTRY.keys()),
        default=_DEFAULT_NOTEBOOKS,
        metavar='NB_NAME',
        help=(
            'Notebook short names to process '
            '(default: all four; choose from: '
            + ', '.join(NOTEBOOK_REGISTRY.keys()) + ')'
        ),
    )
    p.add_argument(
        '--scratch-dir', metavar='DIR',
        default=str(_SCRIPT_DIR / '_executed_notebooks'),
        help='Directory for executed notebook copies (never overwrites originals)',
    )
    p.add_argument(
        '--timeout', type=int, default=_TIMEOUT_SECONDS,
        help='Per-notebook execution timeout in seconds',
    )
    p.add_argument('--verbose', '-v', action='store_true')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    mode_dir = args.capture or args.compare
    baseline_dir = Path(mode_dir)

    nb_names = args.notebooks

    print(f"Notebook regression harness — project root: {_PROJECT_ROOT}")
    print(f"Notebooks: {nb_names}")
    print(f"Scratch dir: {scratch}")
    print(f"Baseline dir: {baseline_dir}")

    if args.capture:
        exit_code = run_capture(nb_names, baseline_dir, scratch,
                                timeout=args.timeout, verbose=args.verbose)
    else:
        exit_code = run_compare(nb_names, baseline_dir, scratch,
                                timeout=args.timeout, verbose=args.verbose)

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
