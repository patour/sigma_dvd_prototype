#!/usr/bin/env python3
"""
PDN IR-Drop/Ground-Bounce Solver - DC voltage analysis for power delivery networks.

This solver computes static voltage drop across power delivery networks by solving
the DC resistive network equations. It delegates to UnifiedIRDropSolver for the
actual matrix building and solving, while orchestrating multi-net analysis and
report generation.

Features:
- Automatic floating node/island detection and removal (via UnifiedIRDropSolver)
- Multi-net support with independent solves
- Direct sparse solve using cholmod/splu backend with configurable parameters
- Detailed timing and memory profiling in verbose mode
- Layer-wise voltage heatmap visualization
- Top-K worst IR-drop reporting with instance names
- Config file support for CLI parameters

Usage Examples:
    # Basic IR-drop solve
    from parser.netlist import NetlistParser
    from solver.pdn_solver import PDNSolver

    parser = NetlistParser('./netlist_data')
    graph = parser.parse()

    solver = PDNSolver(graph)
    results = solver.solve()

    # With cholmod configuration
    solver = PDNSolver(graph, use_cholmod=True, cholmod_mode='supernodal')
    results = solver.solve()

    # Using config file
    python pdn_solver.py --config solver_config.yaml --netlist-dir ./netlist_data

Author: Based on mpower power grid analysis
Date: December 12, 2025
"""

import sys
import os
import logging
import argparse
import time
import tracemalloc
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field, asdict

try:
    import numpy as np
except ImportError:
    print("ERROR: NumPy is required. Install with: pip install numpy")
    sys.exit(1)

# Optional YAML support for config files
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from graph.rx_graph import RustworkxGraphWrapper, RustworkxMultiDiGraphWrapper

# Import unified solver infrastructure
try:
    from model.factory import create_model_from_pdn
    from .unified_solver import UnifiedIRDropSolver
    from model.solver_results import UnifiedSolveResult
    from model.unified_model import UnifiedPowerGridModel
    from .unified_solver import (
        set_use_cholmod, get_use_cholmod, get_active_backend,
        set_cholmod_mode, get_cholmod_mode,
        set_cholmod_ordering, get_cholmod_ordering,
        set_cholmod_use_long, get_cholmod_use_long,
        HAS_CHOLMOD,
    )
except ImportError:
    print("ERROR: core module required. Make sure core/ package is available.")
    sys.exit(1)

# Import plotter module
from visualization.pdn_plotter import PDNPlotter

# Import reports module
from reports.floating_nodes import collect_floating_nodes_flat, generate_floating_nodes_report


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TimingStats:
    """Timing statistics for solve operations."""
    model_creation: float = 0.0
    current_extraction: float = 0.0
    matrix_factorization: float = 0.0
    solve: float = 0.0
    voltage_storage: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    peak_mb: float = 0.0
    current_mb: float = 0.0
    model_mb: float = 0.0
    solve_mb: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class PDNSolveResult(UnifiedSolveResult):
    """Extended solve result with PDN-specific statistics.

    Extends UnifiedSolveResult with additional statistics relevant to
    PDN analysis including node counts, current injection, island detection,
    and detailed timing/memory profiling.

    Attributes:
        voltages: Node -> voltage mapping (inherited)
        ir_drop: Node -> IR-drop mapping (inherited)
        nominal_voltage: Reference Vdd (inherited)
        net_name: Net name (inherited)
        metadata: Additional metadata (inherited)

        # Node statistics
        num_nodes: Total number of nodes in solution
        num_free_nodes: Number of non-pad nodes
        num_vsrc_nodes: Number of voltage source nodes

        # Current statistics
        num_isources: Number of current sources
        total_current_ma: Total injected current in mA

        # Island statistics
        islands_removed: Number of floating islands removed
        nodes_removed: Number of nodes removed due to islands

        # Voltage statistics
        max_voltage: Maximum voltage in solution
        min_voltage: Minimum voltage in solution
        avg_voltage: Average voltage
        max_ir_drop: Maximum IR-drop
        avg_ir_drop: Average IR-drop

        # Profiling
        timings: Detailed timing breakdown
        memory: Memory usage statistics
        backend: Solver backend used ('cholmod' or 'splu')
    """
    # Node statistics
    num_nodes: int = 0
    num_free_nodes: int = 0
    num_vsrc_nodes: int = 0

    # Current statistics
    num_isources: int = 0
    total_current_ma: float = 0.0

    # Island statistics
    islands_removed: int = 0
    nodes_removed: int = 0

    # Voltage statistics
    max_voltage: float = 0.0
    min_voltage: float = 0.0
    avg_voltage: float = 0.0
    max_ir_drop: float = 0.0
    avg_ir_drop: float = 0.0

    # Profiling
    timings: TimingStats = field(default_factory=TimingStats)
    memory: MemoryStats = field(default_factory=MemoryStats)
    backend: str = 'unknown'

    # Model reference (for report generation)
    model: Optional['UnifiedPowerGridModel'] = None


@dataclass
class SolveResults:
    """Complete solve results for all nets."""
    net_results: Dict[str, PDNSolveResult] = field(default_factory=dict)
    total_solve_time: float = 0.0
    total_memory_peak_mb: float = 0.0


# ============================================================================
# Solver Class
# ============================================================================

class PDNSolver:
    """
    DC IR-drop and ground-bounce solver for power delivery networks.

    Delegates to UnifiedIRDropSolver for matrix building and solving,
    while orchestrating multi-net analysis and report generation.

    Solves the linear system G*V = I where:
    - G is the conductance matrix (from resistors)
    - V is the vector of node voltages (unknowns)
    - I is the current injection vector (from current sources)

    Voltage sources are treated as boundary conditions with fixed voltages.
    """

    def __init__(
        self,
        graph: RustworkxMultiDiGraphWrapper,
        verbose: bool = False,
        net_filter: Optional[str] = None,
        # Solver backend options
        use_cholmod: Optional[bool] = None,
        cholmod_mode: str = 'auto',
        cholmod_ordering: str = 'default',
        cholmod_use_long: Optional[bool] = None,
        # Plotting options
        anisotropic_bins: bool = True,
        bin_aspect_ratio: int = 50,
        layer_orientations: Optional[Dict[str, str]] = None,
        # Profiling options
        profile_memory: bool = False,
    ):
        """
        Initialize PDN solver.

        Args:
            graph: Parsed PDN graph from NetlistParser
            verbose: Enable verbose logging with timing details
            net_filter: Solve only specific net (e.g., 'VDD')

            use_cholmod: If True, use cholmod (Cholesky). If False, use splu.
                        If None (default), auto-detect based on availability.
            cholmod_mode: Cholmod factorization mode: 'auto', 'simplicial', 'supernodal'
            cholmod_ordering: Fill-reducing ordering: 'default', 'natural', 'amd',
                            'metis', 'nesdis', 'colamd', 'best'
            cholmod_use_long: If True, force 64-bit indices. If False, force 32-bit.
                            If None (default), auto-detect based on matrix size.

            anisotropic_bins: Enable orientation-aware anisotropic binning for heatmaps
            bin_aspect_ratio: Aspect ratio for anisotropic bins (default: 50)
            layer_orientations: Manual layer orientation override dict

            profile_memory: Enable memory profiling (slower but detailed)
        """
        self.graph = graph
        self.net_filter = net_filter
        self.anisotropic_bins = anisotropic_bins
        self.bin_aspect_ratio = bin_aspect_ratio
        self.layer_orientations = layer_orientations or {}
        self._verbose = verbose
        self._profile_memory = profile_memory

        # Store solver backend configuration
        self._use_cholmod = use_cholmod
        self._cholmod_mode = cholmod_mode
        self._cholmod_ordering = cholmod_ordering
        self._cholmod_use_long = cholmod_use_long

        # Apply solver configuration globally
        self._configure_solver_backend()

        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

        # Extract metadata
        self.net_connectivity = graph.graph.get('net_connectivity', {})
        self.vsrc_dict = graph.graph.get('vsrc_dict', {})
        self.parameters = graph.graph.get('parameters', {})
        self.instance_node_map = graph.graph.get('instance_node_map', {})
        self.vsrc_nodes_global = graph.graph.get('vsrc_nodes', set())

        # Results storage
        self.results = SolveResults()

        backend = get_active_backend()
        self.logger.info(f"Initialized PDN solver (backend: {backend})")
        if verbose and backend == 'cholmod':
            self.logger.info(f"  cholmod mode: {cholmod_mode}, ordering: {cholmod_ordering}")

    def _configure_solver_backend(self):
        """Configure the global solver backend settings."""
        if self._use_cholmod is not None:
            try:
                set_use_cholmod(self._use_cholmod)
            except ImportError as e:
                if self._use_cholmod:
                    raise
                # If False was requested, that's fine even without cholmod

        if self._cholmod_mode != 'auto':
            set_cholmod_mode(self._cholmod_mode)

        if self._cholmod_ordering != 'default':
            set_cholmod_ordering(self._cholmod_ordering)

        if self._cholmod_use_long is not None:
            set_cholmod_use_long(self._cholmod_use_long)

    def solve(self) -> SolveResults:
        """
        Solve IR-drop for all nets in the graph.

        Returns:
            SolveResults object with PDNSolveResult for each net
        """
        start_time = time.perf_counter()

        # Start memory profiling if enabled
        if self._profile_memory:
            tracemalloc.start()

        self.logger.info("=" * 70)
        self.logger.info("Starting PDN IR-Drop/Ground-Bounce Analysis")
        self.logger.info("=" * 70)

        # Get list of nets to solve
        all_nets = list(self.net_connectivity.keys())

        if not all_nets:
            self.logger.warning("No nets found in graph. Nothing to solve.")
            return self.results

        # Filter nets if specific net was requested
        if self.net_filter:
            net_filter_lower = self.net_filter.lower()
            nets_to_solve = [net for net in all_nets if net.lower() == net_filter_lower]

            if not nets_to_solve:
                self.logger.warning(f"Requested net '{self.net_filter}' not found in graph.")
                self.logger.info(f"Available nets: {', '.join(all_nets)}")
                return self.results

            self.logger.info(f"Filtering to requested net: {nets_to_solve[0]}")
        else:
            nets_to_solve = all_nets

        self.logger.info(f"Solving {len(nets_to_solve)} net(s): {', '.join(nets_to_solve)}")

        # Solve each net independently
        for net_name in nets_to_solve:
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Solving net: {net_name}")
            self.logger.info(f"{'='*70}")

            try:
                result = self._solve_net(net_name)
                self.results.net_results[net_name] = result

                self.logger.info(f"Net {net_name} solved successfully")
                self._print_result_stats(result)

            except Exception as e:
                self.logger.error(f"Failed to solve net {net_name}: {e}")
                import traceback
                traceback.print_exc()

        self.results.total_solve_time = time.perf_counter() - start_time

        # Get peak memory if profiling
        if self._profile_memory:
            current, peak = tracemalloc.get_traced_memory()
            self.results.total_memory_peak_mb = peak / (1024 * 1024)
            tracemalloc.stop()

        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"All nets solved in {self.results.total_solve_time:.3f} seconds")
        if self._profile_memory:
            self.logger.info(f"Peak memory usage: {self.results.total_memory_peak_mb:.1f} MB")
        self.logger.info(f"{'='*70}")

        return self.results

    def _solve_net(self, net_name: str) -> PDNSolveResult:
        """Solve IR-drop for a single net using UnifiedIRDropSolver."""
        timings = TimingStats()
        memory = MemoryStats()

        # Get nodes for this net (for logging)
        net_nodes_list = self.net_connectivity.get(net_name, [])
        self.logger.info(f"Net has {len(net_nodes_list)} nodes")

        # Memory snapshot before model creation
        if self._profile_memory:
            tracemalloc.reset_peak()

        # Step 1: Create unified model
        t0 = time.perf_counter()
        try:
            model = create_model_from_pdn(self.graph, net_name, lazy_factor=True)
        except ValueError as e:
            self.logger.warning(f"Could not create model for net {net_name}: {e}")
            return self._create_empty_result(net_name)
        timings.model_creation = time.perf_counter() - t0

        if self._profile_memory:
            _, peak = tracemalloc.get_traced_memory()
            memory.model_mb = peak / (1024 * 1024)

        if self._verbose:
            self.logger.debug(f"  Model creation: {timings.model_creation*1000:.1f} ms")

        # Step 2: Extract current sources
        t0 = time.perf_counter()
        load_currents = model.extract_current_sources()
        timings.current_extraction = time.perf_counter() - t0

        if self._verbose:
            self.logger.debug(f"  Current extraction: {timings.current_extraction*1000:.1f} ms")

        # Step 3: Solve using UnifiedIRDropSolver
        t0 = time.perf_counter()
        solver = UnifiedIRDropSolver(model, verbose=self._verbose)
        base_result = solver.solve(load_currents)
        timings.solve = time.perf_counter() - t0

        if self._profile_memory:
            _, peak = tracemalloc.get_traced_memory()
            memory.solve_mb = peak / (1024 * 1024)
            memory.peak_mb = max(memory.model_mb, memory.solve_mb)

        if self._verbose:
            self.logger.debug(f"  Solve: {timings.solve*1000:.1f} ms")

        # Step 4: Store voltages in original graph
        t0 = time.perf_counter()
        self._store_voltages_from_result(net_name, base_result)
        timings.voltage_storage = time.perf_counter() - t0

        timings.total = (timings.model_creation + timings.current_extraction +
                        timings.solve + timings.voltage_storage)

        # Step 5: Build PDNSolveResult with statistics
        result = self._build_pdn_result(net_name, model, base_result, load_currents,
                                        timings, memory)

        return result

    def _store_voltages_from_result(self, net_name: str, result: UnifiedSolveResult):
        """Store voltages from UnifiedSolveResult in original graph nodes."""
        for node, voltage in result.voltages.items():
            if node in self.graph:
                self.graph.nodes_dict[node]['voltage'] = voltage

    def _build_pdn_result(
        self,
        net_name: str,
        model: UnifiedPowerGridModel,
        base_result: UnifiedSolveResult,
        load_currents: Dict[Any, float],
        timings: TimingStats,
        memory: MemoryStats,
    ) -> PDNSolveResult:
        """Build PDNSolveResult from base result and model."""
        # Get node counts
        pad_set = set(model.pad_nodes)
        free_nodes = [n for n in base_result.voltages if n not in pad_set and n != '0']

        # Voltage statistics
        free_voltages = [base_result.voltages[n] for n in free_nodes]
        free_drops = [base_result.ir_drop[n] for n in free_nodes]

        if free_voltages:
            max_voltage = float(np.max(free_voltages))
            min_voltage = float(np.min(free_voltages))
            avg_voltage = float(np.mean(free_voltages))
            max_ir_drop = float(np.max(free_drops)) if free_drops else 0.0
            avg_ir_drop = float(np.mean(free_drops)) if free_drops else 0.0
        else:
            max_voltage = min_voltage = avg_voltage = model.vdd
            max_ir_drop = avg_ir_drop = 0.0

        # Island stats
        island_stats = model.island_stats

        # Current statistics
        total_current = sum(abs(c) for c in load_currents.values())

        return PDNSolveResult(
            # Inherited from UnifiedSolveResult
            voltages=base_result.voltages,
            ir_drop=base_result.ir_drop,
            nominal_voltage=model.vdd,
            net_name=net_name,
            metadata=base_result.metadata,
            # Node statistics
            num_nodes=len(base_result.voltages),
            num_free_nodes=len(free_nodes),
            num_vsrc_nodes=len(pad_set),
            # Current statistics
            num_isources=len(load_currents),
            total_current_ma=total_current,
            # Island statistics
            islands_removed=island_stats.get('islands_removed', 0),
            nodes_removed=island_stats.get('nodes_removed', 0),
            # Voltage statistics
            max_voltage=max_voltage,
            min_voltage=min_voltage,
            avg_voltage=avg_voltage,
            max_ir_drop=max_ir_drop,
            avg_ir_drop=avg_ir_drop,
            # Profiling
            timings=timings,
            memory=memory,
            backend=get_active_backend(),
            # Model reference
            model=model,
        )

    def _create_empty_result(self, net_name: str) -> PDNSolveResult:
        """Create empty result for nets that couldn't be solved."""
        return PDNSolveResult(
            voltages={},
            ir_drop={},
            nominal_voltage=0.0,
            net_name=net_name,
        )

    def _print_result_stats(self, result: PDNSolveResult):
        """Print statistics for a solved net."""
        self.logger.info(f"\nNet Statistics:")
        self.logger.info(f"  Nodes: {result.num_nodes} total, {result.num_free_nodes} free, "
                        f"{result.num_vsrc_nodes} voltage source")
        self.logger.info(f"  Current sources: {result.num_isources}, "
                        f"total injection: {result.total_current_ma:.3f} mA")

        if result.islands_removed > 0:
            self.logger.info(f"  Removed {result.islands_removed} floating islands "
                           f"({result.nodes_removed} nodes)")

        self.logger.info(f"\nVoltage Results:")
        self.logger.info(f"  Nominal: {result.nominal_voltage:.6f} V")
        self.logger.info(f"  Range: [{result.min_voltage:.6f}, {result.max_voltage:.6f}] V")
        self.logger.info(f"  Average: {result.avg_voltage:.6f} V")

        if result.nominal_voltage > 0:
            drop_pct = result.max_ir_drop / result.nominal_voltage * 100
            avg_pct = result.avg_ir_drop / result.nominal_voltage * 100
            self.logger.info(f"\nIR-Drop Results:")
            self.logger.info(f"  Maximum: {result.max_ir_drop*1000:.3f} mV ({drop_pct:.2f}%)")
            self.logger.info(f"  Average: {result.avg_ir_drop*1000:.3f} mV ({avg_pct:.2f}%)")

        if self._verbose:
            self.logger.info(f"\nTiming Breakdown:")
            self.logger.info(f"  Model creation:     {result.timings.model_creation*1000:8.2f} ms")
            self.logger.info(f"  Current extraction: {result.timings.current_extraction*1000:8.2f} ms")
            self.logger.info(f"  Solve:              {result.timings.solve*1000:8.2f} ms")
            self.logger.info(f"  Voltage storage:    {result.timings.voltage_storage*1000:8.2f} ms")
            self.logger.info(f"  Total:              {result.timings.total*1000:8.2f} ms")
            self.logger.info(f"  Backend: {result.backend}")

            if self._profile_memory and result.memory.peak_mb > 0:
                self.logger.info(f"\nMemory Usage:")
                self.logger.info(f"  Model:  {result.memory.model_mb:8.1f} MB")
                self.logger.info(f"  Solve:  {result.memory.solve_mb:8.1f} MB")
                self.logger.info(f"  Peak:   {result.memory.peak_mb:8.1f} MB")

    def generate_reports(
        self,
        output_dir: str = '.',
        top_k: int = 100,
        plot_layers: Optional[List[str]] = None,
        plot_bin_size: Optional[int] = None,
        anisotropic_bins: Optional[bool] = None,
        bin_aspect_ratio: Optional[int] = None,
        stripe_mode: bool = False,
        max_stripes: int = 50,
        stripe_bin_size: Optional[int] = None,
        show_irdrop: bool = True,
    ):
        """
        Generate visualization and reports.

        Args:
            output_dir: Directory to save outputs
            top_k: Number of worst nodes to report
            plot_layers: List of layer IDs to generate heatmaps for. None = all layers
            plot_bin_size: Bin size for heatmap grid aggregation. None = auto-calculate
            anisotropic_bins: Enable anisotropic binning. None = use solver default
            bin_aspect_ratio: Aspect ratio for anisotropic bins. None = use solver default
            stripe_mode: Enable stripe-based plotting mode (default: False)
            max_stripes: Maximum number of stripes before consolidation (default: 50)
            stripe_bin_size: Bin size for within-stripe aggregation. None = auto-calculate
            show_irdrop: If True (default), show IR-drop in mV. If False, show voltage.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        use_anisotropic = anisotropic_bins if anisotropic_bins is not None else self.anisotropic_bins
        use_aspect_ratio = bin_aspect_ratio if bin_aspect_ratio is not None else self.bin_aspect_ratio

        self.logger.info(f"\n{'='*70}")
        self.logger.info("Generating Reports and Visualizations")
        if show_irdrop:
            self.logger.info("Plotting IR-drop/ground-bounce heatmaps (mV)")
        else:
            self.logger.info("Plotting voltage heatmaps (V)")
        if stripe_mode:
            self.logger.info(f"Using stripe-based plotting mode (max_stripes={max_stripes})")
        elif use_anisotropic:
            self.logger.info(f"Using anisotropic binning with aspect ratio {use_aspect_ratio}:1")
        self.logger.info(f"{'='*70}")

        plotter = PDNPlotter(self.graph, self.net_connectivity, self.logger)

        for net_name, result in self.results.net_results.items():
            self.logger.info(f"\nGenerating reports for net: {net_name}")

            # Top-K worst IR-drop report
            self._generate_topk_report(net_name, result, output_path, top_k)

            # Floating nodes report
            if result.model is not None:
                try:
                    self.logger.info("  Generating floating nodes report...")
                    floating_data = collect_floating_nodes_flat(result.model, net_name=net_name)
                    generate_floating_nodes_report(floating_data, output_path, verbose=self._verbose)
                    if floating_data.total_floating > 0:
                        self.logger.info(f"    Removed {floating_data.total_floating:,} floating nodes "
                                       f"({100.0 * floating_data.total_floating / floating_data.total_nodes:.2f}%)")
                except Exception as e:
                    self.logger.warning(f"  Failed to generate floating nodes report: {e}")

            if stripe_mode:
                plotter.generate_stripe_heatmaps(
                    net_name, output_path, plot_layers,
                    max_stripes, stripe_bin_size,
                    is_current=False, layer_orientations=self.layer_orientations,
                    nominal_voltage=result.nominal_voltage, show_irdrop=show_irdrop)
                plotter.generate_stripe_heatmaps(
                    net_name, output_path, plot_layers,
                    max_stripes, stripe_bin_size,
                    is_current=True, layer_orientations=self.layer_orientations,
                    nominal_voltage=result.nominal_voltage, show_irdrop=show_irdrop)
            else:
                plotter.generate_layer_heatmaps(
                    net_name, output_path, plot_layers, plot_bin_size,
                    use_anisotropic, use_aspect_ratio, self.layer_orientations,
                    nominal_voltage=result.nominal_voltage, show_irdrop=show_irdrop)
                plotter.generate_current_heatmaps(
                    net_name, output_path, plot_layers, plot_bin_size,
                    use_anisotropic, use_aspect_ratio, self.layer_orientations)

    def _generate_topk_report(
        self,
        net_name: str,
        result: PDNSolveResult,
        output_path: Path,
        top_k: int,
    ):
        """Generate top-K worst IR-drop report.

        Delegates to ``reports.topk_irdrop.generate_topk_report()`` after
        building a flat voltages dict from graph node attributes.
        """
        from reports.topk_irdrop import generate_topk_report

        net_nodes = self.net_connectivity.get(net_name, [])

        # Build flat voltages dict from graph node attrs (only nodes with voltage)
        voltages: Dict[str, float] = {}
        for node in net_nodes:
            if node == '0':
                continue
            voltage = self.graph.nodes_dict[node].get('voltage')
            if voltage is not None:
                voltages[node] = voltage

        # Build reverse instance map
        node_to_instance: Dict[str, str] = {}
        for inst, nodes in self.instance_node_map.items():
            for node in nodes:
                node_to_instance[node] = inst

        # Flat solver extra header lines (backend + solve time)
        extra_header_lines = [
            f"Backend: {result.backend}",
            f"Solve Time: {result.timings.total * 1000:.2f} ms",
        ]

        generate_topk_report(
            voltages=voltages,
            nominal_voltage=result.nominal_voltage,
            net_name=net_name,
            pad_nodes=self.vsrc_nodes_global,
            output_dir=str(output_path),
            top_k=top_k,
            node_to_instance=node_to_instance,
            extra_header_lines=extra_header_lines,
        )


# ============================================================================
# Config File Support
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file.

    Args:
        config_path: Path to config file (.yaml, .yml, or .json)

    Returns:
        Dict of configuration parameters

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If config file doesn't exist
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, 'r') as f:
        if path.suffix in ('.yaml', '.yml'):
            if not HAS_YAML:
                raise ValueError("YAML support requires PyYAML. Install with: pip install pyyaml")
            return yaml.safe_load(f)
        elif path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}. Use .yaml, .yml, or .json")


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """Merge config file values with CLI args.

    Config file values are used as defaults. CLI args always take precedence
    when explicitly provided. Since we can't distinguish between default and
    explicitly-provided values in argparse, config file values are always applied.

    Args:
        config: Dict from config file
        args: Parsed CLI arguments

    Returns:
        Updated args namespace with merged values
    """
    # Map config keys to arg names (handle both styles)
    key_mapping = {
        'netlist_dir': 'netlist_dir',
        'netlist-dir': 'netlist_dir',
        'input': 'input',
        'output': 'output',
        'net': 'net',
        'verbose': 'verbose',
        'top_k': 'top_k',
        'top-k': 'top_k',
        'use_cholmod': 'use_cholmod',
        'use-cholmod': 'use_cholmod',
        'cholmod_mode': 'cholmod_mode',
        'cholmod-mode': 'cholmod_mode',
        'cholmod_ordering': 'cholmod_ordering',
        'cholmod-ordering': 'cholmod_ordering',
        'cholmod_use_long': 'cholmod_use_long',
        'cholmod-use-long': 'cholmod_use_long',
        'anisotropic_bins': 'anisotropic_bins',
        'anisotropic-bins': 'anisotropic_bins',
        'bin_aspect_ratio': 'bin_aspect_ratio',
        'bin-aspect-ratio': 'bin_aspect_ratio',
        'layer_orientations': 'layer_orientations',
        'layer-orientations': 'layer_orientations',
        'plot_layers': 'plot_layers',
        'plot-layers': 'plot_layers',
        'plot_bin_size': 'plot_bin_size',
        'plot-bin-size': 'plot_bin_size',
        'stripe_mode': 'stripe_mode',
        'stripe-mode': 'stripe_mode',
        'max_stripes': 'max_stripes',
        'max-stripes': 'max_stripes',
        'stripe_bin_size': 'stripe_bin_size',
        'stripe-bin-size': 'stripe_bin_size',
        'show_irdrop': 'show_irdrop',
        'show-irdrop': 'show_irdrop',
        'profile_memory': 'profile_memory',
        'profile-memory': 'profile_memory',
        'mode': 'mode',
        't_start': 't_start',
        't-start': 't_start',
        't_end': 't_end',
        't-end': 't_end',
        'dt': 'dt',
        'n_points': 'n_points',
        'n-points': 'n_points',
        'method': 'method',
        'smooth': 'smooth',
    }

    # Keys that require numeric type coercion (YAML may parse e.g. '10e-9' as str)
    _float_keys = {'t_start', 't_end', 'dt'}
    _int_keys = {'n_points', 'top_k', 'max_stripes', 'stripe_bin_size',
                 'plot_bin_size', 'bin_aspect_ratio'}

    # Reject unknown config keys.
    # 'solver' is allowed as a nested section (per-role solver configs)
    # but not mapped to an arg — it's consumed separately by the
    # distributed CLI's _apply_yaml_role_configs().
    _allowed_extra = {'solver'}
    unknown = set(config.keys()) - set(key_mapping.keys()) - _allowed_extra
    if unknown:
        raise ValueError(
            f"Unknown config key(s): {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(set(key_mapping.keys()) | _allowed_extra))}"
        )

    for config_key, arg_name in key_mapping.items():
        if config_key in config:
            value = config[config_key]
            if arg_name in _float_keys and isinstance(value, str):
                value = float(value)
            elif arg_name in _int_keys and isinstance(value, str):
                value = int(value)
            setattr(args, arg_name, value)

    return args


# ============================================================================
# CLI
# ============================================================================

def main():
    """Command-line interface for PDN solver."""
    parser = argparse.ArgumentParser(
        description='PDN IR-Drop/Ground-Bounce Solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic solve from parsed graph
  python pdn_solver.py --input pdn.pkl --output ./results

  # Parse and solve with verbose timing
  python pdn_solver.py --netlist-dir ./netlist_data --verbose

  # Use cholmod with specific ordering
  python pdn_solver.py --netlist-dir ./netlist_data --use-cholmod --cholmod-ordering amd

  # Use config file
  python pdn_solver.py --config solver_config.yaml

Config file example (YAML):
  netlist_dir: ./netlist_data
  output: ./results
  verbose: true
  use_cholmod: true
  cholmod_mode: supernodal
  cholmod_ordering: amd
  net: VDD
  top_k: 50
        """
    )

    # Config file
    parser.add_argument('--config', '-c', type=str,
                       help='Config file path (.yaml, .yml, or .json)')

    # Input/output
    parser.add_argument('--input', '-i', type=str,
                       help='Input graph file (.pkl)')
    parser.add_argument('--netlist-dir', type=str,
                       help='Parse netlist from directory (alternative to --input)')
    parser.add_argument('--output', '-o', type=str, default='./results',
                       help='Output directory for results (default: ./results)')
    parser.add_argument('--net', type=str,
                       help='Solve only specific net (default: all nets)')

    # Solver backend
    parser.add_argument('--use-cholmod', action='store_true', default=None,
                       help='Force cholmod backend (requires sksparse)')
    parser.add_argument('--use-splu', action='store_true',
                       help='Force splu backend (scipy)')
    parser.add_argument('--cholmod-mode', type=str, default='auto',
                       choices=['auto', 'simplicial', 'supernodal'],
                       help='Cholmod factorization mode (default: auto)')
    parser.add_argument('--cholmod-ordering', type=str, default='default',
                       choices=['default', 'natural', 'amd', 'metis', 'nesdis', 'colamd', 'best'],
                       help='Cholmod fill-reducing ordering (default: default)')
    parser.add_argument('--cholmod-use-long', action='store_true', default=None,
                       help='Force 64-bit indices in cholmod')

    # Reporting
    parser.add_argument('--top-k', type=int, default=100,
                       help='Number of worst nodes to report (default: 100)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with timing details')
    parser.add_argument('--profile-memory', action='store_true',
                       help='Enable memory profiling (slower)')

    # Plotting
    parser.add_argument('--plot-layers', type=str,
                       help='Generate heatmaps only for specified layers (comma-separated)')
    parser.add_argument('--plot-bin-size', type=int,
                       help='Override base bin size for heatmaps')
    parser.add_argument('--no-anisotropic-bins', dest='anisotropic_bins', action='store_false', default=True,
                       help='Disable orientation-aware anisotropic binning')
    parser.add_argument('--bin-aspect-ratio', type=int, default=50,
                       help='Aspect ratio for anisotropic bins (default: 50)')
    parser.add_argument('--layer-orientations', type=str,
                       help='Manual layer orientation override (e.g., "M1:H,M2:V")')
    parser.add_argument('--stripe-mode', action='store_true',
                       help='Enable stripe-based heatmap plotting mode')
    parser.add_argument('--max-stripes', type=int, default=50,
                       help='Maximum number of stripes before grouping (default: 50)')
    parser.add_argument('--stripe-bin-size', type=int,
                       help='Physical bin size for within-stripe aggregation')
    parser.add_argument('--show-voltage', dest='show_irdrop', action='store_false', default=True,
                       help='Show voltage heatmaps instead of IR-drop')

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        try:
            config = load_config(args.config)
            args = merge_config_with_args(config, args)
            print(f"Loaded config from: {args.config}")
        except Exception as e:
            print(f"ERROR loading config: {e}")
            return 1

    # Handle use_cholmod / use_splu
    use_cholmod = None
    if args.use_cholmod:
        use_cholmod = True
    elif hasattr(args, 'use_splu') and args.use_splu:
        use_cholmod = False

    # Load or parse graph
    if args.input:
        print(f"Loading graph from {args.input}...")

        if args.input.endswith('.pkl'):
            from parser.netlist import load_pdn_pickle
            graph = load_pdn_pickle(args.input)
        else:
            print(f"ERROR: Unsupported input format. Use .pkl")
            return 1

    elif args.netlist_dir:
        print(f"Parsing netlist from {args.netlist_dir}...")
        from parser.netlist import NetlistParser

        parser_obj = NetlistParser(args.netlist_dir,
                             net_filter=args.net,
                             verbose=args.verbose)
        graph = parser_obj.parse()

    else:
        print("ERROR: Either --input or --netlist-dir must be specified")
        parser.print_help()
        return 1

    # Parse layer orientations if provided
    layer_orientations = {}
    if args.layer_orientations:
        for pair in args.layer_orientations.split(','):
            if ':' in pair:
                layer, orient = pair.split(':', 1)
                layer_orientations[layer.strip()] = orient.strip().upper()

    # Create solver
    solver = PDNSolver(
        graph,
        verbose=args.verbose,
        net_filter=args.net,
        use_cholmod=use_cholmod,
        cholmod_mode=args.cholmod_mode,
        cholmod_ordering=args.cholmod_ordering,
        cholmod_use_long=args.cholmod_use_long,
        anisotropic_bins=args.anisotropic_bins,
        bin_aspect_ratio=args.bin_aspect_ratio,
        layer_orientations=layer_orientations,
        profile_memory=args.profile_memory,
    )

    # Solve
    results = solver.solve()

    # Generate reports
    plot_layers = args.plot_layers.split(',') if args.plot_layers else None

    solver.generate_reports(
        output_dir=args.output,
        top_k=args.top_k,
        plot_layers=plot_layers,
        plot_bin_size=args.plot_bin_size,
        stripe_mode=args.stripe_mode,
        max_stripes=args.max_stripes,
        stripe_bin_size=args.stripe_bin_size,
        show_irdrop=args.show_irdrop,
    )

    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"Results saved to: {args.output}")
    print(f"{'='*70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
