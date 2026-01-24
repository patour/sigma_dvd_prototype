"""Unified IR-drop solver supporting both flat and hierarchical solving.

This module provides the UnifiedIRDropSolver class that works with
UnifiedPowerGridModel for both synthetic grids and PDN netlists.

Supports:
- Flat (direct) solving
- Hierarchical solving with top/bottom grid decomposition
- Tiled hierarchical solving with locality-exploiting parallel bottom-grid computation
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy
import scipy.sparse.linalg as spla
from packaging import version as pkg_version

from .solver_results import (
    UnifiedSolveResult,
    UnifiedHierarchicalResult,
    UnifiedCoupledHierarchicalResult,
    TileBounds,
    BottomGridTile,
    TileSolveResult,
    TiledBottomGridResult,
    FlatSolverContext,
    HierarchicalSolverContext,
    CoupledHierarchicalSolverContext,
    TiledHierarchicalSolverContext,
)
from .current_aggregation import CurrentAggregator
from .tiling import TileManager, solve_single_tile
from .unified_model import UnifiedPowerGridModel, LayerID
from .edge_adapter import EdgeInfoExtractor
from .coupled_system import (
    BlockMatrixSystem,
    SchurComplementOperator,
    CoupledSystemOperator,
    BlockDiagonalPreconditioner,
    ILUPreconditioner,
    extract_block_matrices,
    compute_reduced_rhs,
    recover_bottom_voltages,
)

# Logger for solver warnings
logger = logging.getLogger(__name__)

# Scipy version compatibility for iterative solvers
# - scipy < 1.12: uses 'tol' parameter
# - scipy >= 1.12: 'tol' deprecated in favor of 'rtol'
# - scipy >= 1.14: 'tol' removed, only 'rtol' works
_SCIPY_VERSION = pkg_version.parse(scipy.__version__)
_SCIPY_USE_RTOL = _SCIPY_VERSION >= pkg_version.parse("1.12.0")


def _get_tol_kwargs(tol: float, atol: float = 0.0) -> dict:
    """Get tolerance kwargs compatible with current scipy version.

    Args:
        tol: Relative tolerance for convergence
        atol: Absolute tolerance for convergence (default 0.0)

    Returns:
        Dict with appropriate tolerance parameters for scipy version.
    """
    if _SCIPY_USE_RTOL:
        return {"rtol": tol, "atol": atol}
    else:
        # scipy < 1.12 uses 'tol' parameter
        # Note: older scipy had different atol handling ('legacy' default)
        return {"tol": tol}


# Re-export for backward compatibility
__all__ = [
    'UnifiedIRDropSolver',
    'UnifiedSolveResult',
    'UnifiedHierarchicalResult',
    'UnifiedCoupledHierarchicalResult',
    'TileBounds',
    'BottomGridTile',
    'TileSolveResult',
    'TiledBottomGridResult',
]


class UnifiedIRDropSolver:
    """Unified IR-drop solver for power grids.

    Supports:
    - Flat (direct) solving
    - Hierarchical solving with top/bottom grid decomposition
    - Both synthetic and PDN sources via UnifiedPowerGridModel

    Example usage:
        model = create_model_from_synthetic(G, pads, vdd=1.0)
        solver = UnifiedIRDropSolver(model)

        # Flat solve
        result = solver.solve(load_currents)

        # Hierarchical solve
        hier_result = solver.solve_hierarchical(load_currents, partition_layer=2)
    """

    def __init__(self, model: UnifiedPowerGridModel):
        """Initialize solver with a unified model.

        Args:
            model: UnifiedPowerGridModel instance
        """
        self.model = model
        self._aggregator = CurrentAggregator(model)
        self._tile_manager = TileManager(model)

    @property
    def vdd(self) -> float:
        """Get nominal voltage."""
        return self.model.vdd

    def solve(
        self,
        current_injections: Dict[Any, float],
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[FlatSolverContext] = None,
    ) -> UnifiedSolveResult:
        """Solve for voltages using flat (direct) method.

        Args:
            current_injections: Node -> current (positive = sink)
            metadata: Optional metadata to include in result
            context: Optional pre-computed FlatSolverContext for efficiency.
                     If provided, reuses cached LU factorization.

        Returns:
            UnifiedSolveResult with voltages and IR-drop.

        Example:
            # Single solve
            result = solver.solve(currents)

            # Batch solve with cached context
            ctx = solver.prepare_flat()
            results = [solver.solve(stim, context=ctx) for stim in stimuli]
        """
        if context is not None:
            return self.solve_prepared(current_injections, context, metadata)

        voltages = self.model.solve_voltages(current_injections)
        ir_drop = self.model.ir_drop(voltages)

        return UnifiedSolveResult(
            voltages=voltages,
            ir_drop=ir_drop,
            nominal_voltage=self.model.vdd,
            net_name=self.model.net_name,
            metadata=metadata or {},
        )

    def solve_batch(
        self,
        stimuli: List[Dict[Any, float]],
        metadatas: Optional[List[Dict]] = None,
    ) -> List[UnifiedSolveResult]:
        """Solve for multiple stimuli (reuses LU factorization).

        Args:
            stimuli: List of current injection dicts
            metadatas: Optional list of metadata dicts

        Returns:
            List of UnifiedSolveResult, one per stimulus.
        """
        metadatas = metadatas or [{}] * len(stimuli)
        results = []

        for currents, meta in zip(stimuli, metadatas):
            result = self.solve(currents, metadata=meta)
            results.append(result)

        return results

    def prepare_flat(self) -> FlatSolverContext:
        """Prepare context for efficient batch flat solving.

        Builds and caches the reduced system with LU factorization.
        Subsequent calls to solve_prepared() reuse this factorization.

        Returns:
            FlatSolverContext with cached LU factorization.

        Example:
            ctx = solver.prepare_flat()
            results = [solver.solve_prepared(stim, ctx) for stim in stimuli]
        """
        # Ensure reduced system is built (triggers LU factorization)
        _ = self.model.reduced

        return FlatSolverContext(
            reduced_system=self.model.reduced,
            vdd=self.model.vdd,
            net_name=self.model.net_name,
            pad_nodes=set(self.model.pad_nodes),
        )

    def solve_prepared(
        self,
        current_injections: Dict[Any, float],
        context: FlatSolverContext,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UnifiedSolveResult:
        """Solve using pre-computed context for efficiency.

        Uses the cached LU factorization from the context, avoiding
        expensive matrix construction and factorization.

        Args:
            current_injections: Node -> current (positive = sink)
            context: Pre-computed FlatSolverContext from prepare_flat()
            metadata: Optional metadata to include in result

        Returns:
            UnifiedSolveResult with voltages and IR-drop.

        Example:
            ctx = solver.prepare_flat()
            for stim in stimuli:
                result = solver.solve_prepared(stim, ctx)
        """
        rs = context.reduced_system

        if len(rs.unknown_nodes) == 0:
            # All nodes are pads
            voltages = {n: rs.pad_voltage for n in rs.pad_nodes}
        else:
            # Build RHS: I_u (current injections at unknown nodes)
            I_u = np.zeros(len(rs.unknown_nodes), dtype=float)
            for n, cur in current_injections.items():
                if n in rs.index_unknown:
                    # Sink current is positive input, but nodal equation uses negative injection
                    I_u[rs.index_unknown[n]] += -float(cur)

            # Pad voltage contribution: -G_up * V_p
            V_p = np.full(len(rs.pad_nodes), rs.pad_voltage, dtype=float)
            if rs.G_up.shape[1] > 0:
                rhs = I_u - rs.G_up @ V_p
            else:
                rhs = I_u

            # Solve: V_u = lu(rhs)
            V_u = rs.lu(rhs)

            # Build voltage dict
            voltages = {}
            for n in rs.pad_nodes:
                voltages[n] = rs.pad_voltage
            for i, n in enumerate(rs.unknown_nodes):
                voltages[n] = float(V_u[i])

        # Compute IR-drop
        ir_drop = {n: context.vdd - v for n, v in voltages.items()}

        return UnifiedSolveResult(
            voltages=voltages,
            ir_drop=ir_drop,
            nominal_voltage=context.vdd,
            net_name=context.net_name,
            metadata=metadata or {},
        )

    def summarize(
        self, result: "UnifiedSolveResult | UnifiedHierarchicalResult"
    ) -> Dict[str, float]:
        """Compute summary statistics from a solve result.

        Statistics exclude pad nodes to report only free (non-vsrc) node behavior,
        consistent with PDNSolver reporting conventions.

        Args:
            result: UnifiedSolveResult or UnifiedHierarchicalResult

        Returns:
            Dict with nominal_voltage, min_voltage, max_voltage, max_drop, avg_drop.
            For UnifiedHierarchicalResult, also includes num_ports.
        """
        # Exclude pad nodes from statistics (they are fixed at nominal voltage)
        pad_set = set(self.model.pad_nodes)
        free_voltages = [v for n, v in result.voltages.items() if n not in pad_set]
        free_drops = [d for n, d in result.ir_drop.items() if n not in pad_set]

        # Get nominal voltage (UnifiedSolveResult has it directly, hierarchical uses model)
        if hasattr(result, 'nominal_voltage'):
            nominal = result.nominal_voltage
        else:
            nominal = self.model.vdd

        summary = {
            'nominal_voltage': nominal,
            'min_voltage': min(free_voltages) if free_voltages else nominal,
            'max_voltage': max(free_voltages) if free_voltages else nominal,
            'max_drop': max(free_drops) if free_drops else 0.0,
            'avg_drop': sum(free_drops) / len(free_drops) if free_drops else 0.0,
        }

        # Add hierarchical-specific stats
        if isinstance(result, UnifiedHierarchicalResult):
            summary['num_ports'] = len(result.port_nodes)
            summary['partition_layer'] = result.partition_layer

        return summary

    def solve_hierarchical(
        self,
        current_injections: Dict[Any, float],
        partition_layer: LayerID,
        top_k: int = 5,
        weighting: str = "shortest_path",
        rmax: Optional[float] = None,
        verbose: bool = False,
        use_fast_builder: bool = True,
        context: Optional[HierarchicalSolverContext] = None,
    ) -> UnifiedHierarchicalResult:
        """Solve using hierarchical decomposition.

        Decomposes the grid at partition_layer into:
        - Top-grid: layers >= partition_layer (contains pads)
        - Bottom-grid: layers < partition_layer (contains loads)
        - Ports: nodes at partition_layer connecting to bottom-grid

        Steps:
        1. Aggregate bottom-grid currents to ports (using top-k weighting)
        2. Solve top-grid with pad voltages as Dirichlet BC and port injections
        3. Solve bottom-grid with port voltages as Dirichlet BCs

        Args:
            current_injections: Node -> current (positive = sink)
            partition_layer: Layer to partition at
            top_k: Number of nearest ports for current aggregation
            weighting: "effective" (effective resistance) or "shortest_path"
            rmax: Maximum resistance distance for shortest_path weighting.
                  Paths beyond this distance are ignored. None means no limit.
                  Only applies when weighting="shortest_path".
            verbose: If True, print timing information for each step.
            use_fast_builder: If True (default), use vectorized subgrid builder
                  for ~10x speedup. Set to False to use original Python-loop
                  implementation for debugging or validation.
            context: Optional pre-computed HierarchicalSolverContext for efficiency.
                     If provided, reuses cached LU factorizations and path cache.

        Returns:
            UnifiedHierarchicalResult with complete voltages and decomposition info.

        Raises:
            ValueError: If partition layer has no ports, or if load nodes are
                electrically disconnected from ports. The error message will
                suggest alternative partition layers if disconnection is detected.

        Example:
            # Single solve
            result = solver.solve_hierarchical(currents, partition_layer='M2')

            # Batch solve with cached context
            ctx = solver.prepare_hierarchical(partition_layer='M2')
            results = [solver.solve_hierarchical(stim, partition_layer='M2', context=ctx) for stim in stimuli]
        """
        if context is not None:
            return self.solve_hierarchical_prepared(current_injections, context, verbose)

        timings: Dict[str, float] = {}

        # Step 0: Decompose the grid
        t0 = time.perf_counter()
        top_nodes, bottom_nodes, port_nodes, via_edges = self.model._decompose_at_layer(partition_layer)
        timings['decompose'] = time.perf_counter() - t0

        if not port_nodes:
            raise ValueError(
                f"No ports found at partition layer {partition_layer}. "
                "Check that vias connect this layer to the layer below."
            )

        # Step 0.5: Validate load-to-port connectivity before expensive computation
        t0 = time.perf_counter()
        bottom_load_nodes = {
            n for n, c in current_injections.items()
            if n in bottom_nodes and n not in port_nodes and c != 0
        }
        if bottom_load_nodes:
            disconnected_loads = self._find_disconnected_loads(
                bottom_grid_nodes=bottom_nodes,
                port_nodes=port_nodes,
                load_nodes=bottom_load_nodes,
            )
            if disconnected_loads:
                # Find alternative partition layers that might work better
                suggestions = self._suggest_partition_layers(
                    disconnected_loads=disconnected_loads,
                    current_partition=partition_layer,
                )
                raise ValueError(
                    f"{len(disconnected_loads)} load node(s) are electrically disconnected from "
                    f"ports at partition layer {partition_layer}. "
                    f"Example disconnected load: {next(iter(disconnected_loads))}. "
                    f"This typically means the partition layer is too high. "
                    f"{suggestions}"
                )
        timings['connectivity_check'] = time.perf_counter() - t0

        # Step 1: Aggregate bottom-grid currents onto ports
        t0 = time.perf_counter()
        port_currents, aggregation_map = self._aggregator.aggregate_currents_to_ports(
            current_injections=current_injections,
            bottom_grid_nodes=bottom_nodes,
            port_nodes=port_nodes,
            top_k=top_k,
            weighting=weighting,
            rmax=rmax,
        )
        timings['aggregate_currents'] = time.perf_counter() - t0

        # Step 2: Solve top-grid with pads as Dirichlet BC
        # Top-grid includes: top_nodes (layers >= partition_layer)
        # Dirichlet nodes: pads (on top layer)
        pad_set = set(self.model.pad_nodes)
        top_grid_pads = pad_set & top_nodes

        if not top_grid_pads:
            raise ValueError(
                f"No pad nodes found in top-grid (layers >= {partition_layer}). "
                "Pads should be on the top-most layer."
            )

        # Build and solve top-grid system
        t0 = time.perf_counter()
        if use_fast_builder:
            top_system = self.model._build_subgrid_system_fast(
                subgrid_nodes=top_nodes,
                dirichlet_nodes=top_grid_pads,
                dirichlet_voltage=self.model.vdd,
            )
        else:
            top_system = self.model._build_subgrid_system(
                subgrid_nodes=top_nodes,
                dirichlet_nodes=top_grid_pads,
                dirichlet_voltage=self.model.vdd,
            )

        if top_system is None:
            raise ValueError("Failed to build top-grid system")
        timings['build_top_system'] = time.perf_counter() - t0

        # Current injections for top-grid: loads in top-grid + aggregated port currents
        # Note: Any loads that happen to be in top-grid are handled directly
        top_grid_currents = {n: c for n, c in current_injections.items() if n in top_nodes}
        # Add aggregated port currents
        for port, curr in port_currents.items():
            top_grid_currents[port] = top_grid_currents.get(port, 0.0) + curr

        t0 = time.perf_counter()
        top_voltages = self.model._solve_subgrid(
            reduced_system=top_system,
            current_injections=top_grid_currents,
            dirichlet_voltages=None,  # Use vdd for pads
        )
        timings['solve_top'] = time.perf_counter() - t0

        # Extract port voltages for bottom-grid BC
        port_voltages = {p: top_voltages[p] for p in port_nodes}

        # Step 3: Solve bottom-grid with port voltages as Dirichlet BC
        # Bottom-grid includes: bottom_nodes (layers < partition_layer)
        # We need to include the ports as Dirichlet nodes but they're in top-grid
        # Solution: Include ports in the bottom subgrid for the solve
        bottom_subgrid = bottom_nodes | port_nodes

        t0 = time.perf_counter()
        if use_fast_builder:
            bottom_system = self.model._build_subgrid_system_fast(
                subgrid_nodes=bottom_subgrid,
                dirichlet_nodes=port_nodes,
                dirichlet_voltage=self.model.vdd,  # Will be overridden by dirichlet_voltages
            )
        else:
            bottom_system = self.model._build_subgrid_system(
                subgrid_nodes=bottom_subgrid,
                dirichlet_nodes=port_nodes,
                dirichlet_voltage=self.model.vdd,  # Will be overridden by dirichlet_voltages
            )

        if bottom_system is None:
            raise ValueError("Failed to build bottom-grid system")
        timings['build_bottom_system'] = time.perf_counter() - t0

        # Current injections for bottom-grid: original currents in bottom-grid
        bottom_grid_currents = {
            n: c for n, c in current_injections.items()
            if n in bottom_nodes
        }

        t0 = time.perf_counter()
        bottom_voltages = self.model._solve_subgrid(
            reduced_system=bottom_system,
            current_injections=bottom_grid_currents,
            dirichlet_voltages=port_voltages,
        )
        timings['solve_bottom'] = time.perf_counter() - t0

        # Merge voltages: top-grid + bottom-grid (excluding duplicate ports)
        t0 = time.perf_counter()
        all_voltages = {}
        all_voltages.update(top_voltages)
        # Add bottom-grid nodes (excluding ports which are already in top_voltages)
        for n, v in bottom_voltages.items():
            if n not in port_nodes:
                all_voltages[n] = v

        # Compute IR-drop
        ir_drop = self.model.ir_drop(all_voltages)
        timings['merge_results'] = time.perf_counter() - t0

        # Print timing summary if verbose
        if verbose:
            total_time = sum(timings.values())
            print(f"\n=== Hierarchical Solve Timing ===")
            print(f"  Top nodes: {len(top_nodes):,}, Bottom nodes: {len(bottom_nodes):,}, Ports: {len(port_nodes):,}")
            print(f"  Load nodes in bottom-grid: {len(bottom_load_nodes):,}")
            print(f"  ---")
            for step, t in timings.items():
                pct = t / total_time * 100
                print(f"  {step:25s}: {t*1000:8.1f} ms  ({pct:5.1f}%)")
            print(f"  {'TOTAL':25s}: {total_time*1000:8.1f} ms")
            print(f"=================================\n")

        return UnifiedHierarchicalResult(
            voltages=all_voltages,
            ir_drop=ir_drop,
            partition_layer=partition_layer,
            top_grid_voltages=top_voltages,
            bottom_grid_voltages=bottom_voltages,
            port_nodes=port_nodes,
            port_voltages=port_voltages,
            port_currents=port_currents,
            aggregation_map=aggregation_map,
        )

    def prepare_hierarchical(
        self,
        partition_layer: LayerID,
        top_k: int = 5,
        weighting: str = "shortest_path",
        rmax: Optional[float] = None,
        use_fast_builder: bool = True,
    ) -> HierarchicalSolverContext:
        """Prepare context for efficient batch hierarchical solving.

        Pre-computes grid decomposition, builds and factors top/bottom systems,
        and caches shortest-path distances for current aggregation.

        Args:
            partition_layer: Layer to partition at
            top_k: Number of nearest ports for current aggregation
            weighting: "effective" or "shortest_path" for current aggregation
            rmax: Maximum resistance distance for shortest_path weighting
            use_fast_builder: If True (default), use vectorized subgrid builder

        Returns:
            HierarchicalSolverContext with cached artifacts.

        Raises:
            ValueError: If partition layer has no ports or pads not in top-grid.

        Example:
            ctx = solver.prepare_hierarchical(partition_layer='M2', top_k=5)
            results = [solver.solve_hierarchical_prepared(stim, ctx) for stim in stimuli]
        """
        # Step 1: Decompose grid
        top_nodes, bottom_nodes, port_nodes, via_edges = self.model._decompose_at_layer(partition_layer)

        if not port_nodes:
            raise ValueError(
                f"No ports found at partition layer {partition_layer}. "
                "Check that vias connect this layer to the layer below."
            )

        # Step 2: Validate pads in top-grid
        pad_set = set(self.model.pad_nodes)
        top_grid_pads = pad_set & top_nodes

        if not top_grid_pads:
            raise ValueError(
                f"No pad nodes found in top-grid (layers >= {partition_layer}). "
                "Pads should be on the top-most layer."
            )

        # Step 3: Build and factor top-grid system
        if use_fast_builder:
            top_system = self.model._build_subgrid_system_fast(
                subgrid_nodes=top_nodes,
                dirichlet_nodes=top_grid_pads,
                dirichlet_voltage=self.model.vdd,
            )
        else:
            top_system = self.model._build_subgrid_system(
                subgrid_nodes=top_nodes,
                dirichlet_nodes=top_grid_pads,
                dirichlet_voltage=self.model.vdd,
            )

        if top_system is None:
            raise ValueError("Failed to build top-grid system")

        # Step 4: Build and factor bottom-grid system
        bottom_subgrid = bottom_nodes | port_nodes

        if use_fast_builder:
            bottom_system = self.model._build_subgrid_system_fast(
                subgrid_nodes=bottom_subgrid,
                dirichlet_nodes=port_nodes,
                dirichlet_voltage=self.model.vdd,
            )
        else:
            bottom_system = self.model._build_subgrid_system(
                subgrid_nodes=bottom_subgrid,
                dirichlet_nodes=port_nodes,
                dirichlet_voltage=self.model.vdd,
            )

        if bottom_system is None:
            raise ValueError("Failed to build bottom-grid system")

        # Step 5: Pre-compute shortest-path cache for current aggregation
        shortest_path_cache: Dict[Any, List[Tuple[Any, float]]] = {}
        if weighting == "shortest_path":
            shortest_path_cache = self._aggregator.multi_source_port_distances(
                subgrid_nodes=bottom_subgrid,
                port_nodes=list(port_nodes),
                top_k=top_k,
                rmax=rmax,
            )

        return HierarchicalSolverContext(
            partition_layer=partition_layer,
            top_nodes=top_nodes,
            bottom_nodes=bottom_nodes,
            port_nodes=port_nodes,
            via_edges=via_edges,
            top_system=top_system,
            bottom_system=bottom_system,
            top_k=top_k,
            weighting=weighting,
            rmax=rmax,
            shortest_path_cache=shortest_path_cache,
            pad_nodes=pad_set,
            top_grid_pads=top_grid_pads,
            vdd=self.model.vdd,
        )

    def solve_hierarchical_prepared(
        self,
        current_injections: Dict[Any, float],
        context: HierarchicalSolverContext,
        verbose: bool = False,
    ) -> UnifiedHierarchicalResult:
        """Solve using pre-computed hierarchical context for efficiency.

        Uses cached LU factorizations and shortest-path distances,
        avoiding expensive matrix construction and factorization.

        Args:
            current_injections: Node -> current (positive = sink)
            context: Pre-computed HierarchicalSolverContext from prepare_hierarchical()
            verbose: If True, print timing information

        Returns:
            UnifiedHierarchicalResult with voltages and decomposition info.

        Raises:
            ValueError: If load nodes are disconnected from ports.

        Example:
            ctx = solver.prepare_hierarchical(partition_layer='M2')
            for stim in stimuli:
                result = solver.solve_hierarchical_prepared(stim, ctx)
        """
        timings: Dict[str, float] = {}

        # Validate load-to-port connectivity
        t0 = time.perf_counter()
        bottom_load_nodes = {
            n for n, c in current_injections.items()
            if n in context.bottom_nodes and n not in context.port_nodes and c != 0
        }
        if bottom_load_nodes:
            disconnected_loads = self._find_disconnected_loads(
                bottom_grid_nodes=context.bottom_nodes,
                port_nodes=context.port_nodes,
                load_nodes=bottom_load_nodes,
            )
            if disconnected_loads:
                suggestions = self._suggest_partition_layers(
                    disconnected_loads=disconnected_loads,
                    current_partition=context.partition_layer,
                )
                raise ValueError(
                    f"{len(disconnected_loads)} load node(s) are electrically disconnected from "
                    f"ports at partition layer {context.partition_layer}. "
                    f"{suggestions}"
                )
        timings['connectivity_check'] = time.perf_counter() - t0

        # Step 1: Aggregate bottom-grid currents to ports using cached distances
        t0 = time.perf_counter()
        port_currents, aggregation_map = self._aggregate_with_cache(
            current_injections=current_injections,
            context=context,
        )
        timings['aggregate_currents'] = time.perf_counter() - t0

        # Step 2: Solve top-grid using cached LU
        t0 = time.perf_counter()
        top_grid_currents = {n: c for n, c in current_injections.items() if n in context.top_nodes}
        for port, curr in port_currents.items():
            top_grid_currents[port] = top_grid_currents.get(port, 0.0) + curr

        top_voltages = self.model._solve_subgrid(
            reduced_system=context.top_system,
            current_injections=top_grid_currents,
            dirichlet_voltages=None,
        )
        timings['solve_top'] = time.perf_counter() - t0

        # Extract port voltages
        port_voltages = {p: top_voltages[p] for p in context.port_nodes}

        # Step 3: Solve bottom-grid using cached LU with port voltages as Dirichlet BC
        t0 = time.perf_counter()
        bottom_grid_currents = {
            n: c for n, c in current_injections.items()
            if n in context.bottom_nodes
        }

        bottom_voltages = self.model._solve_subgrid(
            reduced_system=context.bottom_system,
            current_injections=bottom_grid_currents,
            dirichlet_voltages=port_voltages,
        )
        timings['solve_bottom'] = time.perf_counter() - t0

        # Merge voltages
        t0 = time.perf_counter()
        all_voltages = {}
        all_voltages.update(top_voltages)
        for n, v in bottom_voltages.items():
            if n not in context.port_nodes:
                all_voltages[n] = v

        ir_drop = self.model.ir_drop(all_voltages)
        timings['merge_results'] = time.perf_counter() - t0

        if verbose:
            total_time = sum(timings.values())
            print(f"\n=== Hierarchical Solve (Prepared) Timing ===")
            print(f"  Top nodes: {len(context.top_nodes):,}, Bottom nodes: {len(context.bottom_nodes):,}, Ports: {len(context.port_nodes):,}")
            print(f"  Load nodes in bottom-grid: {len(bottom_load_nodes):,}")
            print(f"  ---")
            for step, t in timings.items():
                pct = t / total_time * 100
                print(f"  {step:25s}: {t*1000:8.1f} ms  ({pct:5.1f}%)")
            print(f"  {'TOTAL':25s}: {total_time*1000:8.1f} ms")
            print(f"=============================================\n")

        return UnifiedHierarchicalResult(
            voltages=all_voltages,
            ir_drop=ir_drop,
            partition_layer=context.partition_layer,
            top_grid_voltages=top_voltages,
            bottom_grid_voltages=bottom_voltages,
            port_nodes=context.port_nodes,
            port_voltages=port_voltages,
            port_currents=port_currents,
            aggregation_map=aggregation_map,
        )

    def _aggregate_with_cache(
        self,
        current_injections: Dict[Any, float],
        context: HierarchicalSolverContext,
    ) -> Tuple[Dict[Any, float], Dict[Any, List[Tuple[Any, float, float]]]]:
        """Aggregate bottom-grid currents to ports using cached distances.

        Args:
            current_injections: Node -> current
            context: HierarchicalSolverContext with cached shortest_path_cache

        Returns:
            (port_currents, aggregation_map)
        """
        port_currents = {p: 0.0 for p in context.port_nodes}
        aggregation_map: Dict[Any, List[Tuple[Any, float, float]]] = {}

        # Filter currents to bottom-grid nodes (excluding ports)
        bottom_currents = {
            n: c for n, c in current_injections.items()
            if n in context.bottom_nodes and n not in context.port_nodes and c != 0
        }

        if not bottom_currents:
            return port_currents, aggregation_map

        port_list = list(context.port_nodes)

        for load_node, load_current in bottom_currents.items():
            if context.weighting == "effective":
                # Effective resistance weighting (not cached, computed on-the-fly)
                subgraph_nodes = context.bottom_nodes | context.port_nodes
                resistances = self._aggregator.compute_effective_resistance_in_subgrid(
                    subgrid_nodes=subgraph_nodes,
                    source_node=load_node,
                    target_nodes=port_list,
                    dirichlet_nodes=context.port_nodes,
                )
            else:
                # Use cached shortest-path distances
                port_distances = context.shortest_path_cache.get(load_node, [])
                resistances = {p: d for p, d in port_distances if d is not None}

            valid_resistances = {
                p: r for p, r in resistances.items()
                if r is not None and r < float('inf') and r > 0
            }

            if not valid_resistances:
                raise ValueError(
                    f"No valid resistance paths found from load node {load_node} to any port. "
                    f"This indicates the load is electrically isolated from the port layer."
                )

            sorted_ports = sorted(valid_resistances.items(), key=lambda x: x[1])
            selected = sorted_ports[:context.top_k] if context.top_k < len(sorted_ports) else sorted_ports

            inv_R = [(p, 1.0 / R) for p, R in selected]
            total_inv_R = sum(w for _, w in inv_R)

            contributions = []
            for port, inv_r in inv_R:
                weight = inv_r / total_inv_R
                contrib = load_current * weight
                port_currents[port] += contrib
                contributions.append((port, weight, contrib))

            aggregation_map[load_node] = contributions

        return port_currents, aggregation_map

    def _find_disconnected_loads(
        self,
        bottom_grid_nodes: Set[Any],
        port_nodes: Set[Any],
        load_nodes: Set[Any],
    ) -> Set[Any]:
        """Find load nodes that are electrically disconnected from ports.

        Uses connected components analysis on R-type edges to identify loads
        that have no resistive path to any port node.

        Args:
            bottom_grid_nodes: Set of nodes in bottom-grid
            port_nodes: Set of port nodes at partition boundary
            load_nodes: Set of load nodes to check

        Returns:
            Set of load nodes that are disconnected from all ports.
        """
        # Build R-type subgraph for bottom-grid + ports
        subgraph_nodes = bottom_grid_nodes | port_nodes

        # Use Union-Find for efficient component detection
        parent: Dict[Any, Any] = {}

        def find(x: Any) -> Any:
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: Any, y: Any) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Process edges using EdgeInfoExtractor to handle both synthetic and PDN formats
        edge_extractor = self.model._edge_extractor
        for u, v, d in self.model.graph.edges(subgraph_nodes, data=True):
            if u not in subgraph_nodes or v not in subgraph_nodes:
                continue
            edge_info = edge_extractor.get_info(d)
            if not edge_info.is_resistive:
                continue
            R = edge_info.resistance if edge_info.resistance else 0.0
            if R > 0 and R < float('inf'):
                union(u, v)

        # Find components containing ports
        port_components = {find(p) for p in port_nodes if p in parent}

        # Find loads not in any port-containing component
        disconnected = set()
        for load in load_nodes:
            if load not in parent:
                # Node has no R-type edges at all
                disconnected.add(load)
            elif find(load) not in port_components:
                disconnected.add(load)

        return disconnected

    def _suggest_partition_layers(
        self,
        disconnected_loads: Set[Any],
        current_partition: LayerID,
    ) -> str:
        """Suggest alternative partition layers that might include disconnected loads.

        Args:
            disconnected_loads: Set of load nodes disconnected from current partition
            current_partition: The partition layer that failed

        Returns:
            String with suggestions for alternative partition layers.
        """
        # Get layers present in disconnected loads
        load_layers: Dict[LayerID, int] = {}
        for load in list(disconnected_loads)[:1000]:  # Sample for efficiency
            info = self.model.get_node_info(load)
            if info.layer is not None:
                load_layers[info.layer] = load_layers.get(info.layer, 0) + 1

        if not load_layers:
            return "Could not determine layers of disconnected loads."

        # Find the highest layer with disconnected loads
        all_layers = sorted(self.model.get_all_layers())
        pad_set = set(self.model.pad_nodes)

        # Try lower partition layers and check:
        # 1. Has ports
        # 2. Has pads in top-grid
        suggestions = []

        for layer in reversed(all_layers):
            if layer >= current_partition:
                continue

            try:
                top_nodes, bottom_nodes, ports, _ = self.model._decompose_at_layer(layer)
            except ValueError:
                # Skip invalid partition layers (e.g., bottom layer)
                continue
            if not ports:
                continue

            # Check if pads are in top-grid
            pads_in_top = pad_set & top_nodes
            if not pads_in_top:
                continue

            suggestions.append(f"layer {layer} ({len(ports)} ports, {len(pads_in_top)} pads in top-grid)")
            if len(suggestions) >= 3:
                break

        if suggestions:
            return f"Try partitioning at a lower layer: {', '.join(suggestions)}."
        return "Try using a lower partition layer closer to the load layers."

    def solve_hierarchical_tiled(
        self,
        current_injections: Dict[Any, float],
        partition_layer: LayerID,
        N_x: int,
        N_y: int,
        halo_percent: float,
        min_ports_per_tile: Optional[int] = None,
        top_k: int = 5,
        weighting: str = "shortest_path",
        rmax: Optional[float] = None,
        n_workers: Optional[int] = None,
        parallel_backend: str = "thread",
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
        validate_against_flat: bool = False,
        verbose: bool = False,
        override_port_voltages: Optional[Dict[Any, float]] = None,
        context: Optional[TiledHierarchicalSolverContext] = None,
    ) -> TiledBottomGridResult:
        """Solve using tiled hierarchical decomposition with locality exploitation.

        Decomposes the grid into top-grid and bottom-grid at partition_layer,
        then tiles the bottom-grid for parallel localized IR-drop computation.
        Each tile uses halo regions with port nodes as Dirichlet BCs.

        Only supports PDN graphs (not synthetic grids with NodeID).

        Args:
            current_injections: Node -> current (positive = sink)
            partition_layer: Layer to partition at
            N_x: Number of tiles in x direction
            N_y: Number of tiles in y direction
            halo_percent: Halo size as percentage of tile dimensions (0.0 to 1.0)
                          Larger halo improves accuracy but increases computation.
            min_ports_per_tile: Minimum port nodes required per tile.
                                Default: ceil(total_ports / (N_x * N_y))
            top_k: Number of nearest ports for current aggregation
            weighting: "effective" or "shortest_path" for current aggregation
            rmax: Maximum resistance distance for shortest_path weighting
            n_workers: Number of parallel workers (default: CPU count)
            parallel_backend: "thread" (default, safe for most contexts) or
                              "process" (true parallelism, but requires pickling)
            progress_callback: Optional callback(completed, total, tile_id)
                               called after each tile completes
            validate_against_flat: If True, run full bottom-grid solve and
                                   report accuracy statistics
            verbose: If True, print timing information
            override_port_voltages: If provided, skip top-grid solve and use
                                    these port voltages as Dirichlet BCs.
                                    Useful for validation against flat solver.
            context: Optional pre-computed TiledHierarchicalSolverContext for efficiency.
                     If provided, reuses cached structures.

        Returns:
            TiledBottomGridResult with voltages, tiles, and optional validation stats

        Raises:
            ValueError: If graph is synthetic, partition layer is invalid,
                        or tile constraints cannot be satisfied

        Example:
            # Single solve
            result = solver.solve_hierarchical_tiled(currents, partition_layer='M2', N_x=4, N_y=4)

            # Batch solve with cached context
            ctx = solver.prepare_hierarchical_tiled(partition_layer='M2', N_x=4, N_y=4)
            results = [solver.solve_hierarchical_tiled(stim, partition_layer='M2', N_x=4, N_y=4, context=ctx) for stim in stimuli]
        """
        if context is not None:
            return self.solve_hierarchical_tiled_prepared(
                current_injections, context, progress_callback, validate_against_flat, verbose
            )

        import os
        timings: Dict[str, float] = {}

        # ================================================================
        # Step 1: Standard hierarchical decomposition and top-grid solve
        # ================================================================
        t0 = time.perf_counter()
        top_nodes, bottom_nodes, port_nodes, via_edges = self.model._decompose_at_layer(partition_layer)
        timings['decompose'] = time.perf_counter() - t0

        if not port_nodes:
            raise ValueError(
                f"No ports found at partition layer {partition_layer}. "
                "Check that vias connect this layer to the layer below."
            )

        # Validate load-to-port connectivity
        t0 = time.perf_counter()
        bottom_load_nodes = {
            n for n, c in current_injections.items()
            if n in bottom_nodes and n not in port_nodes and c != 0
        }
        if bottom_load_nodes:
            disconnected_loads = self._find_disconnected_loads(
                bottom_grid_nodes=bottom_nodes,
                port_nodes=port_nodes,
                load_nodes=bottom_load_nodes,
            )
            if disconnected_loads:
                suggestions = self._suggest_partition_layers(
                    disconnected_loads=disconnected_loads,
                    current_partition=partition_layer,
                )
                raise ValueError(
                    f"{len(disconnected_loads)} load node(s) are electrically disconnected from "
                    f"ports at partition layer {partition_layer}. "
                    f"{suggestions}"
                )
        timings['connectivity_check'] = time.perf_counter() - t0

        # ================================================================
        # Use override port voltages OR solve top-grid
        # ================================================================
        if override_port_voltages is not None:
            # Skip top-grid solve; use provided port voltages directly
            port_voltages = {p: override_port_voltages.get(p, self.model.vdd) for p in port_nodes}
            port_currents: Dict[Any, float] = {}  # Not needed for bottom-grid-only validation
            aggregation_map: Dict[Any, List[Tuple[Any, float, float]]] = {}
            timings['aggregate_currents'] = 0.0
            timings['build_top_system'] = 0.0
            timings['solve_top'] = 0.0
            top_voltages: Dict[Any, float] = {}
            if verbose:
                print("Using override port voltages (skipping top-grid solve)")
        else:
            # Aggregate currents to ports
            t0 = time.perf_counter()
            port_currents, aggregation_map = self._aggregator.aggregate_currents_to_ports(
                current_injections=current_injections,
                bottom_grid_nodes=bottom_nodes,
                port_nodes=port_nodes,
                top_k=top_k,
                weighting=weighting,
                rmax=rmax,
            )
            timings['aggregate_currents'] = time.perf_counter() - t0

            # Build and solve top-grid
            t0 = time.perf_counter()
            pad_set = set(self.model.pad_nodes)
            top_grid_pads = pad_set & top_nodes

            if not top_grid_pads:
                raise ValueError(
                    f"No pad nodes found in top-grid (layers >= {partition_layer})."
                )

            top_system = self.model._build_subgrid_system_fast(
                subgrid_nodes=top_nodes,
                dirichlet_nodes=top_grid_pads,
                dirichlet_voltage=self.model.vdd,
            )
            if top_system is None:
                raise ValueError("Failed to build top-grid system")
            timings['build_top_system'] = time.perf_counter() - t0

            t0 = time.perf_counter()
            top_grid_currents = {n: c for n, c in current_injections.items() if n in top_nodes}
            for port, curr in port_currents.items():
                top_grid_currents[port] = top_grid_currents.get(port, 0.0) + curr

            top_voltages = self.model._solve_subgrid(
                reduced_system=top_system,
                current_injections=top_grid_currents,
                dirichlet_voltages=None,
            )
            timings['solve_top'] = time.perf_counter() - t0

            port_voltages = {p: top_voltages[p] for p in port_nodes}

        # ================================================================
        # Step 2: Extract coordinates and generate tiles
        # ================================================================
        t0 = time.perf_counter()
        bottom_coords, port_coords, grid_bounds = self._tile_manager.extract_bottom_grid_coordinates(
            bottom_nodes, port_nodes
        )
        timings['extract_coords'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        initial_bounds = self._tile_manager.generate_uniform_tile_bounds(N_x, N_y, grid_bounds)

        # Determine minimum ports per tile
        total_ports_with_coords = len(port_coords)
        if min_ports_per_tile is None:
            min_ports_per_tile = max(1, math.ceil(total_ports_with_coords / (N_x * N_y)))

        # Assign nodes and adjust constraints
        adjusted_bounds, tile_nodes, tile_ports, tile_loads = self._tile_manager.assign_nodes_to_tiles(
            bottom_coords=bottom_coords,
            port_coords=port_coords,
            tile_bounds=initial_bounds,
            port_nodes=port_nodes,
            load_nodes=bottom_load_nodes,
            min_ports_per_tile=min_ports_per_tile,
            N_x=N_x,
            N_y=N_y,
        )
        timings['generate_tiles'] = time.perf_counter() - t0

        # ================================================================
        # Step 3: Expand tiles with halos
        # ================================================================
        t0 = time.perf_counter()
        tiles: List[BottomGridTile] = []
        for bounds in adjusted_bounds:
            tid = bounds.tile_id
            tile = self._tile_manager.expand_tile_with_halo(
                tile=bounds,
                tile_core_nodes=tile_nodes.get(tid, set()),
                bottom_coords=bottom_coords,
                grid_bounds=grid_bounds,
                halo_percent=halo_percent,
                port_nodes=port_nodes,
                port_coords=port_coords,
                load_nodes=bottom_load_nodes,
            )
            tiles.append(tile)
        timings['expand_halos'] = time.perf_counter() - t0

        # ================================================================
        # Step 3b: Validate and fix tile connectivity
        # Ensure all core nodes are connected to ports through tile edges
        # ================================================================
        t0 = time.perf_counter()
        tiles = self._tile_manager.validate_and_fix_tile_connectivity(tiles, bottom_nodes, port_nodes)
        timings['fix_connectivity'] = time.perf_counter() - t0

        # Validate disjoint core regions
        all_core_nodes: Set[Any] = set()
        for tile in tiles:
            overlap = all_core_nodes & tile.core_nodes
            if overlap:
                raise ValueError(
                    f"Tile cores are not disjoint. Overlap detected: {list(overlap)[:5]}..."
                )
            all_core_nodes.update(tile.core_nodes)

        # ================================================================
        # Step 4: Build tile solve arguments
        # ================================================================
        t0 = time.perf_counter()
        solve_args = self._tile_manager.build_tile_solve_args(tiles, port_voltages, current_injections)
        timings['build_args'] = time.perf_counter() - t0

        # ================================================================
        # Step 5: Parallel tile solves
        # ================================================================
        t0 = time.perf_counter()
        tile_results: Dict[int, Dict[Any, float]] = {}
        per_tile_times: Dict[int, float] = {}

        if n_workers is None:
            n_workers = os.cpu_count() or 1

        n_tiles = len(tiles)
        disconnected_halo_counts: Dict[int, int] = {}

        if n_tiles == 0:
            # No tiles to solve
            pass
        elif n_workers == 1 or n_tiles == 1:
            # Sequential execution
            for args in solve_args:
                tile_id, voltages, solve_time, disconnected_halo = solve_single_tile(*args)
                tile_results[tile_id] = voltages
                per_tile_times[tile_id] = solve_time
                if disconnected_halo:
                    disconnected_halo_counts[tile_id] = len(disconnected_halo)
                if progress_callback:
                    progress_callback(len(tile_results), n_tiles, tile_id)
        else:
            # Parallel execution
            ExecutorClass = (
                ProcessPoolExecutor if parallel_backend == "process"
                else ThreadPoolExecutor
            )

            with ExecutorClass(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(solve_single_tile, *args): args[0]
                    for args in solve_args
                }

                for future in as_completed(futures):
                    try:
                        tile_id, voltages, solve_time, disconnected_halo = future.result()
                        tile_results[tile_id] = voltages
                        per_tile_times[tile_id] = solve_time
                        if disconnected_halo:
                            disconnected_halo_counts[tile_id] = len(disconnected_halo)
                    except Exception as e:
                        tile_id = futures[future]
                        logger.error(f"Tile {tile_id} solve failed: {e}")
                        raise  # Re-raise to propagate disconnected core errors

                    if progress_callback:
                        progress_callback(len(tile_results), n_tiles, tile_id)

        timings['solve_tiles'] = time.perf_counter() - t0

        # Log disconnected halo statistics
        total_disconnected_halo = sum(disconnected_halo_counts.values())
        if total_disconnected_halo > 0 and verbose:
            print(f"  Dropped {total_disconnected_halo} disconnected halo nodes across {len(disconnected_halo_counts)} tiles")

        # ================================================================
        # Step 6: Merge results
        # ================================================================
        t0 = time.perf_counter()
        bottom_voltages, halo_warnings = self._tile_manager.merge_tiled_voltages(
            tiles=tiles,
            tile_results=tile_results,
            bottom_grid_nodes=bottom_nodes,
        )

        # Merge with top-grid voltages (if we solved top-grid)
        all_voltages: Dict[Any, float] = {}
        if override_port_voltages is None:
            # Full hierarchical solve - include top-grid voltages
            all_voltages.update(top_voltages)
        else:
            # Bottom-grid only validation - include port voltages as provided
            all_voltages.update(override_port_voltages)
        all_voltages.update(bottom_voltages)

        # Compute IR-drop
        ir_drop = self.model.ir_drop(all_voltages)
        timings['merge_results'] = time.perf_counter() - t0

        # ================================================================
        # Step 7: Optional accuracy validation
        # ================================================================
        validation_stats = None
        if validate_against_flat:
            t0 = time.perf_counter()
            validation_stats = self._tile_manager.validate_tiled_accuracy(
                tiled_voltages=bottom_voltages,
                bottom_grid_nodes=bottom_nodes,
                port_nodes=port_nodes,
                port_voltages=port_voltages,
                current_injections=current_injections,
            )
            timings['validate'] = time.perf_counter() - t0

        # ================================================================
        # Verbose output
        # ================================================================
        if verbose:
            total_time = sum(timings.values())
            print(f"\n=== Tiled Hierarchical Solve Timing ===")
            print(f"  Top nodes: {len(top_nodes):,}, Bottom nodes: {len(bottom_nodes):,}")
            print(f"  Ports: {len(port_nodes):,}, Tiles: {len(tiles)}")
            print(f"  Grid: {N_x}x{N_y}, Halo: {halo_percent*100:.0f}%")
            print(f"  Workers: {n_workers}, Backend: {parallel_backend}")
            print(f"  ---")
            for step, t in timings.items():
                pct = t / total_time * 100
                print(f"  {step:25s}: {t*1000:8.1f} ms  ({pct:5.1f}%)")
            print(f"  {'TOTAL':25s}: {total_time*1000:8.1f} ms")

            if validation_stats:
                print(f"  --- Validation ---")
                print(f"  Max diff: {validation_stats['max_diff']*1000:.4f} mV")
                print(f"  Mean diff: {validation_stats['mean_diff']*1000:.4f} mV")
                print(f"  RMSE: {validation_stats['rmse']*1000:.4f} mV")

            if halo_warnings:
                print(f"  --- Warnings ---")
                for w in halo_warnings[:5]:
                    print(f"  {w}")

            print(f"=========================================\n")

        # Determine top_grid_voltages for result
        if override_port_voltages is None:
            result_top_voltages = top_voltages
        else:
            # In validation mode, we don't have top-grid voltages
            result_top_voltages = {}

        return TiledBottomGridResult(
            voltages=all_voltages,
            ir_drop=ir_drop,
            partition_layer=partition_layer,
            top_grid_voltages=result_top_voltages,
            bottom_grid_voltages=bottom_voltages,
            port_nodes=port_nodes,
            port_voltages=port_voltages,
            port_currents=port_currents,
            aggregation_map=aggregation_map,
            tiles=tiles,
            per_tile_solve_times=per_tile_times,
            halo_clip_warnings=halo_warnings,
            validation_stats=validation_stats,
            tiling_params={
                'N_x': N_x,
                'N_y': N_y,
                'halo_percent': halo_percent,
                'min_ports_per_tile': min_ports_per_tile,
                'n_workers': n_workers,
                'parallel_backend': parallel_backend,
            },
        )

    def prepare_hierarchical_tiled(
        self,
        partition_layer: LayerID,
        N_x: int,
        N_y: int,
        halo_percent: float,
        min_ports_per_tile: Optional[int] = None,
        top_k: int = 5,
        weighting: str = "shortest_path",
        rmax: Optional[float] = None,
        n_workers: Optional[int] = None,
        parallel_backend: str = "thread",
    ) -> TiledHierarchicalSolverContext:
        """Prepare context for efficient batch tiled hierarchical solving.

        Pre-computes grid decomposition, top-grid system, tile structure,
        and shortest-path cache for current aggregation.

        Note: Individual tile solves still perform their own LU factorization
        per solve, as tile port voltages change with each stimulus.

        Args:
            partition_layer: Layer to partition at
            N_x: Number of tiles in x direction
            N_y: Number of tiles in y direction
            halo_percent: Halo size as percentage of tile dimensions
            min_ports_per_tile: Minimum port nodes per tile
            top_k: Number of nearest ports for current aggregation
            weighting: "effective" or "shortest_path" for current aggregation
            rmax: Maximum resistance distance for shortest_path weighting
            n_workers: Number of parallel workers (default: CPU count)
            parallel_backend: "thread" or "process"

        Returns:
            TiledHierarchicalSolverContext with cached artifacts.

        Example:
            ctx = solver.prepare_hierarchical_tiled(partition_layer='M2', N_x=4, N_y=4)
            results = [solver.solve_hierarchical_tiled_prepared(stim, ctx) for stim in stimuli]
        """
        import os

        # Step 1: Decompose grid
        top_nodes, bottom_nodes, port_nodes, via_edges = self.model._decompose_at_layer(partition_layer)

        if not port_nodes:
            raise ValueError(
                f"No ports found at partition layer {partition_layer}. "
                "Check that vias connect this layer to the layer below."
            )

        # Validate pads in top-grid
        pad_set = set(self.model.pad_nodes)
        top_grid_pads = pad_set & top_nodes

        if not top_grid_pads:
            raise ValueError(
                f"No pad nodes found in top-grid (layers >= {partition_layer})."
            )

        # Step 2: Build and factor top-grid system
        top_system = self.model._build_subgrid_system_fast(
            subgrid_nodes=top_nodes,
            dirichlet_nodes=top_grid_pads,
            dirichlet_voltage=self.model.vdd,
        )
        if top_system is None:
            raise ValueError("Failed to build top-grid system")

        # Step 3: Extract coordinates and generate tiles
        bottom_coords, port_coords, grid_bounds = self._tile_manager.extract_bottom_grid_coordinates(
            bottom_nodes, port_nodes
        )

        initial_bounds = self._tile_manager.generate_uniform_tile_bounds(N_x, N_y, grid_bounds)

        total_ports_with_coords = len(port_coords)
        if min_ports_per_tile is None:
            min_ports_per_tile = max(1, math.ceil(total_ports_with_coords / (N_x * N_y)))

        # Note: We need dummy load_nodes for tile assignment - will be updated per solve
        # For prepare, we use empty set since actual loads vary per stimulus
        adjusted_bounds, tile_nodes, tile_ports, tile_loads = self._tile_manager.assign_nodes_to_tiles(
            bottom_coords=bottom_coords,
            port_coords=port_coords,
            tile_bounds=initial_bounds,
            port_nodes=port_nodes,
            load_nodes=set(),  # Empty for prepare, actual loads handled per solve
            min_ports_per_tile=min_ports_per_tile,
            N_x=N_x,
            N_y=N_y,
        )

        # Step 4: Expand tiles with halos (using empty load_nodes for now)
        tiles: List[BottomGridTile] = []
        for bounds in adjusted_bounds:
            tid = bounds.tile_id
            tile = self._tile_manager.expand_tile_with_halo(
                tile=bounds,
                tile_core_nodes=tile_nodes.get(tid, set()),
                bottom_coords=bottom_coords,
                grid_bounds=grid_bounds,
                halo_percent=halo_percent,
                port_nodes=port_nodes,
                port_coords=port_coords,
                load_nodes=set(),  # Empty for prepare
            )
            tiles.append(tile)

        # Step 5: Validate and fix tile connectivity
        tiles = self._tile_manager.validate_and_fix_tile_connectivity(tiles, bottom_nodes, port_nodes)

        # Step 6: Pre-compute shortest-path cache
        bottom_subgrid = bottom_nodes | port_nodes
        shortest_path_cache: Dict[Any, List[Tuple[Any, float]]] = {}
        if weighting == "shortest_path":
            shortest_path_cache = self._aggregator.multi_source_port_distances(
                subgrid_nodes=bottom_subgrid,
                port_nodes=list(port_nodes),
                top_k=top_k,
                rmax=rmax,
            )

        if n_workers is None:
            n_workers = os.cpu_count() or 1

        return TiledHierarchicalSolverContext(
            partition_layer=partition_layer,
            top_nodes=top_nodes,
            bottom_nodes=bottom_nodes,
            port_nodes=port_nodes,
            top_system=top_system,
            top_grid_pads=top_grid_pads,
            tiles=tiles,
            bottom_coords=bottom_coords,
            port_coords=port_coords,
            grid_bounds=grid_bounds,
            top_k=top_k,
            weighting=weighting,
            rmax=rmax,
            shortest_path_cache=shortest_path_cache,
            N_x=N_x,
            N_y=N_y,
            halo_percent=halo_percent,
            min_ports_per_tile=min_ports_per_tile,
            n_workers=n_workers,
            parallel_backend=parallel_backend,
            vdd=self.model.vdd,
            pad_nodes=pad_set,
        )

    def solve_hierarchical_tiled_prepared(
        self,
        current_injections: Dict[Any, float],
        context: TiledHierarchicalSolverContext,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
        validate_against_flat: bool = False,
        verbose: bool = False,
    ) -> TiledBottomGridResult:
        """Solve using pre-computed tiled hierarchical context for efficiency.

        Uses cached top-grid system and tile structure.
        Per-solve operations: current aggregation, top-grid solve, tile solves.

        Args:
            current_injections: Node -> current (positive = sink)
            context: Pre-computed TiledHierarchicalSolverContext
            progress_callback: Optional callback(completed, total, tile_id)
            validate_against_flat: If True, run validation against flat solve
            verbose: If True, print timing information

        Returns:
            TiledBottomGridResult with voltages and tile information.

        Example:
            ctx = solver.prepare_hierarchical_tiled(partition_layer='M2', N_x=4, N_y=4)
            for stim in stimuli:
                result = solver.solve_hierarchical_tiled_prepared(stim, ctx)
        """
        timings: Dict[str, float] = {}

        # Validate load-to-port connectivity
        t0 = time.perf_counter()
        bottom_load_nodes = {
            n for n, c in current_injections.items()
            if n in context.bottom_nodes and n not in context.port_nodes and c != 0
        }
        if bottom_load_nodes:
            disconnected_loads = self._find_disconnected_loads(
                bottom_grid_nodes=context.bottom_nodes,
                port_nodes=context.port_nodes,
                load_nodes=bottom_load_nodes,
            )
            if disconnected_loads:
                suggestions = self._suggest_partition_layers(
                    disconnected_loads=disconnected_loads,
                    current_partition=context.partition_layer,
                )
                raise ValueError(
                    f"{len(disconnected_loads)} load node(s) are electrically disconnected from "
                    f"ports at partition layer {context.partition_layer}. "
                    f"{suggestions}"
                )
        timings['connectivity_check'] = time.perf_counter() - t0

        # Aggregate currents using cached distances
        t0 = time.perf_counter()
        # Create a temporary hierarchical context for aggregation
        temp_hier_ctx = HierarchicalSolverContext(
            partition_layer=context.partition_layer,
            top_nodes=context.top_nodes,
            bottom_nodes=context.bottom_nodes,
            port_nodes=context.port_nodes,
            via_edges=set(),
            top_system=context.top_system,
            bottom_system=context.top_system,  # Not used for aggregation
            top_k=context.top_k,
            weighting=context.weighting,
            rmax=context.rmax,
            shortest_path_cache=context.shortest_path_cache,
            pad_nodes=context.pad_nodes,
            top_grid_pads=context.top_grid_pads,
            vdd=context.vdd,
        )
        port_currents, aggregation_map = self._aggregate_with_cache(
            current_injections=current_injections,
            context=temp_hier_ctx,
        )
        timings['aggregate_currents'] = time.perf_counter() - t0

        # Solve top-grid using cached LU
        t0 = time.perf_counter()
        top_grid_currents = {n: c for n, c in current_injections.items() if n in context.top_nodes}
        for port, curr in port_currents.items():
            top_grid_currents[port] = top_grid_currents.get(port, 0.0) + curr

        top_voltages = self.model._solve_subgrid(
            reduced_system=context.top_system,
            current_injections=top_grid_currents,
            dirichlet_voltages=None,
        )
        timings['solve_top'] = time.perf_counter() - t0

        port_voltages = {p: top_voltages[p] for p in context.port_nodes}

        # Build tile solve arguments
        t0 = time.perf_counter()
        solve_args = self._tile_manager.build_tile_solve_args(
            context.tiles, port_voltages, current_injections
        )
        timings['build_args'] = time.perf_counter() - t0

        # Parallel tile solves
        t0 = time.perf_counter()
        tile_results: Dict[int, Dict[Any, float]] = {}
        per_tile_times: Dict[int, float] = {}
        n_tiles = len(context.tiles)
        disconnected_halo_counts: Dict[int, int] = {}

        if n_tiles == 0:
            pass
        elif context.n_workers == 1 or n_tiles == 1:
            for args in solve_args:
                tile_id, voltages, solve_time, disconnected_halo = solve_single_tile(*args)
                tile_results[tile_id] = voltages
                per_tile_times[tile_id] = solve_time
                if disconnected_halo:
                    disconnected_halo_counts[tile_id] = len(disconnected_halo)
                if progress_callback:
                    progress_callback(len(tile_results), n_tiles, tile_id)
        else:
            ExecutorClass = (
                ProcessPoolExecutor if context.parallel_backend == "process"
                else ThreadPoolExecutor
            )

            with ExecutorClass(max_workers=context.n_workers) as executor:
                futures = {
                    executor.submit(solve_single_tile, *args): args[0]
                    for args in solve_args
                }

                for future in as_completed(futures):
                    try:
                        tile_id, voltages, solve_time, disconnected_halo = future.result()
                        tile_results[tile_id] = voltages
                        per_tile_times[tile_id] = solve_time
                        if disconnected_halo:
                            disconnected_halo_counts[tile_id] = len(disconnected_halo)
                    except Exception as e:
                        tile_id = futures[future]
                        logger.error(f"Tile {tile_id} solve failed: {e}")
                        raise

                    if progress_callback:
                        progress_callback(len(tile_results), n_tiles, tile_id)

        timings['solve_tiles'] = time.perf_counter() - t0

        if sum(disconnected_halo_counts.values()) > 0 and verbose:
            print(f"  Dropped {sum(disconnected_halo_counts.values())} disconnected halo nodes")

        # Merge results
        t0 = time.perf_counter()
        bottom_voltages, halo_warnings = self._tile_manager.merge_tiled_voltages(
            tiles=context.tiles,
            tile_results=tile_results,
            bottom_grid_nodes=context.bottom_nodes,
        )

        all_voltages: Dict[Any, float] = {}
        all_voltages.update(top_voltages)
        all_voltages.update(bottom_voltages)

        ir_drop = self.model.ir_drop(all_voltages)
        timings['merge_results'] = time.perf_counter() - t0

        # Optional validation
        validation_stats = None
        if validate_against_flat:
            t0 = time.perf_counter()
            validation_stats = self._tile_manager.validate_tiled_accuracy(
                tiled_voltages=bottom_voltages,
                bottom_grid_nodes=context.bottom_nodes,
                port_nodes=context.port_nodes,
                port_voltages=port_voltages,
                current_injections=current_injections,
            )
            timings['validate'] = time.perf_counter() - t0

        if verbose:
            total_time = sum(timings.values())
            print(f"\n=== Tiled Hierarchical Solve (Prepared) ===")
            print(f"  Tiles: {len(context.tiles)}, Workers: {context.n_workers}")
            print(f"  ---")
            for step, t in timings.items():
                pct = t / total_time * 100
                print(f"  {step:25s}: {t*1000:8.1f} ms  ({pct:5.1f}%)")
            print(f"  {'TOTAL':25s}: {total_time*1000:8.1f} ms")
            if validation_stats:
                print(f"  --- Validation ---")
                print(f"  Max diff: {validation_stats['max_diff']*1000:.4f} mV")
            print(f"============================================\n")

        return TiledBottomGridResult(
            voltages=all_voltages,
            ir_drop=ir_drop,
            partition_layer=context.partition_layer,
            top_grid_voltages=top_voltages,
            bottom_grid_voltages=bottom_voltages,
            port_nodes=context.port_nodes,
            port_voltages=port_voltages,
            port_currents=port_currents,
            aggregation_map=aggregation_map,
            tiles=context.tiles,
            per_tile_solve_times=per_tile_times,
            halo_clip_warnings=halo_warnings,
            validation_stats=validation_stats,
            tiling_params={
                'N_x': context.N_x,
                'N_y': context.N_y,
                'halo_percent': context.halo_percent,
                'min_ports_per_tile': context.min_ports_per_tile,
                'n_workers': context.n_workers,
                'parallel_backend': context.parallel_backend,
            },
        )

    def solve_hierarchical_coupled(
        self,
        current_injections: Dict[Any, float],
        partition_layer: LayerID,
        solver: str = 'gmres',
        tol: float = 1e-8,
        maxiter: int = 500,
        preconditioner: str = 'block_diagonal',
        verbose: bool = False,
        context: Optional[CoupledHierarchicalSolverContext] = None,
    ) -> UnifiedCoupledHierarchicalResult:
        """Solve using coupled hierarchical decomposition (exact up to tolerance).

        This method solves the coupled top-grid + bottom-grid system exactly
        (up to iterative tolerance) using a matrix-free Schur complement approach.
        Unlike solve_hierarchical(), which approximates port currents via weighted
        distribution, this method preserves the exact coupling between grids.

        Mathematical formulation:
        - Decompose grid at partition_layer into top-grid and bottom-grid
        - Eliminate bottom-grid interior via Schur complement onto ports
        - Solve coupled system: (G^T + S^B) at ports, coupled with top interior
        - Recover bottom-grid voltages from port voltages

        Args:
            current_injections: Node -> current (positive = sink)
            partition_layer: Layer to partition at
            solver: Iterative solver to use:
                - 'gmres': GMRES (robust, works for non-symmetric systems)
                - 'bicgstab': BiCGSTAB (often faster than GMRES)
                - 'cg': Conjugate Gradient (optimal for SPD systems, recommended)
            tol: Convergence tolerance for iterative solver
            maxiter: Maximum iterations for iterative solver
            preconditioner: Preconditioner type:
                - 'none': No preconditioning
                - 'block_diagonal': Block diagonal approximation (default)
                - 'ilu': ILU-based preconditioner
                - 'amg': Algebraic Multigrid (best for large problems, requires pyamg)
            verbose: If True, print timing and convergence information
            context: Optional pre-computed CoupledHierarchicalSolverContext for efficiency.
                     If provided, reuses cached matrices and operators.

        Returns:
            UnifiedCoupledHierarchicalResult with exact (up to tol) voltages.

        Raises:
            ValueError: If partition layer has no ports, or if pads not in top-grid
            RuntimeError: If iterative solver does not converge within maxiter

        Example:
            # Single solve
            result = solver.solve_hierarchical_coupled(currents, partition_layer='M2')

            # Batch solve with cached context
            ctx = solver.prepare_hierarchical_coupled(partition_layer='M2')
            results = [solver.solve_hierarchical_coupled(stim, partition_layer='M2', context=ctx) for stim in stimuli]
        """
        if context is not None:
            return self.solve_hierarchical_coupled_prepared(current_injections, context, verbose)

        timings: Dict[str, float] = {}

        # ================================================================
        # Step 1: Decompose grid at partition layer
        # ================================================================
        t0 = time.perf_counter()
        top_nodes, bottom_nodes, port_nodes, via_edges = self.model._decompose_at_layer(partition_layer)
        timings['decompose'] = time.perf_counter() - t0

        if not port_nodes:
            raise ValueError(
                f"No ports found at partition layer {partition_layer}. "
                "Check that vias connect this layer to the layer below."
            )

        # Validate pads are in top-grid
        pad_set = set(self.model.pad_nodes)
        top_grid_pads = pad_set & top_nodes

        if not top_grid_pads:
            raise ValueError(
                f"No pad nodes found in top-grid (layers >= {partition_layer}). "
                "Pads should be on the top-most layer."
            )

        # ================================================================
        # Step 2: Extract block matrices for top-grid and bottom-grid
        # ================================================================
        t0 = time.perf_counter()

        # Top-grid: includes top_nodes, pads are Dirichlet
        # Port nodes are the interface to bottom-grid
        top_blocks, rhs_dirichlet_top = extract_block_matrices(
            model=self.model,
            grid_nodes=top_nodes,
            dirichlet_nodes=top_grid_pads,
            port_nodes=port_nodes,
            dirichlet_voltage=self.model.vdd,
        )
        timings['extract_top_blocks'] = time.perf_counter() - t0

        t0 = time.perf_counter()

        # Bottom-grid: includes bottom_nodes + port_nodes
        # Port nodes are the interface; no Dirichlet nodes in bottom-grid itself
        # IMPORTANT: exclude_port_to_port=True to avoid double-counting lateral
        # port connections that are already in top-grid G_pp
        bottom_subgrid = bottom_nodes | port_nodes
        bottom_blocks, rhs_dirichlet_bottom = extract_block_matrices(
            model=self.model,
            grid_nodes=bottom_subgrid,
            dirichlet_nodes=set(),  # No Dirichlet in bottom-grid
            port_nodes=port_nodes,
            dirichlet_voltage=self.model.vdd,
            exclude_port_to_port=True,
        )
        timings['extract_bottom_blocks'] = time.perf_counter() - t0

        # ================================================================
        # Step 3: Factor bottom-grid interior (for Schur complement)
        # ================================================================
        t0 = time.perf_counter()
        bottom_blocks.factor_interior()
        timings['factor_bottom_interior'] = time.perf_counter() - t0

        # ================================================================
        # Step 4: Build Schur complement operator for bottom-grid
        # ================================================================
        t0 = time.perf_counter()
        schur_B = SchurComplementOperator(
            G_pp=bottom_blocks.G_pp,
            G_pi=bottom_blocks.G_pi,
            G_ip=bottom_blocks.G_ip,
            lu_ii=bottom_blocks.lu_ii,
        )
        timings['build_schur_operator'] = time.perf_counter() - t0

        # ================================================================
        # Step 5: Build coupled system operator
        # ================================================================
        t0 = time.perf_counter()
        coupled_op = CoupledSystemOperator(top_blocks, schur_B)
        timings['build_coupled_operator'] = time.perf_counter() - t0

        # ================================================================
        # Step 6: Build coupled RHS
        # ================================================================
        t0 = time.perf_counter()

        # Current injections for bottom-grid (includes ports and bottom interior)
        # Port currents are included here and will contribute to r^B
        bottom_currents = {n: c for n, c in current_injections.items() if n in bottom_subgrid}

        # Reduced RHS from bottom-grid: r^B = i_p - G^B_pi * inv(G^B_ii) * i_i
        # This includes port currents since ports are in bottom_subgrid
        r_B = compute_reduced_rhs(bottom_blocks, bottom_currents, rhs_dirichlet_bottom)

        # Build full RHS for coupled system
        # Port equations: r^B + rhs_dirichlet_top[ports] (port currents already in r^B)
        # Top interior equations: i_t + rhs_dirichlet_top[interior]
        n_ports = top_blocks.n_ports
        n_top_interior = top_blocks.n_interior

        # Current vector for top-grid INTERIOR nodes only (negated for nodal equation)
        # Note: Port currents are NOT added here - they're already in r^B
        i_t = np.zeros(n_top_interior, dtype=np.float64)

        for node, current in current_injections.items():
            if node in top_blocks.interior_to_idx:
                i_t[top_blocks.interior_to_idx[node]] -= current

        # Combine RHS
        # Port equations: r^B + Dirichlet contribution from pads coupling to ports
        # Top interior equations: i_t + Dirichlet contribution from pads coupling to top interior
        rhs_p = r_B + rhs_dirichlet_top[:n_ports]
        rhs_t = i_t + rhs_dirichlet_top[n_ports:]
        rhs = np.concatenate([rhs_p, rhs_t])

        timings['build_rhs'] = time.perf_counter() - t0

        # ================================================================
        # Step 7: Build preconditioner
        # ================================================================
        t0 = time.perf_counter()
        M = self._build_coupled_preconditioner(
            preconditioner_type=preconditioner,
            top_blocks=top_blocks,
            bottom_blocks=bottom_blocks,
        )
        timings['build_preconditioner'] = time.perf_counter() - t0

        # ================================================================
        # Step 8: Solve coupled system iteratively
        # ================================================================
        t0 = time.perf_counter()

        # Track iteration count and residuals via callback
        iteration_count = [0]
        true_residual_history: List[float] = []  # True residual ||b - Ax|| at each iteration
        initial_residual_norm = np.linalg.norm(rhs)

        if verbose:
            # Use iterate-based callback to compute true residual ||b - Ax||
            def callback(x):
                iteration_count[0] += 1
                true_res = np.linalg.norm(rhs - coupled_op @ x)
                true_residual_history.append(true_res)

            gmres_callback_type = 'x'
        else:
            # Lightweight callback - just count iterations
            def callback(residual):
                iteration_count[0] += 1
                if isinstance(residual, np.ndarray):
                    true_residual_history.append(np.linalg.norm(residual))
                else:
                    true_residual_history.append(residual)

            gmres_callback_type = 'pr_norm'

        # Select solver
        # Use _get_tol_kwargs for scipy version compatibility (tol vs rtol)
        tol_kwargs = _get_tol_kwargs(tol)

        if solver.lower() == 'gmres':
            solution, info = spla.gmres(
                coupled_op, rhs, **tol_kwargs, maxiter=maxiter, M=M, callback=callback,
                callback_type=gmres_callback_type
            )
        elif solver.lower() == 'bicgstab':
            solution, info = spla.bicgstab(
                coupled_op, rhs, **tol_kwargs, maxiter=maxiter, M=M, callback=callback
            )
        elif solver.lower() == 'cg':
            # CG is optimal for SPD systems (conductance matrices are SPD)
            # Note: CG callback receives residual norm, not solution vector
            solution, info = spla.cg(
                coupled_op, rhs, **tol_kwargs, maxiter=maxiter, M=M, callback=callback
            )
        else:
            raise ValueError(f"Unknown solver: {solver}. Use 'gmres', 'bicgstab', or 'cg'.")

        timings['iterative_solve'] = time.perf_counter() - t0

        # Check convergence and compute final true residual
        converged = (info == 0)
        iterations = iteration_count[0]
        final_residual = np.linalg.norm(rhs - coupled_op @ solution)
        final_relative_residual = final_residual / initial_residual_norm if initial_residual_norm > 0 else final_residual

        # Build residual history string for error reporting
        def _format_residual_history(history: List[float], max_entries: int = 20) -> str:
            """Format residual history for display."""
            if not history:
                return "  (no iterations recorded)"
            lines = []
            step = max(1, len(history) // max_entries)
            for i in range(0, len(history), step):
                rel_res = history[i] / initial_residual_norm if initial_residual_norm > 0 else history[i]
                lines.append(f"  iter {i+1:4d}: ||r|| = {history[i]:.6e}, ||r||/||b|| = {rel_res:.6e}")
            # Always include the last entry
            if (len(history) - 1) % step != 0:
                rel_res = history[-1] / initial_residual_norm if initial_residual_norm > 0 else history[-1]
                lines.append(f"  iter {len(history):4d}: ||r|| = {history[-1]:.6e}, ||r||/||b|| = {rel_res:.6e}")
            return "\n".join(lines)

        if not converged:
            residual_info = _format_residual_history(true_residual_history)
            raise RuntimeError(
                f"Coupled iterative solver did not converge after {maxiter} iterations.\n"
                f"Final true residual ||r||: {final_residual:.2e}\n"
                f"Final relative residual ||r||/||b||: {final_relative_residual:.2e}\n"
                f"Tolerance (rtol): {tol:.2e}\n"
                f"Initial RHS norm ||b||: {initial_residual_norm:.2e}\n"
                f"Residual history:\n{residual_info}\n"
                f"Try increasing maxiter, loosening tol, or using a different preconditioner."
            )

        # ================================================================
        # Step 9: Extract port and top-interior voltages
        # ================================================================
        t0 = time.perf_counter()

        v_p = solution[:n_ports]
        v_t = solution[n_ports:]

        # Map port voltages
        port_voltages: Dict[Any, float] = {}
        for i, node in enumerate(top_blocks.port_nodes):
            port_voltages[node] = float(v_p[i])

        # Map top-interior voltages
        top_grid_voltages: Dict[Any, float] = {}
        for node in top_grid_pads:
            top_grid_voltages[node] = self.model.vdd
        for node, v in port_voltages.items():
            top_grid_voltages[node] = v
        for i, node in enumerate(top_blocks.interior_nodes):
            top_grid_voltages[node] = float(v_t[i])

        timings['extract_top_voltages'] = time.perf_counter() - t0

        # ================================================================
        # Step 10: Recover bottom-grid voltages
        # ================================================================
        t0 = time.perf_counter()

        bottom_interior_voltages = recover_bottom_voltages(
            bottom_blocks=bottom_blocks,
            port_voltages=v_p,
            current_injections=bottom_currents,
            rhs_dirichlet_bottom=rhs_dirichlet_bottom,
        )

        # Combine with port voltages for full bottom-grid
        bottom_grid_voltages: Dict[Any, float] = {}
        bottom_grid_voltages.update(port_voltages)
        bottom_grid_voltages.update(bottom_interior_voltages)

        timings['recover_bottom_voltages'] = time.perf_counter() - t0

        # ================================================================
        # Step 11: Merge results
        # ================================================================
        t0 = time.perf_counter()

        all_voltages: Dict[Any, float] = {}
        all_voltages.update(top_grid_voltages)
        # Add bottom-grid nodes (excluding ports already in top)
        for n, v in bottom_interior_voltages.items():
            all_voltages[n] = v

        ir_drop = self.model.ir_drop(all_voltages)

        timings['merge_results'] = time.perf_counter() - t0

        # ================================================================
        # Verbose output
        # ================================================================
        if verbose:
            total_time = sum(timings.values())
            print(f"\n=== Coupled Hierarchical Solve ===")
            print(f"  Top nodes: {len(top_nodes):,}, Bottom nodes: {len(bottom_nodes):,}")
            print(f"  Ports: {len(port_nodes):,}, Top interior: {n_top_interior:,}")
            print(f"  Bottom interior: {bottom_blocks.n_interior:,}")
            print(f"  Solver: {solver}, Preconditioner: {preconditioner}")
            print(f"  ---")
            print(f"  Convergence:")
            print(f"    Iterations: {iterations}")
            print(f"    Initial ||b||: {initial_residual_norm:.6e}")
            print(f"    Final ||r||: {final_residual:.6e}")
            print(f"    Final ||r||/||b||: {final_relative_residual:.6e}")
            print(f"    Tolerance (rtol): {tol:.2e}")
            print(f"  ---")
            print(f"  Residual history (true residual ||b - Ax|| at each iteration):")
            print(_format_residual_history(true_residual_history))
            print(f"  ---")
            print(f"  Timing breakdown:")
            for step, t in timings.items():
                pct = t / total_time * 100
                print(f"    {step:25s}: {t*1000:8.1f} ms  ({pct:5.1f}%)")
            print(f"    {'TOTAL':25s}: {total_time*1000:8.1f} ms")
            print(f"==================================\n")

        return UnifiedCoupledHierarchicalResult(
            voltages=all_voltages,
            ir_drop=ir_drop,
            partition_layer=partition_layer,
            top_grid_voltages=top_grid_voltages,
            bottom_grid_voltages=bottom_grid_voltages,
            port_nodes=port_nodes,
            port_voltages=port_voltages,
            iterations=iterations,
            final_residual=final_residual,
            converged=converged,
            preconditioner_type=preconditioner,
            timings=timings,
        )

    def prepare_hierarchical_coupled(
        self,
        partition_layer: LayerID,
        solver: str = 'gmres',
        tol: float = 1e-8,
        maxiter: int = 500,
        preconditioner: str = 'block_diagonal',
    ) -> CoupledHierarchicalSolverContext:
        """Prepare context for efficient batch coupled hierarchical solving.

        Pre-computes grid decomposition, block matrices, LU factorizations,
        Schur complement operator, coupled system operator, and preconditioner.

        Args:
            partition_layer: Layer to partition at
            solver: Iterative solver to use ('gmres' or 'bicgstab')
            tol: Convergence tolerance for iterative solver
            maxiter: Maximum iterations for iterative solver
            preconditioner: Preconditioner type ('none', 'block_diagonal', 'ilu')

        Returns:
            CoupledHierarchicalSolverContext with cached artifacts.

        Raises:
            ValueError: If partition layer has no ports or pads not in top-grid.

        Example:
            ctx = solver.prepare_hierarchical_coupled(partition_layer='M2', tol=1e-8)
            results = [solver.solve_hierarchical_coupled_prepared(stim, ctx) for stim in stimuli]
        """
        # Step 1: Decompose grid
        top_nodes, bottom_nodes, port_nodes, via_edges = self.model._decompose_at_layer(partition_layer)

        if not port_nodes:
            raise ValueError(
                f"No ports found at partition layer {partition_layer}. "
                "Check that vias connect this layer to the layer below."
            )

        # Validate pads in top-grid
        pad_set = set(self.model.pad_nodes)
        top_grid_pads = pad_set & top_nodes

        if not top_grid_pads:
            raise ValueError(
                f"No pad nodes found in top-grid (layers >= {partition_layer}). "
                "Pads should be on the top-most layer."
            )

        # Step 2: Extract block matrices
        top_blocks, rhs_dirichlet_top = extract_block_matrices(
            model=self.model,
            grid_nodes=top_nodes,
            dirichlet_nodes=top_grid_pads,
            port_nodes=port_nodes,
            dirichlet_voltage=self.model.vdd,
        )

        bottom_subgrid = bottom_nodes | port_nodes
        bottom_blocks, rhs_dirichlet_bottom = extract_block_matrices(
            model=self.model,
            grid_nodes=bottom_subgrid,
            dirichlet_nodes=set(),
            port_nodes=port_nodes,
            dirichlet_voltage=self.model.vdd,
            exclude_port_to_port=True,
        )

        # Step 3: Factor bottom-grid interior
        bottom_blocks.factor_interior()

        # Step 4: Build Schur complement operator
        schur_B = SchurComplementOperator(
            G_pp=bottom_blocks.G_pp,
            G_pi=bottom_blocks.G_pi,
            G_ip=bottom_blocks.G_ip,
            lu_ii=bottom_blocks.lu_ii,
        )

        # Step 5: Build coupled system operator
        coupled_op = CoupledSystemOperator(top_blocks, schur_B)

        # Step 6: Build preconditioner
        M = self._build_coupled_preconditioner(
            preconditioner_type=preconditioner,
            top_blocks=top_blocks,
            bottom_blocks=bottom_blocks,
        )

        return CoupledHierarchicalSolverContext(
            partition_layer=partition_layer,
            top_nodes=top_nodes,
            bottom_nodes=bottom_nodes,
            port_nodes=port_nodes,
            bottom_subgrid=bottom_subgrid,
            top_blocks=top_blocks,
            bottom_blocks=bottom_blocks,
            rhs_dirichlet_top=rhs_dirichlet_top,
            rhs_dirichlet_bottom=rhs_dirichlet_bottom,
            schur_B=schur_B,
            coupled_op=coupled_op,
            preconditioner=M,
            preconditioner_type=preconditioner,
            solver=solver,
            tol=tol,
            maxiter=maxiter,
            vdd=self.model.vdd,
            top_grid_pads=top_grid_pads,
            n_ports=top_blocks.n_ports,
            n_top_interior=top_blocks.n_interior,
        )

    def solve_hierarchical_coupled_prepared(
        self,
        current_injections: Dict[Any, float],
        context: CoupledHierarchicalSolverContext,
        verbose: bool = False,
    ) -> UnifiedCoupledHierarchicalResult:
        """Solve using pre-computed coupled hierarchical context for efficiency.

        Uses cached block matrices, LU factorizations, and operators.
        Only builds RHS and runs iterative solver.

        Args:
            current_injections: Node -> current (positive = sink)
            context: Pre-computed CoupledHierarchicalSolverContext
            verbose: If True, print timing and convergence information

        Returns:
            UnifiedCoupledHierarchicalResult with exact (up to tol) voltages.

        Raises:
            RuntimeError: If iterative solver does not converge within maxiter.

        Example:
            ctx = solver.prepare_hierarchical_coupled(partition_layer='M2')
            for stim in stimuli:
                result = solver.solve_hierarchical_coupled_prepared(stim, ctx)
        """
        timings: Dict[str, float] = {}

        # Build RHS from current injections
        t0 = time.perf_counter()

        bottom_currents = {n: c for n, c in current_injections.items() if n in context.bottom_subgrid}
        r_B = compute_reduced_rhs(context.bottom_blocks, bottom_currents, context.rhs_dirichlet_bottom)

        i_t = np.zeros(context.n_top_interior, dtype=np.float64)
        for node, current in current_injections.items():
            if node in context.top_blocks.interior_to_idx:
                i_t[context.top_blocks.interior_to_idx[node]] -= current

        rhs_p = r_B + context.rhs_dirichlet_top[:context.n_ports]
        rhs_t = i_t + context.rhs_dirichlet_top[context.n_ports:]
        rhs = np.concatenate([rhs_p, rhs_t])

        timings['build_rhs'] = time.perf_counter() - t0

        # Iterative solve
        t0 = time.perf_counter()

        iteration_count = [0]
        true_residual_history: List[float] = []
        initial_residual_norm = np.linalg.norm(rhs)

        if verbose:
            def callback(x):
                iteration_count[0] += 1
                true_res = np.linalg.norm(rhs - context.coupled_op @ x)
                true_residual_history.append(true_res)
            gmres_callback_type = 'x'
        else:
            def callback(residual):
                iteration_count[0] += 1
                if isinstance(residual, np.ndarray):
                    true_residual_history.append(np.linalg.norm(residual))
                else:
                    true_residual_history.append(residual)
            gmres_callback_type = 'pr_norm'

        # Use _get_tol_kwargs for scipy version compatibility (tol vs rtol)
        tol_kwargs = _get_tol_kwargs(context.tol)

        if context.solver.lower() == 'gmres':
            solution, info = spla.gmres(
                context.coupled_op, rhs, **tol_kwargs,
                maxiter=context.maxiter, M=context.preconditioner,
                callback=callback, callback_type=gmres_callback_type
            )
        elif context.solver.lower() == 'bicgstab':
            solution, info = spla.bicgstab(
                context.coupled_op, rhs, **tol_kwargs,
                maxiter=context.maxiter, M=context.preconditioner,
                callback=callback
            )
        elif context.solver.lower() == 'cg':
            # CG is optimal for SPD systems (conductance matrices are SPD)
            solution, info = spla.cg(
                context.coupled_op, rhs, **tol_kwargs,
                maxiter=context.maxiter, M=context.preconditioner,
                callback=callback
            )
        else:
            raise ValueError(f"Unknown solver: {context.solver}. Use 'gmres', 'bicgstab', or 'cg'.")

        timings['iterative_solve'] = time.perf_counter() - t0

        converged = (info == 0)
        iterations = iteration_count[0]
        final_residual = np.linalg.norm(rhs - context.coupled_op @ solution)
        final_relative_residual = final_residual / initial_residual_norm if initial_residual_norm > 0 else final_residual

        if not converged:
            def _format_residual_history(history: List[float], max_entries: int = 20) -> str:
                if not history:
                    return "  (no iterations recorded)"
                lines = []
                step = max(1, len(history) // max_entries)
                for i in range(0, len(history), step):
                    rel_res = history[i] / initial_residual_norm if initial_residual_norm > 0 else history[i]
                    lines.append(f"  iter {i+1:4d}: ||r|| = {history[i]:.6e}, ||r||/||b|| = {rel_res:.6e}")
                if (len(history) - 1) % step != 0:
                    rel_res = history[-1] / initial_residual_norm if initial_residual_norm > 0 else history[-1]
                    lines.append(f"  iter {len(history):4d}: ||r|| = {history[-1]:.6e}, ||r||/||b|| = {rel_res:.6e}")
                return "\n".join(lines)

            residual_info = _format_residual_history(true_residual_history)
            raise RuntimeError(
                f"Coupled iterative solver did not converge after {context.maxiter} iterations.\n"
                f"Final true residual ||r||: {final_residual:.2e}\n"
                f"Final relative residual ||r||/||b||: {final_relative_residual:.2e}\n"
                f"Tolerance (rtol): {context.tol:.2e}\n"
                f"Residual history:\n{residual_info}\n"
                f"Try increasing maxiter, loosening tol, or using a different preconditioner."
            )

        # Extract voltages
        t0 = time.perf_counter()

        v_p = solution[:context.n_ports]
        v_t = solution[context.n_ports:]

        port_voltages: Dict[Any, float] = {}
        for i, node in enumerate(context.top_blocks.port_nodes):
            port_voltages[node] = float(v_p[i])

        top_grid_voltages: Dict[Any, float] = {}
        for node in context.top_grid_pads:
            top_grid_voltages[node] = context.vdd
        for node, v in port_voltages.items():
            top_grid_voltages[node] = v
        for i, node in enumerate(context.top_blocks.interior_nodes):
            top_grid_voltages[node] = float(v_t[i])

        timings['extract_top_voltages'] = time.perf_counter() - t0

        # Recover bottom-grid voltages
        t0 = time.perf_counter()

        bottom_interior_voltages = recover_bottom_voltages(
            bottom_blocks=context.bottom_blocks,
            port_voltages=v_p,
            current_injections=bottom_currents,
            rhs_dirichlet_bottom=context.rhs_dirichlet_bottom,
        )

        bottom_grid_voltages: Dict[Any, float] = {}
        bottom_grid_voltages.update(port_voltages)
        bottom_grid_voltages.update(bottom_interior_voltages)

        timings['recover_bottom_voltages'] = time.perf_counter() - t0

        # Merge results
        t0 = time.perf_counter()

        all_voltages: Dict[Any, float] = {}
        all_voltages.update(top_grid_voltages)
        for n, v in bottom_interior_voltages.items():
            all_voltages[n] = v

        ir_drop = self.model.ir_drop(all_voltages)

        timings['merge_results'] = time.perf_counter() - t0

        if verbose:
            total_time = sum(timings.values())
            print(f"\n=== Coupled Hierarchical Solve (Prepared) ===")
            print(f"  Ports: {context.n_ports:,}, Top interior: {context.n_top_interior:,}")
            print(f"  Bottom interior: {context.bottom_blocks.n_interior:,}")
            print(f"  Solver: {context.solver}, Preconditioner: {context.preconditioner_type}")
            print(f"  ---")
            print(f"  Convergence:")
            print(f"    Iterations: {iterations}")
            print(f"    Initial ||b||: {initial_residual_norm:.6e}")
            print(f"    Final ||r||: {final_residual:.6e}")
            print(f"    Final ||r||/||b||: {final_relative_residual:.6e}")
            print(f"  ---")
            print(f"  Timing breakdown:")
            for step, t in timings.items():
                pct = t / total_time * 100
                print(f"    {step:25s}: {t*1000:8.1f} ms  ({pct:5.1f}%)")
            print(f"    {'TOTAL':25s}: {total_time*1000:8.1f} ms")
            print(f"==============================================\n")

        return UnifiedCoupledHierarchicalResult(
            voltages=all_voltages,
            ir_drop=ir_drop,
            partition_layer=context.partition_layer,
            top_grid_voltages=top_grid_voltages,
            bottom_grid_voltages=bottom_grid_voltages,
            port_nodes=context.port_nodes,
            port_voltages=port_voltages,
            iterations=iterations,
            final_residual=final_residual,
            converged=converged,
            preconditioner_type=context.preconditioner_type,
            timings=timings,
        )

    def _build_coupled_preconditioner(
        self,
        preconditioner_type: str,
        top_blocks: BlockMatrixSystem,
        bottom_blocks: BlockMatrixSystem,
    ) -> Optional[spla.LinearOperator]:
        """Build preconditioner for coupled system.

        Args:
            preconditioner_type: 'none', 'block_diagonal', or 'ilu'
            top_blocks: Top-grid block matrices
            bottom_blocks: Bottom-grid block matrices

        Returns:
            LinearOperator for preconditioning, or None if 'none'
        """
        if preconditioner_type == 'none':
            return None

        if preconditioner_type == 'block_diagonal':
            # Use diagonal of bottom G_pp as approximation to Schur complement diagonal
            bottom_G_pp_diag = np.array(bottom_blocks.G_pp.diagonal()).flatten()

            # Factor top interior if not already done
            if top_blocks.lu_ii is None and top_blocks.n_interior > 0:
                top_blocks.factor_interior()

            return BlockDiagonalPreconditioner(top_blocks, bottom_G_pp_diag)

        if preconditioner_type == 'ilu':
            return ILUPreconditioner(top_blocks, bottom_blocks.G_pp)

        if preconditioner_type == 'amg':
            from core.coupled_system import AMGPreconditioner, HAS_PYAMG
            if not HAS_PYAMG:
                raise ImportError(
                    "pyamg is required for AMG preconditioner. "
                    "Install it with: pip install pyamg"
                )
            return AMGPreconditioner(top_blocks, bottom_blocks.G_pp)

        raise ValueError(
            f"Unknown preconditioner type: {preconditioner_type}. "
            "Use 'none', 'block_diagonal', 'ilu', or 'amg'."
        )
