#!/usr/bin/env python3
"""
Test script for validating tiled bottom-grid solver accuracy.

This script validates the tiled bottom-grid IR-drop solver independently of
top-grid accuracy. It uses the flat solver's port voltages as ground truth
Dirichlet boundary conditions, isolating tiling-induced errors.

Validation approach:
1. Run flat (direct) solver to get reference voltages for all nodes
2. Extract port voltages from flat solver result
3. Run tiled bottom-grid solver using flat port voltages as Dirichlet BCs
4. Compare tiled bottom-grid results against flat bottom-grid results

Usage:
    python test_tiled_accuracy.py --netlist /path/to/netlist --tilings 2x2,4x4 --halo 0.2 --partition 32

Example:
    python test_tiled_accuracy.py \\
        --netlist /wv/bwdev1/patrasej/dev/sigma_dvd/mpower_testcases/minion/db.sir/db/netlist \\
        --tilings 2x2,4x4,10x10 \\
        --halo 0.2 \\
        --partition 32 \\
        --net VDD_XLV \\
        --workers 8
"""

import argparse
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pdn'))

from pdn_parser import NetlistParser
from core import create_model_from_pdn, UnifiedIRDropSolver


class Logger:
    """Simple logger that writes to both stdout and a log file."""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_file = open(log_path, 'w')
        
    def log(self, msg: str = "", to_stdout: bool = True):
        """Write message to log file and optionally to stdout."""
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        if to_stdout:
            print(msg)
    
    def close(self):
        self.log_file.close()


def parse_tilings(tilings_str: str) -> List[Tuple[int, int]]:
    """Parse comma-separated tiling specifications like '2x2,4x4,10x10'."""
    tilings = []
    for spec in tilings_str.split(','):
        spec = spec.strip()
        if 'x' not in spec:
            raise ValueError(f"Invalid tiling spec '{spec}'. Expected format: NxM (e.g., 2x2)")
        parts = spec.split('x')
        if len(parts) != 2:
            raise ValueError(f"Invalid tiling spec '{spec}'. Expected format: NxM (e.g., 2x2)")
        n_x, n_y = int(parts[0]), int(parts[1])
        tilings.append((n_x, n_y))
    return tilings


def load_pdn_graph(netlist_dir: Path, logger: Logger):
    """Load PDN graph from pickle cache or parse from netlist."""
    pkl_path = netlist_dir / 'pdn_graph.pkl'
    
    if pkl_path.exists():
        logger.log(f"Loading cached graph from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            graph = pickle.load(f)
        logger.log("Loaded cached graph successfully!")
    else:
        logger.log(f"Parsing PDN netlist from {netlist_dir}...")
        logger.log("(This may take several minutes for large netlists)")
        parser = NetlistParser(str(netlist_dir), validate=True)
        graph = parser.parse()
        logger.log("Parsing complete!")
    
    logger.log(f"\nParsed PDN Graph Statistics:")
    logger.log(f"  Nodes: {graph.number_of_nodes()}")
    logger.log(f"  Edges: {graph.number_of_edges()}")
    
    return graph


def create_model_and_extract_currents(graph, net_name: str, logger: Logger):
    """Create unified power grid model and extract load currents."""
    logger.log(f"\nCreating model for net: {net_name}")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = create_model_from_pdn(graph, net_name)
        for warning in w:
            logger.log(f"  Warning: {warning.message}")
    
    logger.log(f"  Vdd: {model.vdd} V")
    logger.log(f"  Pad nodes: {len(model.pad_nodes)}")
    logger.log(f"  Unknowns: {len(model.reduced.unknown_nodes)}")
    
    # Extract current sources
    load_currents = model.extract_current_sources()
    logger.log(f"\nExtracted {len(load_currents)} current sources")
    
    if load_currents:
        total_current = sum(load_currents.values())
        logger.log(f"  Total current: {total_current:.4f} mA")
    else:
        logger.log("ERROR: No current sources found in PDN graph!")
        sys.exit(1)
    
    return model, load_currents


def run_flat_solver(solver: UnifiedIRDropSolver, load_currents: Dict, logger: Logger):
    """Run flat (direct) solver to get reference results."""
    logger.log("\n" + "="*70)
    logger.log("FLAT SOLVER (Reference)")
    logger.log("="*70)
    
    t0 = time.perf_counter()
    flat_result = solver.solve(load_currents)
    flat_time = time.perf_counter() - t0
    
    summary = solver.summarize(flat_result)
    
    logger.log(f"  Solve time: {flat_time*1000:.1f} ms")
    logger.log(f"  Nominal voltage: {summary['nominal_voltage']:.6f} V")
    logger.log(f"  Min voltage: {summary['min_voltage']:.6f} V")
    logger.log(f"  Max voltage: {summary['max_voltage']:.6f} V")
    logger.log(f"  Max IR-drop: {summary['max_drop']*1000:.4f} mV")
    logger.log(f"  Avg IR-drop: {summary['avg_drop']*1000:.4f} mV")
    
    return flat_result, flat_time


def validate_bottom_grid_solver(
    model, solver: UnifiedIRDropSolver, 
    load_currents: Dict, flat_result,
    partition_layer: str, logger: Logger
) -> bool:
    """
    Validate bottom-grid solver using flat solver port voltages.
    Returns True if validation passes (errors within numerical noise).
    """
    logger.log("\n" + "="*70)
    logger.log("BOTTOM-GRID SOLVER VALIDATION")
    logger.log("="*70)
    logger.log("Using flat solver port voltages as Dirichlet boundary conditions")
    
    # Decompose the PDN
    top_nodes, bottom_nodes, ports, via_edges = model._decompose_at_layer(partition_layer)
    flat_voltages = flat_result.voltages
    
    # Get the flat solver's port voltages (ground truth)
    flat_port_voltages = {p: flat_voltages[p] for p in ports if p in flat_voltages}
    
    logger.log(f"\nDecomposition at partition layer '{partition_layer}':")
    logger.log(f"  Top-grid nodes: {len(top_nodes)}")
    logger.log(f"  Bottom-grid nodes: {len(bottom_nodes)}")
    logger.log(f"  Port nodes: {len(ports)}")
    
    # Build bottom-grid system with ports as Dirichlet nodes
    bottom_subgrid = bottom_nodes | ports
    bottom_system = model._build_subgrid_system(
        subgrid_nodes=bottom_subgrid,
        dirichlet_nodes=ports,
        dirichlet_voltage=model.vdd,
    )
    
    if bottom_system is None:
        logger.log("ERROR: Could not build bottom-grid system for validation!")
        return False
    
    # Get bottom-grid currents
    bottom_grid_currents = {n: c for n, c in load_currents.items() if n in bottom_nodes}
    logger.log(f"  Bottom-grid loads: {len(bottom_grid_currents)}")
    
    # Solve with flat solver's port voltages
    validated_bottom_voltages = model._solve_subgrid(
        reduced_system=bottom_system,
        current_injections=bottom_grid_currents,
        dirichlet_voltages=flat_port_voltages,
    )
    
    # Compare with flat solver's bottom-grid voltages
    errors = []
    for node in bottom_nodes:
        if node in validated_bottom_voltages and node in flat_voltages:
            error = abs(validated_bottom_voltages[node] - flat_voltages[node])
            errors.append(error)
    
    if not errors:
        logger.log("ERROR: No bottom-grid nodes to validate!")
        return False
    
    errors = np.array(errors)
    
    logger.log(f"\nValidation Results:")
    logger.log(f"  Max error:  {errors.max():.2e} V ({errors.max()*1e6:.4f} µV)")
    logger.log(f"  Mean error: {errors.mean():.2e} V ({errors.mean()*1e6:.4f} µV)")
    logger.log(f"  RMS error:  {np.sqrt((errors**2).mean()):.2e} V")
    
    # Check if errors are within numerical noise (< 1 nV)
    if errors.max() < 1e-9:
        logger.log(f"\n✓ VALIDATION PASSED: Bottom-grid solver produces accurate results")
        return True
    else:
        logger.log(f"\n✗ VALIDATION FAILED: Errors exceed numerical noise threshold (1 nV)")
        return False


def run_tiled_bottom_grid_validation(
    solver: UnifiedIRDropSolver,
    load_currents: Dict,
    flat_result,
    partition_layer: str,
    n_x: int, n_y: int,
    halo_percent: float,
    n_workers: int,
    logger: Logger
):
    """
    Run tiled bottom-grid solver ONLY using flat solver's port voltages as BCs.
    
    This validates the tiled bottom-grid solver independently of top-grid accuracy,
    by using the flat solver's port voltages as ground truth Dirichlet BCs.
    """
    logger.log(f"\n--- Tiling: {n_x}x{n_y} ---")
    
    # Get port voltages from flat solver
    model = solver.model
    top_nodes, bottom_nodes, port_nodes, via_edges = model._decompose_at_layer(partition_layer)
    flat_port_voltages = {p: flat_result.voltages[p] for p in port_nodes if p in flat_result.voltages}
    
    logger.log(f"  Using {len(flat_port_voltages)} port voltages from flat solver as Dirichlet BCs")
    
    t0 = time.perf_counter()
    tiled_result = solver.solve_hierarchical_tiled(
        current_injections=load_currents,
        partition_layer=partition_layer,
        N_x=n_x,
        N_y=n_y,
        halo_percent=halo_percent,
        top_k=5,  # Not used when override_port_voltages is set
        weighting='shortest_path',  # Not used when override_port_voltages is set
        n_workers=n_workers,
        parallel_backend='thread',
        validate_against_flat=True,
        verbose=False,
        override_port_voltages=flat_port_voltages,  # Key: use flat solver's port voltages
    )
    total_time = time.perf_counter() - t0
    
    return tiled_result, total_time, bottom_nodes, port_nodes


def compute_tile_errors(tiled_result, flat_result, model) -> List[Dict]:
    """Compute per-tile error statistics."""
    tile_stats = []
    flat_voltages = flat_result.voltages
    
    for tile in tiled_result.tiles:
        # Count bottom-grid nodes in core (excluding port nodes)
        core_nodes_count = len(tile.core_nodes)
        halo_nodes_count = len(tile.halo_nodes)
        port_nodes_count = len(tile.port_nodes)
        load_nodes_count = len(tile.load_nodes)
        
        # Halo percentage relative to core
        if core_nodes_count > 0:
            halo_percent = halo_nodes_count / core_nodes_count * 100
        else:
            halo_percent = 0.0
        
        # Compute voltage errors for core nodes
        errors = []
        for node in tile.core_nodes:
            if node in tiled_result.voltages and node in flat_voltages:
                error = abs(tiled_result.voltages[node] - flat_voltages[node])
                errors.append(error)
        
        if errors:
            errors = np.array(errors)
            max_error = errors.max()
            mean_error = errors.mean()
            rmse = np.sqrt((errors**2).mean())
        else:
            max_error = mean_error = rmse = 0.0
        
        # Get solve time
        solve_time = tiled_result.per_tile_solve_times.get(tile.tile_id, 0.0) / 1000.0  # Convert ms to s
        
        tile_stats.append({
            'tile_id': tile.tile_id,
            'core_nodes': core_nodes_count,
            'halo_nodes': halo_nodes_count,
            'halo_percent': halo_percent,
            'port_nodes': port_nodes_count,
            'load_nodes': load_nodes_count,
            'max_error_mV': max_error * 1000,
            'mean_error_mV': mean_error * 1000,
            'rmse_mV': rmse * 1000,
            'solve_time_s': solve_time,
            'halo_clipped': tile.halo_clipped,
        })
    
    return tile_stats


def print_tile_table(tile_stats: List[Dict], logger: Logger):
    """Print per-tile error table."""
    logger.log("\nPer-Tile Error Report:")
    logger.log("-" * 130)
    header = (
        f"{'Tile':>6} | {'Core':>8} | {'Halo':>8} | {'Halo%':>7} | "
        f"{'Ports':>6} | {'Loads':>6} | {'Max Err':>10} | {'Mean Err':>10} | "
        f"{'RMSE':>10} | {'Time':>8} | {'Clipped':>7}"
    )
    logger.log(header)
    logger.log("-" * 130)
    
    for s in tile_stats:
        row = (
            f"{s['tile_id']:>6} | {s['core_nodes']:>8} | {s['halo_nodes']:>8} | "
            f"{s['halo_percent']:>6.1f}% | {s['port_nodes']:>6} | {s['load_nodes']:>6} | "
            f"{s['max_error_mV']:>9.4f} | {s['mean_error_mV']:>9.4f} | "
            f"{s['rmse_mV']:>9.4f} | {s['solve_time_s']:>7.3f}s | "
            f"{'Yes' if s['halo_clipped'] else 'No':>7}"
        )
        logger.log(row)
    
    logger.log("-" * 130)


def main():
    parser = argparse.ArgumentParser(
        description='Test tiled hierarchical solver accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--netlist', '-n',
        type=str,
        default='/wv/bwdev1/patrasej/dev/sigma_dvd/mpower_testcases/minion/db.sir/db/netlist',
        help='Path to PDN netlist directory'
    )
    parser.add_argument(
        '--tilings', '-t',
        type=str,
        default='2x2,4x4',
        help='Comma-separated list of tilings (e.g., 2x2,4x4,10x10)'
    )
    parser.add_argument(
        '--halo', '-H',
        type=float,
        default=0.2,
        help='Halo percentage (e.g., 0.2 for 20%%)'
    )
    parser.add_argument(
        '--partition', '-p',
        type=str,
        default='32',
        help='Partition layer name'
    )
    parser.add_argument(
        '--net', '-N',
        type=str,
        default='VDD_XLV',
        help='Power net name'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of parallel workers for tile solving'
    )
    
    args = parser.parse_args()
    
    # Parse tilings
    try:
        tilings = parse_tilings(args.tilings)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = Path(f'tiled_accuracy_{timestamp}.log')
    logger = Logger(log_path)
    
    # Print header
    logger.log("=" * 70)
    logger.log("TILED BOTTOM-GRID SOLVER ACCURACY TEST")
    logger.log("=" * 70)
    logger.log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Log file: {log_path}")
    logger.log(f"\nParameters:")
    logger.log(f"  Netlist: {args.netlist}")
    logger.log(f"  Net name: {args.net}")
    logger.log(f"  Partition layer: {args.partition}")
    logger.log(f"  Halo percent: {args.halo*100:.0f}%")
    logger.log(f"  Tilings: {', '.join(f'{n}x{m}' for n, m in tilings)}")
    logger.log(f"  Workers: {args.workers}")
    
    # Suppress warnings during execution
    warnings.filterwarnings('ignore')
    
    try:
        # Load PDN graph
        netlist_dir = Path(args.netlist)
        if not netlist_dir.exists():
            logger.log(f"ERROR: Netlist directory not found: {netlist_dir}")
            sys.exit(1)
        
        graph = load_pdn_graph(netlist_dir, logger)
        
        # Create model and extract currents
        model, load_currents = create_model_and_extract_currents(graph, args.net, logger)
        
        # Create solver
        solver = UnifiedIRDropSolver(model)
        
        # Run flat solver (reference)
        flat_result, flat_time = run_flat_solver(solver, load_currents, logger)
        flat_summary = solver.summarize(flat_result)
        
        # Validate bottom-grid solver
        validation_passed = validate_bottom_grid_solver(
            model, solver, load_currents, flat_result, args.partition, logger
        )
        
        if not validation_passed:
            logger.log("\nERROR: Bottom-grid solver validation failed!")
            logger.log("Exiting with error code 1.")
            logger.close()
            sys.exit(1)
        
        # Run tiled bottom-grid solver for each configuration
        logger.log("\n" + "=" * 70)
        logger.log("TILED BOTTOM-GRID SOLVER VALIDATION")
        logger.log("(Using flat solver's port voltages as Dirichlet BCs)")
        logger.log("=" * 70)
        
        summary_results = []
        
        for n_x, n_y in tilings:
            tiled_result, total_time, bottom_nodes, port_nodes = run_tiled_bottom_grid_validation(
                solver, load_currents, flat_result, args.partition,
                n_x, n_y, args.halo, args.workers, logger
            )
            
            # Compute per-tile errors
            tile_stats = compute_tile_errors(tiled_result, flat_result, model)
            
            # Print tile table
            print_tile_table(tile_stats, logger)
            
            # Compute overall statistics
            validation_stats = tiled_result.validation_stats
            
            # Compute errors vs flat solver for BOTTOM-GRID NODES ONLY
            tiled_errors = []
            for node in bottom_nodes:
                if node in tiled_result.voltages and node in flat_result.voltages:
                    error = abs(flat_result.voltages[node] - tiled_result.voltages[node])
                    tiled_errors.append(error)
            tiled_errors = np.array(tiled_errors)
            
            # Compute IR-drop for bottom-grid nodes
            tiled_ir_drops = [
                model.vdd - tiled_result.voltages[n]
                for n in bottom_nodes
                if n in tiled_result.voltages
            ]
            max_irdrop = max(tiled_ir_drops) if tiled_ir_drops else 0.0
            
            logger.log(f"\nOverall Statistics for {n_x}x{n_y}:")
            logger.log(f"  Total tiles: {len(tiled_result.tiles)}")
            logger.log(f"  Bottom-grid nodes compared: {len(tiled_errors)}")
            logger.log(f"  Total solve time: {total_time:.3f} s")
            logger.log(f"  Max IR-drop (bottom-grid): {max_irdrop*1000:.4f} mV")
            
            if validation_stats:
                logger.log(f"\n  Validation vs Non-Tiled Bottom-Grid:")
                logger.log(f"    Max diff:  {validation_stats['max_diff']*1000:.4f} mV")
                logger.log(f"    Mean diff: {validation_stats['mean_diff']*1000:.4f} mV")
                logger.log(f"    RMSE:      {validation_stats['rmse']*1000:.4f} mV")
            
            logger.log(f"\n  Errors vs Flat Solver (bottom-grid only):")
            logger.log(f"    Max error:  {tiled_errors.max()*1000:.4f} mV")
            logger.log(f"    Mean error: {tiled_errors.mean()*1000:.4f} mV")
            logger.log(f"    RMSE:       {np.sqrt((tiled_errors**2).mean())*1000:.4f} mV")
            
            # Store for summary table
            summary_results.append({
                'tiling': f'{n_x}x{n_y}',
                'n_tiles': len(tiled_result.tiles),
                'bottom_nodes': len(tiled_errors),
                'max_error_mV': tiled_errors.max() * 1000,
                'mean_error_mV': tiled_errors.mean() * 1000,
                'rmse_mV': np.sqrt((tiled_errors**2).mean()) * 1000,
                'total_time_s': total_time,
                'max_irdrop_mV': max_irdrop * 1000,
                'validation_max_diff_mV': validation_stats['max_diff'] * 1000 if validation_stats else 0,
            })
        
        # Print summary table
        logger.log("\n" + "=" * 70)
        logger.log("SUMMARY TABLE")
        logger.log("=" * 70)
        logger.log(f"\nReference (Flat Solver):")
        logger.log(f"  Solve time: {flat_time*1000:.1f} ms")
        logger.log(f"  Max IR-drop: {flat_summary['max_drop']*1000:.4f} mV")
        
        logger.log(f"\nTiled Bottom-Grid Solver Comparison (vs Flat Solver):")
        logger.log("-" * 125)
        header = (
            f"{'Tiling':>8} | {'Tiles':>6} | {'BG Nodes':>9} | {'Max Err':>10} | {'Mean Err':>10} | "
            f"{'RMSE':>10} | {'Max IR-drop':>12} | {'Time':>10} | {'Valid Diff':>10}"
        )
        logger.log(header)
        logger.log("-" * 125)
        
        for r in summary_results:
            row = (
                f"{r['tiling']:>8} | {r['n_tiles']:>6} | {r['bottom_nodes']:>9} | {r['max_error_mV']:>9.4f} | "
                f"{r['mean_error_mV']:>9.4f} | {r['rmse_mV']:>9.4f} | "
                f"{r['max_irdrop_mV']:>11.4f} | {r['total_time_s']:>9.3f}s | "
                f"{r['validation_max_diff_mV']:>9.4f}"
            )
            logger.log(row)
        
        logger.log("-" * 125)
        logger.log(f"\nUnits: Errors and IR-drop in mV, Time in seconds")
        logger.log(f"BG Nodes = Bottom-grid nodes compared")
        logger.log(f"Valid Diff = Max voltage difference vs non-tiled bottom-grid solve (same port BCs)")
        
        # Final status
        logger.log("\n" + "=" * 70)
        logger.log("TEST COMPLETED SUCCESSFULLY")
        logger.log("=" * 70)
        logger.log(f"Log file saved to: {log_path.absolute()}")
        
    except Exception as e:
        logger.log(f"\nERROR: {e}")
        import traceback
        logger.log(traceback.format_exc())
        logger.close()
        sys.exit(1)
    
    logger.close()
    print(f"\nDone! Log saved to: {log_path}")


if __name__ == '__main__':
    main()
