#!/usr/bin/env python3
"""
PDN Netlist Parser - CLI entry point.

Sub-module layout:
  parser.metadata        — PDNNodeAttrs, ParseStats, net type tables
  parser.spice_lexer     — Constants, regex, _parse_spice_value, SpiceLineReader, net filters
  parser.current_sources — Pulse, PWL, CurrentSource, waveform parsing, ContextVars
  parser.graph_builder   — GraphBuilder, optimized-edges ContextVar
  parser.netlist         — NetlistParser facade, _PDNUnpickler, load_pdn_pickle
"""

import pickle
import sys
from pathlib import Path

from .netlist import NetlistParser, _PDNUnpickler, load_pdn_pickle  # noqa: F401
from .current_sources import InstanceInfo, Pulse, PWL, CurrentSource  # noqa: F401


def main():
    """Command-line interface"""
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description='Parse PDN netlist and convert to NetworkX graph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic parsing
  python -m parser.pdn_parser --netlist-dir /path/to/netlist

  # With validation and output
  python -m parser.pdn_parser --netlist-dir /path/to/netlist --validate --output pdn.pkl

  # Strict mode (fail on errors)
  python -m parser.pdn_parser --netlist-dir /path/to/netlist --validate --strict

  # Filter specific net
  python -m parser.pdn_parser --netlist-dir /path/to/netlist --net vdd

  # With memory profiling
  python -m parser.pdn_parser --netlist-dir /path/to/netlist --profile-memory
        """
    )

    parser.add_argument('--netlist-dir', type=str, default='.',
                       help='Directory containing ckt.sp (default: current directory)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file (.graphml or .pkl)')
    parser.add_argument('--validate', action='store_true',
                       help='Enable sanity checks (shorts, floating nodes, etc.)')
    parser.add_argument('--strict', action='store_true',
                       help='Treat warnings as errors (fail on validation issues)')
    parser.add_argument('--net', type=str,
                       help='Filter specific power net (e.g., vdd, vcc)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output (debug logging)')
    parser.add_argument('--profile-memory', action='store_true',
                       help='Profile memory usage (requires memory_profiler)')
    parser.add_argument('--vsrc-resistor-pattern', type=str, default='rs',
                       help='Resistor name pattern for voltage source identification (default: rs)')
    parser.add_argument('--vsrc-depth-limit', type=int, default=3,
                       help='Depth limit for voltage source node propagation (default: 3)')

    # Parallel parsing options
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel tile parsing using multiprocessing')
    parser.add_argument('--n-workers', type=int, default=None,
                       help='Number of parallel workers (default: min(cpu_count, 16))')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Lines per chunk for parallel file reading (default: 10000)')

    # Plotting options
    parser.add_argument('--plot-layer', type=str,
                       help='Plot specific layer (e.g., "5", "M1")')
    parser.add_argument('--plot-all-layers', action='store_true',
                       help='Plot all layers in a single figure')
    parser.add_argument('--plot-output', type=str,
                       help='Output file for plot (default: show plot)')
    parser.add_argument('--plot-bin-size', type=int,
                       help='Bin size for grid aggregation (auto-calculated if not specified)')
    parser.add_argument('--plot-statistic', type=str, default='node_count',
                       choices=['node_count', 'avg_voltage', 'total_capacitance', 'avg_resistance'],
                       help='Statistic to display in plot (default: node_count)')

    args = parser.parse_args()

    # Memory profiling
    if args.profile_memory:
        try:
            from memory_profiler import profile
            # Wrap parse function
            netlist_parser = NetlistParser(
                args.netlist_dir,
                validate=args.validate,
                strict=args.strict,
                net_filter=args.net,
                verbose=args.verbose,
                vsrc_resistor_pattern=args.vsrc_resistor_pattern,
                vsrc_depth_limit=args.vsrc_depth_limit,
                parallel=args.parallel,
                n_workers=args.n_workers,
                chunk_size=args.chunk_size
            )
            profiled_parse = profile(netlist_parser.parse)
            graph = profiled_parse()
        except ImportError:
            print("ERROR: memory_profiler not installed. Install with: pip install memory_profiler")
            sys.exit(1)
    else:
        # Normal parsing
        netlist_parser = NetlistParser(
            args.netlist_dir,
            validate=args.validate,
            strict=args.strict,
            net_filter=args.net,
            verbose=args.verbose,
            vsrc_resistor_pattern=args.vsrc_resistor_pattern,
            vsrc_depth_limit=args.vsrc_depth_limit,
            parallel=args.parallel,
            n_workers=args.n_workers,
            chunk_size=args.chunk_size
        )
        graph = netlist_parser.parse()

    # Save output
    if args.output:
        output_path = Path(args.output)

        if output_path.suffix == '.graphml':
            print("ERROR: GraphML export not supported with rustworkx backend")
            print("Use .pkl format instead")
            sys.exit(1)
        elif output_path.suffix == '.pkl':
            # Register pickle handlers to fix __main__ module references
            if __name__ == '__main__':
                import copyreg
                from dataclasses import fields
                import parser.pdn_parser as target_module

                def make_reducer(cls_name):
                    """Create reducer that uses class from parser.pdn_parser module."""
                    def reducer(obj):
                        correct_cls = getattr(target_module, cls_name)
                        field_values = tuple(
                            getattr(obj, f.name) for f in fields(obj) if f.init
                        )
                        return (correct_cls, field_values)
                    return reducer

                for cls_name in ('InstanceInfo', 'Pulse', 'PWL', 'CurrentSource'):
                    cls = globals()[cls_name]
                    copyreg.pickle(cls, make_reducer(cls_name))

            with open(output_path, 'wb') as f:
                pickle.dump(graph, f)
            print(f"Graph saved to: {output_path}")
        else:
            print(f"ERROR: Unsupported output format: {output_path.suffix}")
            print("Supported formats: .pkl")
            sys.exit(1)

    # Print summary
    print(f"\nGraph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    print(f"Instance-node mappings: {len(graph.graph.get('instance_node_map', {}))}")

    # Plotting using PDNPlotter
    if args.plot_layer or args.plot_all_layers:
        try:
            from visualization.pdn_plotter import PDNPlotter

            # Get net connectivity for PDNPlotter
            net_connectivity = graph.graph.get('net_connectivity', {})

            # Create plotter
            plotter = PDNPlotter(graph, net_connectivity, logging.getLogger(__name__))

            # Set up output path and filename
            if args.plot_output:
                output_path = Path(args.plot_output)
                if output_path.suffix:  # It's a file
                    output_dir = output_path.parent
                    output_filename = output_path.name
                else:  # It's a directory
                    output_dir = output_path
                    output_filename = None
            else:
                output_dir = Path('./results')
                output_filename = None

            output_dir.mkdir(parents=True, exist_ok=True)

            # Determine net name for plotting
            net_name = args.net if args.net else 'PDN'

            if args.plot_layer:
                print(f"\nPlotting layer {args.plot_layer}...")
                layers = [l.strip() for l in args.plot_layer.split(',')]

                plotter.generate_layer_heatmaps(
                    net_name=net_name,
                    output_path=output_dir,
                    plot_layers=layers,
                    plot_bin_size=args.plot_bin_size,
                    anisotropic_bins=True,
                    output_filename=output_filename
                )

            elif args.plot_all_layers:
                print("\nPlotting all layers...")

                plotter.generate_layer_heatmaps(
                    net_name=net_name,
                    output_path=output_dir,
                    plot_layers=None,
                    plot_bin_size=args.plot_bin_size,
                    anisotropic_bins=True,
                    output_filename=output_filename
                )

        except ImportError:
            print("ERROR: PDNPlotter not available.")
        except Exception as e:
            print(f"ERROR during plotting: {e}")
            import traceback
            traceback.print_exc()

    return 0


if __name__ == '__main__':
    sys.exit(main())
