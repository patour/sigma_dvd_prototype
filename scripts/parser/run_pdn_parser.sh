#!/bin/bash

# Parse PDN netlist and generate pickle graph
#
# Usage:
#   ./run_pdn_parser.sh [OPTIONS] [NETLIST_DIR] [NET_NAME]
#
# Examples:
#   ./run_pdn_parser.sh                                    # Uses defaults
#   ./run_pdn_parser.sh ./netlist/netlist_test                # Custom netlist dir
#   ./run_pdn_parser.sh ./netlist/netlist_test VSS            # Custom dir and net name
#   ./run_pdn_parser.sh -p ./netlist/netlist_multi_tile       # Enable parallel parsing
#   ./run_pdn_parser.sh -p -w 8 ./netlist/netlist_multi_tile  # Parallel with 8 workers
#   ./run_pdn_parser.sh -d ./netlist/netlist_sampled VDD_XLV  # Distributed parse only
#   ./run_pdn_parser.sh -d -b ray ./netlist/netlist_sampled VDD_XLV  # Distributed with Ray
#   ./run_pdn_parser.sh -d -o ./pkl_out ./netlist/netlist_sampled VDD_XLV
#
# Options:
#   -p, --parallel       Enable parallel tile parsing (recommended for multi-tile netlists)
#   -w, --workers N      Number of parallel workers (default: auto, max 16)
#   -c, --chunk-size N   Lines per chunk for parallel reading (default: 10000)
#   -d, --distributed    Dump per-tile .pkl files via distributed parser (parse_and_dump)
#   -b, --backend MODE   Compute backend for distributed mode: local or ray (default: local)
#   -o, --output DIR     Output directory for distributed .pkl files (default: <netlist_dir>/distributed_pkl)
#   -h, --help           Show this help message
#
# Positional Parameters:
#   NETLIST_DIR - Netlist directory (default: ./netlist_data)
#   NET_NAME    - Power net name to filter (default: VDD)
#
# Output (default mode):
#   <netlist_dir>/pdn_graph.pkl - Serialized graph
# Output (distributed mode):
#   <output_dir>/metadata.pkl + tile_X_Y.pkl files

# Default parameters
PARALLEL=""
N_WORKERS=""
CHUNK_SIZE=""
DISTRIBUTED=""
BACKEND=""
OUTPUT_DIR=""
NETLIST_DIR=""
NET_NAME=""

# Parse command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--parallel)
            PARALLEL="--parallel"
            shift
            ;;
        -w|--workers)
            N_WORKERS="--n-workers $2"
            shift 2
            ;;
        -c|--chunk-size)
            CHUNK_SIZE="--chunk-size $2"
            shift 2
            ;;
        -d|--distributed)
            DISTRIBUTED="1"
            shift
            ;;
        -b|--backend)
            BACKEND="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            head -35 "$0" | tail -33
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
        *)
            # Positional arguments
            if [[ -z "$NETLIST_DIR" ]]; then
                NETLIST_DIR="$1"
            elif [[ -z "$NET_NAME" ]]; then
                NET_NAME="$1"
            else
                echo "Too many positional arguments"
                exit 1
            fi
            shift
            ;;
    esac
done

# Apply defaults for positional parameters
NETLIST_DIR="${NETLIST_DIR:-./netlist_data}"
NET_NAME="${NET_NAME:-VDD}"

if [[ -n "$BACKEND" && -z "$DISTRIBUTED" ]]; then
    echo "Error: -b/--backend requires -d/--distributed mode"
    exit 1
fi

if [[ -n "$DISTRIBUTED" ]]; then
    # Distributed mode: dump per-tile .pkl files
    CMD_ARGS="parse \"$NETLIST_DIR\" --net \"$NET_NAME\" -v"
    if [[ -n "$BACKEND" ]]; then
        CMD_ARGS="$CMD_ARGS -b \"$BACKEND\""
    fi
    if [[ -n "$OUTPUT_DIR" ]]; then
        CMD_ARGS="$CMD_ARGS --output \"$OUTPUT_DIR\""
    fi
    CMD="python -m distributed $CMD_ARGS"
else
    # Default mode: monolithic graph pickle
    OUTPUT="${NETLIST_DIR}/pdn_graph.pkl"
    CMD="python -m src.parser.pdn_parser --netlist-dir \"$NETLIST_DIR\" --net \"$NET_NAME\" --output \"$OUTPUT\" --verbose $PARALLEL $N_WORKERS $CHUNK_SIZE"
fi

echo "Running: $CMD"
echo ""

eval $CMD
