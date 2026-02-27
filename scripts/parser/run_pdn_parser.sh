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
#
# Options:
#   -p, --parallel     Enable parallel tile parsing (recommended for multi-tile netlists)
#   -w, --workers N    Number of parallel workers (default: auto, max 16)
#   -c, --chunk-size N Lines per chunk for parallel reading (default: 10000)
#   -h, --help         Show this help message
#
# Positional Parameters:
#   NETLIST_DIR - Netlist directory (default: ./netlist_data)
#   NET_NAME    - Power net name to filter (default: VDD)
#
# Output:
#   <netlist_dir>/pdn_graph.pkl - Serialized graph

# Default parameters
PARALLEL=""
N_WORKERS=""
CHUNK_SIZE=""
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
        -h|--help)
            head -27 "$0" | tail -25
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
OUTPUT="${NETLIST_DIR}/pdn_graph.pkl"

# Build and run command
CMD="python -m src.parser.pdn_parser --netlist-dir \"$NETLIST_DIR\" --net \"$NET_NAME\" --output \"$OUTPUT\" --verbose $PARALLEL $N_WORKERS $CHUNK_SIZE"

echo "Running: $CMD"
echo ""

eval $CMD
