#!/bin/bash
#
# PDN Solver Runner - Run IR-drop analysis from a pre-parsed PDN graph
#
# PRECEDENCE: Config file values OVERRIDE CLI arguments.
#             If a parameter is specified in both, the config file value is used.
#
# ============================================================================
# CONFIG FILE PARAMETERS (YAML/JSON)
# ============================================================================
#
# Input/Output:
#   input: ./pdn_graph.pkl          # Input graph file (.pkl)
#   netlist_dir: ./netlist_data     # Alternative: parse netlist from directory
#   output: ./results               # Output directory for results
#   net: VDD                        # Solve only specific net (default: all nets)
#
# Solver Backend (cholmod vs splu):
#   use_cholmod: true               # true=cholmod, false=splu, omit=auto-detect
#   cholmod_mode: auto              # 'auto', 'simplicial', 'supernodal'
#   cholmod_ordering: default       # 'default', 'natural', 'amd', 'metis',
#                                   # 'nesdis', 'colamd', 'best'
#   cholmod_use_long: false         # Force 64-bit indices
#
# Reporting:
#   top_k: 100                      # Number of worst nodes to report
#   verbose: true                   # Enable verbose output with timing details
#   profile_memory: false           # Enable memory profiling (slower)
#
# Heatmap Plotting - Standard Mode:
#   plot_layers: "M1,M2,M3"         # Layers to plot (comma-separated), omit=all
#   plot_bin_size: 1000             # Override base bin size for heatmaps
#   anisotropic_bins: true          # Enable orientation-aware binning
#   bin_aspect_ratio: 50            # Aspect ratio for anisotropic bins
#   layer_orientations: "M1:H,M2:V" # Manual layer orientation override
#   show_irdrop: true               # true=IR-drop (mV), false=voltage (V)
#
# Heatmap Plotting - Stripe Mode:
#   stripe_mode: true               # Enable stripe-based plotting
#   max_stripes: 50                 # Max stripes before consolidation
#   stripe_bin_size: 10000          # Bin size for within-stripe aggregation
#
# ============================================================================

# Usage function
usage() {
    echo "Usage: $0 -i INPUT_PKL -n NET_NAME [-o OUTPUT_DIR] [-c CONFIG_FILE] [-v]"
    echo ""
    echo "Required arguments:"
    echo "  -i INPUT_PKL    Path to input PDN graph pickle file"
    echo "  -n NET_NAME     Power net name (e.g., VDD, VSS)"
    echo ""
    echo "Optional arguments:"
    echo "  -o OUTPUT_DIR   Output directory (default: ./results)"
    echo "  -c CONFIG_FILE  Config file (.yaml, .yml, or .json) for solver parameters"
    echo "  -v              Enable verbose output"
    echo ""
    echo "PRECEDENCE: Config file values OVERRIDE CLI arguments."
    echo ""
    echo "Example:"
    echo "  $0 -i ./netlist_data/pdn_graph.pkl -n VDD -o ./results"
    echo "  $0 -i ./netlist_data/pdn_graph.pkl -n VDD -c solver_config.yaml"
    echo ""
    echo "See script header for full list of config file parameters."
    exit 1
}

# Default values
OUTPUT_DIR="./results"
CONFIG_FILE=""
VERBOSE=""

# Parse command-line arguments
while getopts "i:n:o:c:vh" opt; do
    case $opt in
        i)
            INPUT_PKL="$OPTARG"
            ;;
        n)
            NET_NAME="$OPTARG"
            ;;
        o)
            OUTPUT_DIR="$OPTARG"
            ;;
        c)
            CONFIG_FILE="$OPTARG"
            ;;
        v)
            VERBOSE="--verbose"
            ;;
        h)
            usage
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            ;;
    esac
done

# Check required parameters
if [ -z "$INPUT_PKL" ]; then
    echo "Error: Input pickle file (-i) is required" >&2
    usage
fi

if [ -z "$NET_NAME" ]; then
    echo "Error: Net name (-n) is required" >&2
    usage
fi

# Check if input file exists
if [ ! -f "$INPUT_PKL" ]; then
    echo "Error: Input file '$INPUT_PKL' does not exist" >&2
    exit 1
fi

# Check if config file exists (if specified)
if [ -n "$CONFIG_FILE" ] && [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' does not exist" >&2
    exit 1
fi

# Build command arguments
CMD_ARGS="--input \"$INPUT_PKL\" --net \"$NET_NAME\" --output \"$OUTPUT_DIR\""

if [ -n "$CONFIG_FILE" ]; then
    CMD_ARGS="$CMD_ARGS --config \"$CONFIG_FILE\""
else
    # Use default parameters when no config file is provided
    CMD_ARGS="$CMD_ARGS --stripe-mode --max-stripes 2000 --stripe-bin-size 10000"
fi

if [ -n "$VERBOSE" ]; then
    CMD_ARGS="$CMD_ARGS --verbose"
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add script directory to PYTHONPATH so 'core' module can be found
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Run the PDN solver
eval python pdn/pdn_solver.py $CMD_ARGS
