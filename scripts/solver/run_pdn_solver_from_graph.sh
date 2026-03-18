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
# Time-Domain (distributed only):
#   mode: transient                  # dc, quasi-static, transient (default: dc)
#   t_start: 0.0                     # Start time in seconds
#   t_end: 100e-9                    # End time in seconds
#   dt: 0.1e-9                       # Time step for transient
#   n_points: 101                    # Number of time points for quasi-static
#   method: be                       # Integration method: be or trap
#   smooth: true                     # Enable PWL smoothing
#
# ============================================================================

# Usage function
usage() {
    echo "Usage: $0 [-d] [-b BACKEND] -i INPUT_PKL -n NET_NAME [-o OUTPUT_DIR] [-c CONFIG_FILE] [-v]"
    echo ""
    echo "Required arguments (standard mode):"
    echo "  -i INPUT_PKL    Path to input PDN graph pickle file (or .pkl dir for -d)"
    echo "  -n NET_NAME     Power net name (e.g., VDD, VSS)"
    echo ""
    echo "Optional arguments:"
    echo "  -o OUTPUT_DIR   Output directory (default: ./results)"
    echo "  -c CONFIG_FILE  Config file (.yaml, .yml, or .json) for solver parameters"
    echo "  -d              Distributed DDM mode: load per-tile .pkl files and run DDM solver"
    echo "  -b BACKEND      Compute backend for distributed mode: local or ray (default: local)"
    echo "  -m MODE         Analysis mode: dc, quasi-static, transient (default: dc) [distributed only]"
    echo "  -t T_END        End time in seconds, e.g. 100e-9 (default: 100ns) [distributed only]"
    echo "  -s DT           Time step for transient in seconds, e.g. 0.1e-9 [distributed only]"
    echo "  -p N_POINTS     Number of time points for quasi-static (default: 101) [distributed only]"
    echo "  -M METHOD       Integration method: be or trap (default: be) [distributed only]"
    echo "  -v              Enable verbose output"
    echo ""
    echo "PRECEDENCE: Config file values OVERRIDE CLI arguments."
    echo ""
    echo "Example (standard):"
    echo "  $0 -i ./netlist_data/pdn_graph.pkl -n VDD -o ./results"
    echo "  $0 -i ./netlist_data/pdn_graph.pkl -n VDD -c solver_config.yaml"
    echo ""
    echo "Example (distributed):"
    echo "  $0 -d -i ./netlist_data/distributed_pkl -o ./results -v"
    echo "  $0 -d -b ray -i ./netlist_data/distributed_pkl -o ./results"
    echo ""
    echo "Example (distributed quasi-static):"
    echo "  $0 -d -i ./netlist_data/distributed_pkl -m quasi-static -t 100e-9 -p 51 -o ./results"
    echo ""
    echo "Example (distributed transient):"
    echo "  $0 -d -i ./netlist_data/distributed_pkl -m transient -t 10e-9 -s 0.1e-9 -o ./results"
    echo "  $0 -d -i ./netlist_data/distributed_pkl -m transient -t 10e-9 -s 0.1e-9 -M trap -v"
    echo ""
    echo "See script header for full list of config file parameters."
    exit 1
}

# Default values
OUTPUT_DIR="./results"
CONFIG_FILE=""
VERBOSE=""
DISTRIBUTED=""
BACKEND="local"

# Parse command-line arguments
while getopts "i:n:o:c:b:m:t:s:p:M:dvh" opt; do
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
        d)
            DISTRIBUTED="1"
            ;;
        b)
            BACKEND="$OPTARG"
            ;;
        m)
            MODE="$OPTARG"
            ;;
        t)
            T_END="$OPTARG"
            ;;
        s)
            DT="$OPTARG"
            ;;
        p)
            N_POINTS="$OPTARG"
            ;;
        M)
            METHOD="$OPTARG"
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

# Distributed DDM mode
if [ -n "$DISTRIBUTED" ]; then
    if [ -z "$INPUT_PKL" ]; then
        echo "Error: Input .pkl directory (-i) is required for distributed mode" >&2
        usage
    fi
    if [ ! -d "$INPUT_PKL" ]; then
        echo "Error: '$INPUT_PKL' is not a directory (expected .pkl directory for -d mode)" >&2
        exit 1
    fi

    CMD_ARGS="solve \"$INPUT_PKL\" --backend \"$BACKEND\" --plot"
    if [ -n "$OUTPUT_DIR" ]; then
        CMD_ARGS="$CMD_ARGS --output \"$OUTPUT_DIR\""
    fi
    if [ -n "$CONFIG_FILE" ]; then
        CMD_ARGS="$CMD_ARGS --config \"$CONFIG_FILE\""
    fi
    if [ -n "$MODE" ]; then
        CMD_ARGS="$CMD_ARGS --mode \"$MODE\""
    fi
    if [ -n "$T_END" ]; then
        CMD_ARGS="$CMD_ARGS --t-end \"$T_END\""
    fi
    if [ -n "$DT" ]; then
        CMD_ARGS="$CMD_ARGS --dt \"$DT\""
    fi
    if [ -n "$N_POINTS" ]; then
        CMD_ARGS="$CMD_ARGS --n-points \"$N_POINTS\""
    fi
    if [ -n "$METHOD" ]; then
        CMD_ARGS="$CMD_ARGS --method \"$METHOD\""
    fi
    if [ -n "$VERBOSE" ]; then
        CMD_ARGS="$CMD_ARGS --verbose"
    fi
    CMD_ARGS="$CMD_ARGS --max-stripes 2000"

    CMD="python -m distributed $CMD_ARGS"
    echo "Running: $CMD"
    echo ""
    eval $CMD
    exit $?
fi

# Standard mode: check required parameters
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
eval python -m src.solver.pdn_solver $CMD_ARGS
