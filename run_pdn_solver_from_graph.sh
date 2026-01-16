#!/bin/bash

# Usage function
usage() {
    echo "Usage: $0 -i INPUT_PKL -n NET_NAME [-o OUTPUT_DIR]"
    echo ""
    echo "Required arguments:"
    echo "  -i INPUT_PKL    Path to input PDN graph pickle file"
    echo "  -n NET_NAME     Power net name (e.g., VDD, VSS)"
    echo ""
    echo "Optional arguments:"
    echo "  -o OUTPUT_DIR   Output directory (default: ./results)"
    echo ""
    echo "Example:"
    echo "  $0 -i ./netlist_data/pdn_graph.pkl -n VDD -o ./results"
    exit 1
}

# Default values
OUTPUT_DIR="./results"

# Parse command-line arguments
while getopts "i:n:o:h" opt; do
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

# Run the PDN solver
python pdn/pdn_solver.py \
    --input "$INPUT_PKL" \
    --net "$NET_NAME" \
    --output "$OUTPUT_DIR" \
    --stripe-mode \
    --max-stripes 2000 \
    --stripe-bin-size 10000 \
    --verbose
