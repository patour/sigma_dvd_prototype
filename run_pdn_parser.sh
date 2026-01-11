#!/bin/bash

# Parse PDN netlist and generate pickle graph
# 
# Usage:
#   ./run_pdn_parser.sh                                    # Uses defaults
#   ./run_pdn_parser.sh ./pdn/netlist_test                # Custom netlist dir
#   ./run_pdn_parser.sh ./pdn/netlist_test VSS            # Custom dir and net name
#
# Parameters:
#   $1 - Netlist directory (default: ./netlist_data)
#   $2 - Power net name (default: VDD)
#
# Output:
#   <netlist_dir>/pdn_graph.pkl - Serialized NetworkX graph

# Default parameters
NETLIST_DIR="${1:-./ netlist_data}"
NET_NAME="${2:-VDD}"
OUTPUT="${NETLIST_DIR}/pdn_graph.pkl"

python ./pdn/pdn_parser.py --netlist-dir "$NETLIST_DIR" --net "$NET_NAME" --output "$OUTPUT" --verbose
