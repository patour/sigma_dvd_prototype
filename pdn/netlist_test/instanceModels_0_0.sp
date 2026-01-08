* Current Sources - Simulated load distribution
* Format: I<name> <node+> <node-> <dc_value> [static_value=<value>]
* Current flows from node+ to node- (sink convention)

* ============================================================================
* High current load cluster (bottom-left region) - Simulates dense logic
* ============================================================================

* Core cluster at (2000, 2000) - Heavy load
i_core:cluster1:2000_2000:vdd:0 2000_2000_M1 0 0.001 static_value=5e-3
i_core:cluster1:2000_2000:vdd:1 2000_2000_M2 0 0.001 static_value=4e-3

* Surrounding nodes in cluster
i_core:cluster1:1000_2000:vdd:0 1000_2000_M1 0 0.001 static_value=3e-3
i_core:cluster1:2000_1000:vdd:0 2000_1000_M1 0 0.001 static_value=3e-3
i_core:cluster1:3000_2000:vdd:0 3000_2000_M1 0 0.001 static_value=2.5e-3
i_core:cluster1:2000_3000:vdd:0 2000_3000_M1 0 0.001 static_value=2.5e-3

* ============================================================================
* Medium current load cluster (center region)
* ============================================================================

* Center load at (3000, 3000)
i_core:cluster2:3000_3000:vdd:0 3000_3000_M1 0 0.001 static_value=3e-3
i_core:cluster2:3000_3000:vdd:1 3000_3000_M2 0 0.001 static_value=2e-3

* Nearby nodes
i_core:cluster2:3000_4000:vdd:0 3000_4000_M1 0 0.001 static_value=2e-3
i_core:cluster2:4000_3000:vdd:0 4000_3000_M1 0 0.001 static_value=2e-3

* ============================================================================
* Light current load (periphery) - Simulates I/O and lighter logic
* ============================================================================

* Top-right corner region
i_io:corner1:5000_5000:vdd:0 5000_5000_M1 0 0.001 static_value=1e-3
i_io:corner1:4000_5000:vdd:0 4000_5000_M1 0 0.001 static_value=1e-3
i_io:corner1:5000_4000:vdd:0 5000_4000_M1 0 0.001 static_value=1e-3

* Bottom-right edge
i_io:edge1:5000_2000:vdd:0 5000_2000_M1 0 0.001 static_value=1.5e-3
i_io:edge1:4000_1000:vdd:0 4000_1000_M1 0 0.001 static_value=1.5e-3

* Top-left edge
i_io:edge2:1000_5000:vdd:0 1000_5000_M1 0 0.001 static_value=1.5e-3
i_io:edge2:1000_4000:vdd:0 1000_4000_M1 0 0.001 static_value=1.5e-3

* ============================================================================
* Total current distribution:
* - High load cluster: ~20 mA (6 sources)
* - Medium load cluster: ~9 mA (4 sources)
* - Light load periphery: ~9 mA (7 sources)
* - Total: ~38 mA across 17 current sources
* ============================================================================
