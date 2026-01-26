* Current Sources - Simulated load distribution
* Format: I<name> <node+> <node-> <dc_value> [static_value=<value>] [pulse(...)] [pwl(...)]
* Current flows from node+ to node- (sink convention)
*
* Waveform formats supported:
*   DC only:     I... node+ node- <value>
*   Static:      I... node+ node- <dc> static_value=<static>
*   Pulse:       I... node+ node- <dc> pulse(v1 v2 delay rise fall width period)
*   PWL:         I... node+ node- <dc> pwl(t1 v1 t2 v2 ...) pwl_period=<T> pwl_delay=<D>
*   Combined:    I... node+ node- <dc> static_value=<static> pulse(...) pwl(...)

* ============================================================================
* High current load cluster (bottom-left region) - Simulates dense logic
* Dynamic waveforms represent switching activity
* ============================================================================

* Core cluster at (2000, 2000) - Heavy load with dynamic activity
* DC=1mA, Static=5mA, plus 2mA pulse at 100MHz (10ns period)
i_core:cluster1:2000_2000:vdd:0 2000_2000_M1 0 0.001 static_value=5e-3 pulse(0 2e-3 0 0.1n 0.1n 4.9n 10n)
* DC=1mA, Static=4mA, plus PWL ramp pattern (200MHz, 5ns period)
i_core:cluster1:2000_2000:vdd:1 2000_2000_M2 0 0.001 static_value=4e-3 pwl(0 0 1n 1.5e-3 2.5n 1.5e-3 3.5n 0) pwl_period=5n

* Surrounding nodes in cluster - with pulse waveforms
* DC=1mA, Static=3mA, plus 1.5mA pulse at 50MHz
i_core:cluster1:1000_2000:vdd:0 1000_2000_M1 0 0.001 static_value=3e-3 pulse(0 1.5e-3 1n 0.2n 0.2n 9n 20n)
* DC=1mA, Static=3mA, plus 1mA pulse at 100MHz
i_core:cluster1:2000_1000:vdd:0 2000_1000_M1 0 0.001 static_value=3e-3 pulse(0 1e-3 0 0.1n 0.1n 4.9n 10n)
* DC=1mA, Static=2.5mA (no dynamic - DC/static only)
i_core:cluster1:3000_2000:vdd:0 3000_2000_M1 0 0.001 static_value=2.5e-3
* DC=1mA, Static=2.5mA, plus triangular PWL at 200MHz
i_core:cluster1:2000_3000:vdd:0 2000_3000_M1 0 0.001 static_value=2.5e-3 pwl(0 0 2.5n 2e-3 5n 0) pwl_period=5n

* ============================================================================
* Medium current load cluster (center region)
* Mix of DC-only and dynamic loads
* ============================================================================

* Center load at (3000, 3000) - with pulse activity
* DC=1mA, Static=3mA, plus 2.5mA burst pulse
i_core:cluster2:3000_3000:vdd:0 3000_3000_M1 0 0.001 static_value=3e-3 pulse(0 2.5e-3 5n 0.5n 0.5n 10n 40n)
* DC=1mA, Static=2mA (DC/static only)
i_core:cluster2:3000_3000:vdd:1 3000_3000_M2 0 0.001 static_value=2e-3

* Nearby nodes - PWL waveforms
* DC=1mA, Static=2mA, plus sawtooth PWL at 125MHz
i_core:cluster2:3000_4000:vdd:0 3000_4000_M1 0 0.001 static_value=2e-3 pwl(0 0 7n 1.8e-3 8n 0) pwl_period=8n
* DC=1mA, Static=2mA (DC/static only)
i_core:cluster2:4000_3000:vdd:0 4000_3000_M1 0 0.001 static_value=2e-3

* ============================================================================
* Light current load (periphery) - Simulates I/O and lighter logic
* Slower switching I/O patterns
* ============================================================================

* Top-right corner region - I/O with slower pulses
* DC=1mA, Static=1mA, plus 0.8mA pulse at 10MHz
i_io:corner1:5000_5000:vdd:0 5000_5000_M1 0 0.001 static_value=1e-3 pulse(0 0.8e-3 0 1n 1n 48n 100n)
* DC=1mA, Static=1mA (DC/static only)
i_io:corner1:4000_5000:vdd:0 4000_5000_M1 0 0.001 static_value=1e-3
* DC=1mA, Static=1mA, plus slow PWL ramp
i_io:corner1:5000_4000:vdd:0 5000_4000_M1 0 0.001 static_value=1e-3 pwl(0 0 25n 0.6e-3 50n 0.6e-3 75n 0) pwl_period=100n

* Bottom-right edge - mixed waveforms
* DC=1mA, Static=1.5mA, plus 1mA pulse at 25MHz
i_io:edge1:5000_2000:vdd:0 5000_2000_M1 0 0.001 static_value=1.5e-3 pulse(0 1e-3 2n 0.5n 0.5n 19n 40n)
* DC=1mA, Static=1.5mA (DC/static only)
i_io:edge1:4000_1000:vdd:0 4000_1000_M1 0 0.001 static_value=1.5e-3

* Top-left edge - PWL dominated
* DC=1mA, Static=1.5mA, plus complex PWL pattern
i_io:edge2:1000_5000:vdd:0 1000_5000_M1 0 0.001 static_value=1.5e-3 pwl(0 0 5n 0.5e-3 15n 1.2e-3 20n 0.5e-3 30n 0) pwl_period=50n pwl_delay=2n
* DC=1mA, Static=1.5mA (DC/static only)
i_io:edge2:1000_4000:vdd:0 1000_4000_M1 0 0.001 static_value=1.5e-3

* ============================================================================
* Total current distribution (DC + static_value):
* - High load cluster: ~20 mA (6 sources) + dynamic waveforms
* - Medium load cluster: ~9 mA (4 sources) + dynamic waveforms  
* - Light load periphery: ~9 mA (7 sources) + dynamic waveforms
* - Total static: ~38 mA across 17 current sources
* - Dynamic: 11 sources have pulse/pwl waveforms, 6 sources DC/static only
* ============================================================================
