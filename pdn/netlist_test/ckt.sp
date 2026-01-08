* PDN Test Netlist - Minimal test case for parser, solver, and plotter validation
* Created: January 7, 2026
* Description: Single 1x1 tile with 5x5 grid on two layers (M1, M2) with package and current sources

* Tile grid configuration
.partition_info 1 1

* Die area (physical dimensions in coordinate units)
.die_area 0 0 6000 6000

* Power net parameters
.parameter VDD 0.75
.parameter VSS 0

* Voltage sources (ideal supplies)
vVDD vVDD 0 VDD
vVSS vVSS 0 VSS

* Include tile netlist (die resistor/capacitor mesh)
.include ./tile_0_0.ckt

* Include package model (bump connections and probe network)
.include ./package.ckt

* Include current sources (load simulation)
.include ./instanceModels_0_0.sp

.end
