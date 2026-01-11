* Small PDN netlist for VDD_XLV
* Generated for faster development/testing
* ~6400 die nodes across 5 layers (M1, M2, M3, M4, M5)

* Tile grid configuration
.partition_info 1 1

* Die area (physical dimensions in coordinate units)  
.die_area 0 0 104000 104000

* Power net parameters
.parameter VDD_XLV 0.75
.parameter VSS 0

* Include tile netlist (die resistor mesh)
.include ./tile_0_0.ckt

* Include package model (voltage sources and bump connections)
.include ./package.ckt

* Include current sources (load simulation)
.include ./instanceModels_0_0.sp

.end
