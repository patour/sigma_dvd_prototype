* Multi-tile PDN netlist for parallel parsing tests
* 3x3 tile grid with VDD and VSS nets
* Each tile has ~100 nodes across 3 layers

.partition_info 3 3

* Power net voltage definitions
.parameter VDD 0.75
.parameter VSS 0.0

* Include tile netlists
.include tile_0_0.ckt
.include tile_0_1.ckt
.include tile_0_2.ckt
.include tile_1_0.ckt
.include tile_1_1.ckt
.include tile_1_2.ckt
.include tile_2_0.ckt
.include tile_2_1.ckt
.include tile_2_2.ckt

* Include instance models
.include instanceModels_0_0.sp
.include instanceModels_0_1.sp
.include instanceModels_0_2.sp
.include instanceModels_1_0.sp
.include instanceModels_1_1.sp
.include instanceModels_1_2.sp
.include instanceModels_2_0.sp
.include instanceModels_2_1.sp
.include instanceModels_2_2.sp

* Include package model
.include package.ckt

.end
