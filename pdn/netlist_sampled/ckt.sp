* Sampled PDN netlist for VDD_XLV
* Generated from netlist_minion/pdn_graph.pkl
* 3x3 tile grid

.partition_info 3 3

* Power net voltage definitions
.parameter VDD_XLV 0.66
.parameter VSS 0.0

* Include tile netlists
.include tile_0_0.ckt
.include tile_1_0.ckt
.include tile_2_0.ckt
.include tile_0_1.ckt
.include tile_1_1.ckt
.include tile_2_1.ckt
.include tile_0_2.ckt
.include tile_1_2.ckt
.include tile_2_2.ckt

* Include instance models
.include instanceModels_0_0.sp
.include instanceModels_1_0.sp
.include instanceModels_2_0.sp
.include instanceModels_0_1.sp
.include instanceModels_1_1.sp
.include instanceModels_2_1.sp
.include instanceModels_0_2.sp
.include instanceModels_1_2.sp
.include instanceModels_2_2.sp

* Include package model
.include package.ckt

.end
