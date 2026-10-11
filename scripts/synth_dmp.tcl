set root [file normalize [file join [file dirname [info script]] ..]]
read_verilog -sv [file join $root rtl dual_multiply_packing dmp.sv]
set part xc7z100ffg900-2
if {$argc > 0} {set part [lindex $argv 0]}
synth_design -top dmp -part $part -mode out_of_context
create_clock -period 5.0 [get_ports clk]
report_utilization -file [file join $root outputs rtl dmp_utilization.rpt]
report_timing_summary -file [file join $root outputs rtl dmp_synth_timing.rpt]
set dsps [llength [get_cells -hier -filter {REF_NAME == DSP48E1}]]
set stream [open [file join $root outputs rtl dmp_synthesis.json] w]
puts $stream "{\"result_kind\":\"MEASURED\",\"part\":\"$part\",\"scope\":\"Vivado synthesis only, not placement or hardware\",\"dsp48e1\":$dsps}"
close $stream
if {$dsps != 1} {error "Expected one DSP48E1, found $dsps"}
