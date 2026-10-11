module tb_buffer;
 logic clk=0,rst=1,in_valid=0,out_ready=0; wire in_ready,out_valid;
 logic [7:0] in_data=0; wire [7:0] out_data;
 integer sent=0,received=0,tick=0,phase=0; logic stalled=0; logic [7:0] held;
 ping_pong_buffer #(.DEPTH(8)) dut(.*);
 always #5 clk=~clk;
 initial begin repeat(3) @(negedge clk); rst=0; end
 always @(negedge clk) if(!rst) begin
  in_valid=(tick%5!=0 && sent<160); in_data=sent%256;
  out_ready=(tick%7<4); tick=tick+1;
  if(tick>2000) $fatal(1,"timeout");
 end
 always @(posedge clk) if(!rst) begin
  if(stalled && (!out_valid || out_data!==held)) $fatal(1,"unstable stalled output");
  stalled=out_valid && !out_ready; held=out_data;
  if(in_valid && in_ready) sent=sent+1;
  if(out_valid && out_ready) begin
   if(out_data!==(received%256)) $fatal(1,"FIFO mismatch at %0d",received);
   received=received+1;
  end
  if(received==160) begin $display("PASS buffer sent=%0d received=%0d cycles=%0d",sent,received,tick); $finish; end
 end
endmodule
