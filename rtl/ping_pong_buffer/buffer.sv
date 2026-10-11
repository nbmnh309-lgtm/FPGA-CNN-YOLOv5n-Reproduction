// Two complete-block banks, ready/valid backpressure and explicit bank ownership.
module ping_pong_buffer #(parameter DEPTH=8, WIDTH=8)
(input logic clk,rst,input logic in_valid,output logic in_ready,
 input logic [WIDTH-1:0] in_data,output logic out_valid,input logic out_ready,
 output logic [WIDTH-1:0] out_data);
 logic [WIDTH-1:0] mem[0:1][0:DEPTH-1];
 logic [1:0] full;
 integer wr_pos,rd_pos;
 logic wr_bank,rd_bank;
 always_comb begin
  in_ready=!full[wr_bank]; out_valid=full[rd_bank];
  out_data=mem[rd_bank][rd_pos];
 end
 always_ff @(posedge clk) begin
  if(rst) begin full<=0; wr_pos<=0; rd_pos<=0; wr_bank<=0; rd_bank<=0; end
  else begin
   if(in_valid && in_ready) begin
    mem[wr_bank][wr_pos]<=in_data;
    if(wr_pos==DEPTH-1) begin full[wr_bank]<=1; wr_pos<=0; wr_bank<=!wr_bank; end
    else wr_pos<=wr_pos+1;
   end
   if(out_valid && out_ready) begin
    if(rd_pos==DEPTH-1) begin full[rd_bank]<=0; rd_pos<=0; rd_bank<=!rd_bank; end
    else rd_pos<=rd_pos+1;
   end
  end
 end
endmodule
