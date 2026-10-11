// Reconstructed 16-lane Q16 MQDA, with whole-pipeline stall on output backpressure.
// CAT=0: affine Add; CAT=1: single source beat aligns using selected domain.
// Four register stages; ratios unsigned24, zero points unsigned8, half-away round.
module mqda_align #(parameter LANES=16,CAT=0)
(input logic clk,rst,in_valid,output logic in_ready,
 input logic [LANES*8-1:0] qa,qb,input logic [7:0] za,zb,zo,
 input logic [23:0] ma,mb,input logic source_b,
 output logic out_valid,input logic out_ready,output logic [LANES*8-1:0] qout);
 logic [3:0] valid_pipe;
 logic [7:0] zpipe[0:1];
 logic signed [33:0] pa[0:LANES-1],pb[0:LANES-1];
 logic signed [34:0] total[0:LANES-1],rounded[0:LANES-1];
 logic advance;integer lane;
 function automatic signed [34:0] round16(input logic signed [34:0] value);
  if(value>=0) round16=(value+35'sd32768)>>>16;
  else round16=-(((-value)+35'sd32768)>>>16);
 endfunction
 always_comb begin
  out_valid=valid_pipe[3];advance=!out_valid || out_ready;in_ready=advance;
 end
 always @(posedge clk) begin
  if(rst) begin valid_pipe<=0;qout<=0;zpipe[0]<=0;zpipe[1]<=0;end
  else if(advance) begin
   valid_pipe<={valid_pipe[2:0],in_valid};zpipe[0]<=zo;zpipe[1]<=zpipe[0];
   for(lane=0;lane<LANES;lane=lane+1) begin
    if(CAT) begin
     pa[lane]<=($signed({1'b0,qa[lane*8+:8]})-$signed({1'b0,(source_b?zb:za)}))*$signed({1'b0,(source_b?mb:ma)});
     pb[lane]<=0;
    end else begin
     pa[lane]<=($signed({1'b0,qa[lane*8+:8]})-$signed({1'b0,za}))*$signed({1'b0,ma});
     pb[lane]<=($signed({1'b0,qb[lane*8+:8]})-$signed({1'b0,zb}))*$signed({1'b0,mb});
    end
    total[lane]<=pa[lane]+pb[lane];
    rounded[lane]<=round16(total[lane])+$signed({1'b0,zpipe[1]});
    if(rounded[lane]<0) qout[lane*8+:8]<=0;
    else if(rounded[lane]>255) qout[lane*8+:8]<=255;
    else qout[lane*8+:8]<=rounded[lane][7:0];
   end
  end
 end
endmodule
