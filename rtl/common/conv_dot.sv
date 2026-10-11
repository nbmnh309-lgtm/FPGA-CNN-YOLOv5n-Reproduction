// Synthesizable convolution arithmetic tile. Window/weight delivery is external.
// K=3,CIN=16,COUT=8: 576 DMPs. K=1,CIN=16,COUT=16:128 DMPs.
// Uint8 activation, signed8 weights, signed24 partial sum. No requantization.
module conv_dot #(parameter K=3,CIN=16,COUT=8)
(input logic clk,rst,valid_in,
 input logic [K*K*CIN*8-1:0] activations,
 input logic [COUT*K*K*CIN*8-1:0] weights,
 output logic valid_out,output logic [COUT*24-1:0] sums);
 localparam N=K*K*CIN;
 wire signed [15:0] products[0:COUT-1][0:N-1];
 wire valid_lane[0:COUT/2-1][0:N-1];
 genvar pair,element;
 generate for(pair=0;pair<COUT/2;pair=pair+1) begin:g_pair
  for(element=0;element<N;element=element+1) begin:g_element
   dmp mul(.clk(clk),.rst(rst),.valid_in(valid_in),
    .a(weights[((2*pair)*N+element)*8+:8]),
    .b(weights[((2*pair+1)*N+element)*8+:8]),
    .c($signed({1'b0,activations[element*8+:8]})),
    .valid_out(valid_lane[pair][element]),
    .hi(products[2*pair][element]),.lo(products[2*pair+1][element]));
  end
 end endgenerate
 integer o,i;logic signed [23:0] accum;
 always @(posedge clk) begin
  if(rst) begin valid_out<=0;sums<=0;end
  else begin
   valid_out<=valid_lane[0][0];
   if(valid_lane[0][0]) for(o=0;o<COUT;o=o+1) begin
    accum=0;
    for(i=0;i<N;i=i+1) accum=accum+products[o][i];
    sums[o*24+:24]<=accum;
   end
  end
 end
endmodule
