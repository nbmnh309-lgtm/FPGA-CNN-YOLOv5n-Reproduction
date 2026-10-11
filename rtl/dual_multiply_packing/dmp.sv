// Reconstructed signed weights / shared signed 9-bit activation, two-stage pipeline.
// A25=a*65536+b. Borrow from low signed product is corrected in the high lane.
module dmp(input logic clk, rst, valid_in,
 input logic signed [7:0] a,b, input logic signed [8:0] c,
 output logic valid_out, output logic signed [15:0] hi,lo);
 logic signed [24:0] packed_operand;
 (* use_dsp="yes" *) logic signed [33:0] product;
 logic valid_product;
 always_comb packed_operand = ($signed(a) * 25'sd65536) + $signed(b);
 always_ff @(posedge clk) begin
  if(rst) begin product<=0; valid_product<=0; valid_out<=0; hi<=0; lo<=0; end
  else begin
   valid_product<=valid_in;
   if(valid_in) product<=packed_operand*c;
   valid_out<=valid_product;
   if(valid_product) begin
    lo<=product[15:0];
    hi<=$signed(product[31:16]) + (product[15] ? 16'sd1 : 16'sd0);
   end
  end
 end
endmodule
