// Behavioral instruction-level NPU for reference simulation, HWC byte memory.
// NOT the paper's parallel datapath, DDR controller, PCIe, or hardware timing.
// Opcodes 0 conv33,1 conv11,2 Add,3 Cat,4 SPPF,5 Upsample,6 Focus.
module npu #(parameter MEM=32768)(input logic clk,rst,start,
 input integer op,h,w,ci,co,stride,pad,src_a,src_b,dst,weight_base,
 input integer za,zb,zo,ma,mb,mo, input logic relu,
 output logic busy,done,output integer cycles);
 logic [7:0] memory[0:MEM-1];
 logic signed [7:0] weights[0:MEM-1];
 integer biases[0:255];
 integer index,oh,ow,oc,total,row,col,ch,kr,kc,ic,ir,ix,k,stage,group,fh,fw;
 longint signed acc,temp,p;
 integer q,maxval;
 function automatic integer rq(input longint signed value,input integer m,z);
  longint signed prod,rounded;
  begin prod=value*m;
   if(prod>=0) rounded=(prod+32768)>>>16;
   else rounded=-(((-prod)+32768)>>>16);
   rounded=rounded+z;
   if(rounded<0) rq=0; else if(rounded>255) rq=255; else rq=rounded;
  end
 endfunction
 always @(posedge clk) begin
  if(rst) begin busy<=0; done<=0; cycles<=0; index<=0; end
  else begin
   done<=0;
   if(start && !busy) begin
    case(op)
     0: begin oh=(h+2*pad-3)/stride+1; ow=(w+2*pad-3)/stride+1; oc=co; end
     1: begin oh=(h+2*pad-1)/stride+1; ow=(w+2*pad-1)/stride+1; oc=co; end
     3: begin oh=h; ow=w; oc=ci+co; end
     4: begin oh=h; ow=w; oc=4*ci; end
     5: begin oh=2*h; ow=2*w; oc=ci; end
     6: begin oh=h/2; ow=w/2; oc=4*ci; end
     default: begin oh=h; ow=w; oc=ci; end
    endcase
    total=oh*ow*oc; index<=0; cycles<=0; busy<=1;
   end else if(busy) begin
    row=index/(ow*oc); col=(index/oc)%ow; ch=index%oc; acc=0; q=0;
    case(op)
     0,1: begin
      k=(op==0)?3:1;
      for(ic=0;ic<ci;ic=ic+1) for(kr=0;kr<k;kr=kr+1) for(kc=0;kc<k;kc=kc+1) begin
       ir=row*stride+kr-pad; ix=col*stride+kc-pad;
       if(ir>=0 && ir<h && ix>=0 && ix<w) begin
        temp=$signed({1'b0,memory[src_a+(ir*w+ix)*ci+ic]})-za;
        acc=acc+temp*$signed(weights[weight_base+((ch*ci+ic)*k+kr)*k+kc]);
       end
      end
      acc=acc+biases[ch]; if(relu && acc<0) acc=0;
      q=rq(acc,mo,zo);
     end
     2: begin
      acc=($signed({1'b0,memory[src_a+index]})-za)*longint'(ma)+($signed({1'b0,memory[src_b+index]})-zb)*longint'(mb);
      q=rq(acc,1,zo);
     end
     3: begin
      if(ch<ci) q=rq($signed({1'b0,memory[src_a+(row*w+col)*ci+ch]})-za,ma,zo);
      else q=rq($signed({1'b0,memory[src_b+(row*w+col)*co+ch-ci]})-zb,mb,zo);
     end
     4: begin
      stage=ch/ci; ic=ch%ci; k=4*stage+1; maxval=0;
      for(kr=0;kr<k;kr=kr+1) for(kc=0;kc<k;kc=kc+1) begin
       ir=row+kr-k/2; ix=col+kc-k/2;
       if(ir>=0 && ir<h && ix>=0 && ix<w && memory[src_a+(ir*w+ix)*ci+ic]>maxval) maxval=memory[src_a+(ir*w+ix)*ci+ic];
      end
      q=maxval;
     end
     5: q=memory[src_a+((row/2)*w+col/2)*ci+ch];
     6: begin
      group=ch/ci; ir=row*2+group%2; ix=col*2+group/2;
      q=memory[src_a+(ir*w+ix)*ci+ch%ci];
     end
     default: $fatal(1,"Invalid opcode");
    endcase
    memory[dst+index]<=q[7:0]; index<=index+1; cycles<=cycles+1;
    if(index==total-1) begin busy<=0; done<=1; end
   end
  end
 end
endmodule
