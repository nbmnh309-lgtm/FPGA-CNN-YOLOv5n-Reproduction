"""Python-generated golden vectors and real Vivado XSim execution."""
import json
import subprocess
from pathlib import Path
import numpy as np
from reproduction.reference import conv, focus, upsample, sppf, requant_q16, dmp

ROOT = Path(__file__).resolve().parents[1]


def command(args, cwd, timeout=180):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, errors='replace', timeout=timeout)
    with (cwd/'commands.log').open('a',encoding='utf-8') as f:
        f.write(' '.join(map(str,args))+'\n'+result.stdout+result.stderr+'\n')
    if result.returncode or 'FATAL_ERROR' in result.stdout or 'Fatal:' in result.stdout:
        raise RuntimeError(f'RTL command failed: {args}; see {cwd}/commands.log')
    return result.stdout


def tools():
    paths=sorted(Path('C:/Xilinx/Vivado').glob('*/bin/xvlog.bat'))
    if not paths: raise FileNotFoundError('Vivado XSim not found')
    return paths[-1].parent


def run_snapshot(name, sources, build):
    bindir=tools()
    command([str(bindir/'xvlog.bat'),'-sv',*map(str,sources)],build)
    command([str(bindir/'xelab.bat'),name,'-s',name+'_snapshot'],build)
    output=command([str(bindir/'xsim.bat'),name+'_snapshot','-runall'],build,300)
    if 'PASS ' not in output or 'FAIL' in output or 'Fatal' in output:
        raise RuntimeError(f'No verified PASS from {name}: {output[-2000:]}')
    return output


def generate_npu(build):
    rng=np.random.default_rng(20261011); body=[]; cases=[]
    def case(op,x,expected,b=None,weights=None,bias=None,**params):
        num=len(cases); a=0; ba=2048; dst=8192
        values={'op':op,'h':x.shape[0],'w':x.shape[1],'ci':x.shape[2], 'co':expected.shape[2], 'stride':1,'pad':0,'src_a':a,'src_b':ba,'dst':dst,'weight_base':0,'za':0,'zb':0,'zo':0,'ma':65536,'mb':65536,'mo':65536,'relu':0}
        values.update(params)
        def memfile(label,array,width):
            filename=f'case{num}_{label}.hex'; flat=np.asarray(array).reshape(-1)
            (build/filename).write_text('\n'.join(f'{int(v)&((1<<width)-1):0{width//4}x}' for v in flat)+'\n')
            return filename
        body.append(f'$readmemh("{memfile("a",x,8)}",dut.memory,{a},{a+x.size-1});')
        if b is not None: body.append(f'$readmemh("{memfile("b",b,8)}",dut.memory,{ba},{ba+b.size-1});')
        if weights is not None: body.append(f'$readmemh("{memfile("w",weights,8)}",dut.weights,0,{weights.size-1});')
        body.append('for(i=0;i<256;i=i+1) dut.biases[i]=0;')
        if bias is not None:body.extend(f'dut.biases[{i}]={int(value)};' for i,value in enumerate(bias))
        body.append(f'$readmemh("{memfile("expected",expected,8)}",expected,0,{expected.size-1});')
        body.extend(f'{key}={value};' for key,value in values.items())
        body.append('@(negedge clk); start=1; @(negedge clk); start=0; wait(done); #1;')
        body.append(f'if(cycles!={expected.size}) $fatal(1,"cycle mismatch case {num}");')
        body.append(f'for(i=0;i<{expected.size};i=i+1) if(dut.memory[{dst}+i]!==expected[i]) $fatal(1,"case {num} op {op} index %0d actual %0d expected %0d",i,dut.memory[{dst}+i],expected[i]);')
        body.append('@(negedge clk);')
        cases.append({'case':num,'op':op,'input_shape':list(x.shape),'output_shape':list(expected.shape),'cycles':int(expected.size)})
    for h,w,ci in [(6,8,3),(4,4,1),(8,6,4)]:
        x=rng.integers(0,256,(h,w,ci),dtype=np.int64); b=rng.integers(0,256,x.shape)
        case(6,x,focus(x)); case(5,x,upsample(x)); case(4,x,sppf(x))
        add=np.clip(np.where(((x-91)*32768+(b-137)*98304)>=0, (((x-91)*32768+(b-137)*98304)+32768)>>16, -((-((x-91)*32768+(b-137)*98304)+32768)>>16))+100,0,255)
        case(2,x,add,b=b,za=91,zb=137,zo=100,ma=32768,mb=98304)
        cat=np.concatenate((requant_q16(x-91,32768,100),requant_q16(b-137,98304,100)),axis=-1)
        case(3,x,cat,b=b,co=ci,za=91,zb=137,zo=100,ma=32768,mb=98304)
        for k in (1,3):
            for stride in (1,2):
                weights=rng.integers(-128,128,(3,ci,k,k)); bias=rng.integers(-1000,1000,3);accum=conv(x,weights,bias=bias,stride=stride,padding=k//2,zero=128)
                relu=(stride==2)
                if relu:accum=np.maximum(accum,0)
                case(0 if k==3 else 1,x,requant_q16(accum,256,100),weights=weights,bias=bias,co=3,stride=stride,pad=k//2,za=128,zo=100,mo=256,relu=int(relu))
    # A real chained sequence reuses results in DUT memory without host reload.
    x=np.arange(6*6*2).reshape(6,6,2); expected=upsample(focus(x))
    fn=build/'chain.hex'; fn.write_text('\n'.join(f'{v:02x}' for v in x.flat)+'\n')
    en=build/'chain_expected.hex'; en.write_text('\n'.join(f'{v:02x}' for v in expected.flat)+'\n')
    body.extend(['$readmemh("chain.hex",dut.memory,0,71);', 'op=6;h=6;w=6;ci=2;src_a=0;dst=2048;', '@(negedge clk);start=1;@(negedge clk);start=0;wait(done);@(negedge clk);', 'op=5;h=3;w=3;ci=8;src_a=2048;dst=8192;', '@(negedge clk);start=1;@(negedge clk);start=0;wait(done);#1;', '$readmemh("chain_expected.hex",expected,0,287);', 'for(i=0;i<288;i=i+1) if(dut.memory[8192+i]!==expected[i]) $fatal(1,"integration mismatch %0d",i);'])
    body.extend(['@(negedge clk);start=1;@(negedge clk);start=0;repeat(5) @(negedge clk);rst=1;@(negedge clk);#1;', 'if(busy || done) $fatal(1,"reset did not abort NPU");rst=0;', '@(negedge clk);start=1;@(negedge clk);start=0;wait(done);#1;', 'if(cycles!=288) $fatal(1,"restart cycle mismatch");for(i=0;i<288;i=i+1) if(dut.memory[8192+i]!==expected[i]) $fatal(1,"reset restart output mismatch");'])
    tb='''module tb_npu;
logic clk=0,rst=1,start=0,relu=0; wire busy,done; wire integer cycles;
integer op,h,w,ci,co,stride,pad,src_a,src_b,dst,weight_base,za,zb,zo,ma,mb,mo,i;
logic [7:0] expected[0:8191]; npu dut(.*);
always #5 clk=~clk;
initial begin #2000000; $fatal(1,"timeout"); end
initial begin
for(i=0;i<256;i=i+1) dut.biases[i]=0;
repeat(3) @(negedge clk); rst=0;
'''+ '\n'.join(body)+f'\n$display("PASS NPU {len(cases)} operator cases + chained integration");$finish;end\nendmodule\n'
    (build/'tb_npu.sv').write_text(tb)
    return cases


def generate_dmp(build):
    rng=np.random.default_rng(22)
    a,b=np.meshgrid(np.arange(-128,128),np.arange(-128,128),indexing='ij')
    edge=np.array([-128,-127,-1,0,1,127,128,255]); aa=np.tile(a.ravel(),len(edge)); bb=np.tile(b.ravel(),len(edge)); cc=np.repeat(edge,a.size)
    aa=np.r_[aa,rng.integers(-128,128,20000)]; bb=np.r_[bb,rng.integers(-128,128,20000)]; cc=np.r_[cc,rng.integers(-128,256,20000)]
    hi,lo=dmp(aa,bb,cc)
    words=((aa&255)<<49)|((bb&255)<<41)|((cc&511)<<32)|((hi&65535)<<16)|(lo&65535)
    (build/'dmp_vectors.hex').write_text('\n'.join(f'{v:016x}' for v in words)+'\n')
    n=len(words)
    tb=f'''module tb_dmp;
logic clk=0,rst=1,valid_in=0; logic signed [7:0] a,b; logic signed [8:0] c;
wire valid_out;wire signed [15:0] hi,lo;logic [63:0] vectors[0:{n-1}];
integer sent=0,checked=0,tick=0;logic [1:0] history=0;dmp dut(.*);always #5 clk=~clk;
initial begin $readmemh("dmp_vectors.hex",vectors);repeat(3) @(negedge clk);rst=0;end
always @(negedge clk) if(!rst) begin
 valid_in=(sent<{n} && tick%7!=0);tick=tick+1;
 if(valid_in) begin a=vectors[sent][56:49];b=vectors[sent][48:41];c=vectors[sent][40:32];sent=sent+1;end
 if(tick>{n*2}) $fatal(1,"timeout");
end
always @(posedge clk) begin
 if(rst) history=0;else history={{history[0],valid_in}};
 #1;
 if(valid_out!==history[1]) $fatal(1,"DMP valid latency/reset mismatch");
 if(!rst && valid_out) begin
  if(hi!==vectors[checked][31:16] || lo!==vectors[checked][15:0]) $fatal(1,"DMP mismatch %0d",checked);
  checked=checked+1;
  if(checked=={n}) begin $display("PASS DMP %0d vectors",checked);$finish;end
 end
end
endmodule\n'''
    (build/'tb_dmp.sv').write_text(tb);return n


def generate_dot(build,k,cout):
    rng=np.random.default_rng(42+k);cin=16;n=k*k*cin;frames=100
    x=rng.integers(0,256,(frames,n));w=rng.integers(-128,128,(frames,cout,n))
    x[0:2]=255;w[0]=-128;w[1]=127;x[2]=0
    expected=(w*x[:,None]).sum(axis=2)
    def word(values,width):return sum((int(v)&((1<<width)-1))<<(i*width) for i,v in enumerate(values))
    prefix=f'dot{k}'
    for label,values,width in [('x',x,8),('w',w.reshape(frames,-1),8),('expected',expected,24)]:
        (build/f'{prefix}_{label}.hex').write_text('\n'.join(f'{word(row,width):0{len(row)*width//4}x}' for row in values)+'\n')
    tb=f'''module tb_dot{k};
logic clk=0,rst=1,valid_in=0;wire valid_out;
logic [{n*8-1}:0] activations;logic [{cout*n*8-1}:0] weights;wire [{cout*24-1}:0] sums;
logic [{n*8-1}:0] xs[0:99];logic [{cout*n*8-1}:0] ws[0:99];logic [{cout*24-1}:0] ys[0:99];
integer sent=0,checked=0,tick=0;logic [2:0] history=0;
conv_dot #(.K({k}),.CIN(16),.COUT({cout})) dut(.*);always #5 clk=~clk;
initial begin $readmemh("{prefix}_x.hex",xs);$readmemh("{prefix}_w.hex",ws);$readmemh("{prefix}_expected.hex",ys);repeat(3) @(negedge clk);rst=0;end
always @(negedge clk) if(!rst) begin
 valid_in=(sent<100 && tick%7!=0);tick=tick+1;
 if(valid_in) begin activations=xs[sent];weights=ws[sent];sent=sent+1;end
 if(tick>300) $fatal(1,"timeout");
end
always @(posedge clk) begin
 if(rst) history=0;else history={{history[1:0],valid_in}};
 #1;
 if(!rst && valid_out!==history[2]) $fatal(1,"dot valid latency");
 if(!rst && valid_out) begin
  if(sums!==ys[checked]) $fatal(1,"dot {k} mismatch %0d",checked);
  checked=checked+1;if(checked==100) begin $display("PASS Conv{k}x{k} DMP tile 100 frames");$finish;end
 end
end
endmodule\n'''
    (build/f'tb_dot{k}.sv').write_text(tb)


def generate_stream(build):
    rng=np.random.default_rng(101);a=rng.integers(-128,128,256);b=rng.integers(-128,128,256);c=rng.integers(-128,256,256);hi,lo=dmp(a,b,c)
    vectors=((a&255)<<49)|((b&255)<<41)|((c&511)<<32)|((hi&65535)<<16)|(lo&65535)
    (build/'stream.hex').write_text('\n'.join(f'{v:016x}' for v in vectors)+'\n')
    (build/'tb_stream.sv').write_text('''module tb_stream;
logic clk=0,rst=1,in_valid=0,out_ready=0;wire in_ready,out_valid;
logic [24:0] in_data;wire [24:0] out_data;logic [63:0] vectors[0:255];
wire result_valid;wire signed [15:0] hi,lo;integer sent=0,checked=0,tick=0;
ping_pong_buffer #(.WIDTH(25),.DEPTH(8)) bank(.*);
dmp mul(.clk(clk),.rst(rst),.valid_in(out_valid && out_ready),.a(out_data[24:17]),.b(out_data[16:9]),.c(out_data[8:0]),.valid_out(result_valid),.hi(hi),.lo(lo));
always #5 clk=~clk;
initial begin $readmemh("stream.hex",vectors);repeat(3) @(negedge clk);rst=0;end
always @(negedge clk) if(!rst) begin
 in_valid=(sent<256 && tick%5!=0);in_data=(sent<256)?vectors[sent][56:32]:0;out_ready=(tick%7<3);tick=tick+1;
 if(tick>3000) $fatal(1,"timeout");
end
always @(posedge clk) begin
 if(!rst && in_valid && in_ready) sent=sent+1;
 #1;
 if(!rst && result_valid) begin
  if({hi,lo}!==vectors[checked][31:0]) $fatal(1,"buffer DMP integration mismatch %0d",checked);
  checked=checked+1;if(checked==256) begin $display("PASS buffer-to-DMP integrated stream 256 vectors");$finish;end
 end
end
endmodule\n''')


def generate_mqda(build,cat):
    rng=np.random.default_rng(45+cat);frames=1024;lanes=16;words=[]
    def packed(values):return sum(int(v)<<(8*i) for i,v in enumerate(values))
    for frame in range(frames):
        a=rng.integers(0,256,lanes);b=rng.integers(0,256,lanes);za,zb,zo=map(int,rng.integers(0,256,3));ma,mb=map(int,rng.integers(1,3*65536,2));source=frame%2
        if frame<16:a[:]=[0,1,127,128,254,255,0,255]*2;b[:]=a[::-1];za=128;zb=255;ma=32768;mb=98304
        if 16<=frame<32:ma=16777215;mb=16777215
        raw=(a-(zb if source else za))*(mb if source else ma) if cat else (a-za)*ma+(b-zb)*mb
        expected=requant_q16(raw,1,zo)
        words.append((source<<456)|(ma<<432)|(mb<<408)|(za<<400)|(zb<<392)|(zo<<384)|(packed(a)<<256)|(packed(b)<<128)|packed(expected))
    (build/f'mqda{cat}.hex').write_text('\n'.join(f'{v:0115x}' for v in words)+'\n')
    tb=f'''module tb_mqda{cat};
logic clk=0,rst=1,in_valid=0,out_ready=0,source_b=0;wire in_ready,out_valid;
logic [127:0] qa,qb;logic [7:0] za,zb,zo;logic [23:0] ma,mb;wire [127:0] qout;
logic [456:0] vectors[0:1023];integer sent=0,checked=0,tick=0;logic stalled=0;logic [127:0] held;
mqda_align #(.CAT({cat})) dut(.*);always #5 clk=~clk;
initial begin $readmemh("mqda{cat}.hex",vectors);repeat(3) @(negedge clk);rst=0;end
always @(negedge clk) if(!rst) begin
 in_valid=(sent<1024 && tick%5!=0);out_ready=(tick%11<7);tick=tick+1;
 if(sent<1024) begin source_b=vectors[sent][456];ma=vectors[sent][455:432];mb=vectors[sent][431:408];za=vectors[sent][407:400];zb=vectors[sent][399:392];zo=vectors[sent][391:384];qa=vectors[sent][383:256];qb=vectors[sent][255:128];end
 if(tick>6000) $fatal(1,"timeout");
end
always @(posedge clk) if(!rst) begin
 if(stalled && (!out_valid || qout!==held)) $fatal(1,"MQDA unstable stalled output");
 stalled=out_valid && !out_ready;held=qout;
 if(in_valid && in_ready) sent=sent+1;
 if(out_valid && out_ready) begin
  if(qout!==vectors[checked][127:0]) $fatal(1,"MQDA cat={cat} mismatch %0d",checked);
  checked=checked+1;if(checked==1024) begin $display("PASS MQDA CAT={cat} 1024 x16 lanes");$finish;end
 end
end
endmodule\n'''
    (build/f'tb_mqda{cat}.sv').write_text(tb)


def main():
    build=ROOT/'outputs/rtl';build.mkdir(parents=True,exist_ok=True)
    cases=generate_npu(build);vectors=generate_dmp(build)
    generate_dot(build,3,8);generate_dot(build,1,16);generate_stream(build)
    generate_mqda(build,0);generate_mqda(build,1)
    result={'result_kind':'SIMULATED','simulator':str(tools()),'dmp_vectors':vectors,'operator_cases':cases,'tests':{}}
    jobs=[('tb_dmp',[ROOT/'rtl/dual_multiply_packing/dmp.sv',build/'tb_dmp.sv']),('tb_buffer',[ROOT/'rtl/ping_pong_buffer/buffer.sv',ROOT/'testbench/unit/tb_buffer.sv']),('tb_npu',[ROOT/'rtl/top/npu.sv',build/'tb_npu.sv'])]
    jobs.extend((f'tb_dot{k}',[ROOT/'rtl/dual_multiply_packing/dmp.sv',ROOT/'rtl/common/conv_dot.sv',build/f'tb_dot{k}.sv']) for k in [3,1])
    jobs.append(('tb_stream',[ROOT/'rtl/dual_multiply_packing/dmp.sv',ROOT/'rtl/ping_pong_buffer/buffer.sv',build/'tb_stream.sv']))
    jobs.extend((f'tb_mqda{cat}',[ROOT/'rtl/mqda/align.sv',build/f'tb_mqda{cat}.sv']) for cat in [0,1])
    for name,sources in jobs:
        output=run_snapshot(name,sources,build);result['tests'][name]={'status':'PASS','output':output[-4000:]}
    (build/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='tests'}))


if __name__=='__main__': main()
