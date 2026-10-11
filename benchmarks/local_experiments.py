"""CUDA FP32, full-network QDQ ablation, COCO128 sanity AP, x86 INT8 kernel.

No VOC checkpoint is provided. QDQ uses floating convolutions with quantized
values, NOT an INT8 GPU backend or FPGA throughput measurement.
"""
import csv
import hashlib
import json
import os
import sys
import time
import types
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
os.environ['YOLOV5_CONFIG_DIR']=str(ROOT/'outputs/yolov5_config')
# Keep project packages ahead of identically named upstream modules on Windows spawn.
sys.path.insert(1,str(ROOT/'models/yolov5'))
from models.experimental import attempt_load
from models.common import Bottleneck, Concat
from utils.augmentations import letterbox
from utils.general import non_max_suppression, check_dataset
from utils.dataloaders import create_dataloader
import utils.dataloaders as data_utils
data_utils.NUM_THREADS=2
from reproduction.reference import buffer_schedule, Domain, mqda_add, naive_add


def identity(path):
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}


def summary(x):
    x=np.asarray(x,dtype=float)
    return {'count':len(x),'mean_ms':float(x.mean()),'median_ms':float(np.median(x)),'p95_ms':float(np.percentile(x,95)),'std_ms':float(x.std()),'fps_from_mean':float(1000/x.mean())}


def preprocess(image):
    a=letterbox(image,640,auto=False)[0][:,:,::-1].transpose(2,0,1)
    return torch.from_numpy(np.ascontiguousarray(a)).to('cuda:0').float().unsqueeze(0)/255


class QuantizedReference:
    def __init__(self,model):
        self.model=model; self.ranges={}; self.mode='calibrate';self.layer_errors={};self.handles=[];self.weights={}
        for name,m in model.named_modules():
            if isinstance(m,torch.nn.Conv2d):
                self.weights[name]=m.weight.detach().clone()
                self.handles.append(m.register_forward_pre_hook(lambda mod,args,n=name:(self.process(n+'/in',args[0]),)))
                self.handles.append(m.register_forward_hook(lambda mod,args,y,n=name:self.process(n+'/out',y)))
            elif isinstance(m,Bottleneck) and m.add:
                def add_forward(mod,x,n=name):
                    y=mod.cv2(mod.cv1(x)); return self.fuse(n,[x,y],cat=False)
                m.forward=types.MethodType(add_forward,m)
            elif isinstance(m,Concat):
                def cat_forward(mod,x,n=name): return self.fuse(n,x,cat=True,dim=mod.d)
                m.forward=types.MethodType(cat_forward,m)

    def observe(self,name,x):
        low,high=float(x.min()),float(x.max()); old=self.ranges.get(name,(0.,0.))
        self.ranges[name]=(min(low,old[0]),max(high,old[1]))

    def domain(self,name):
        low,high=self.ranges[name];s=max((high-low)/255,1e-12);z=int(np.clip(np.rint(-low/s),0,255));return s,z

    def qdq(self,name,x):
        s,z=self.domain(name);return (torch.clamp(torch.round(x/s)+z,0,255)-z)*s

    def process(self,name,x):
        if self.mode=='calibrate': self.observe(name,x);return x
        if self.mode=='fp32':return x
        y=self.qdq(name,x)
        if self.mode=='error':
            e=self.layer_errors.setdefault(name,[0.,0.,0]);e[0]+=float((x-y).square().sum());e[1]+=float(x.square().sum());e[2]+=x.numel()
        return y

    def fuse(self,name,x,cat=False,dim=1):
        if self.mode in ('calibrate','fp32'):
            if self.mode=='calibrate':
                for i,v in enumerate(x):self.observe(f'{name}/branch{i}',v)
            y=torch.cat(x,dim) if cat else x[0]+x[1]
            if self.mode=='calibrate':self.observe(name+'/fused',y)
            return y
        outscale,outzero=self.domain(name+'/fused'); codes=[]
        for i,v in enumerate(x):
            s,z=self.domain(f'{name}/branch{i}');q=torch.clamp(torch.round(v/s)+z,0,255)
            if self.mode=='no_mqda':
                codes.append(q)
            else: codes.append((q-z)*s/outscale)
        if self.mode=='no_mqda':
            q=torch.cat(codes,dim) if cat else codes[0]+codes[1]-outzero
        else:
            raw=torch.cat(codes,dim) if cat else codes[0]+codes[1]
            q=torch.round(raw)+outzero
        return (torch.clamp(q,0,255)-outzero)*outscale

    def set_mode(self,mode):
        self.mode=mode
        for name,m in self.model.named_modules():
            if name in self.weights:
                w=self.weights[name]
                if mode!='fp32':
                    s=w.abs().amax((1,2,3),keepdim=True).clamp_min(1e-12)/127
                    w=torch.round(w/s).clamp(-127,127)*s
                m.weight.copy_(w)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--force',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261011);np.random.seed(20261011)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    out=ROOT/'outputs/experiments';out.mkdir(parents=True,exist_ok=True)
    files=sorted((ROOT/'data/coco128/images/train2017').glob('*.jpg'))
    if len(files)!=128:raise RuntimeError('Need complete 128-image dataset')
    images=[cv2.imread(str(p)) for p in files]
    if any(x is None for x in images):raise ValueError('Invalid dataset image')
    model=attempt_load(str(ROOT/'yolov5n.pt'),device=torch.device('cuda:0')).float().eval()
    result={'result_kind':'MEASURED','device':torch.cuda.get_device_name(0),'torch':torch.__version__,'checkpoint':identity(ROOT/'yolov5n.pt'),'dataset_archive':identity(ROOT/'data/downloads/coco128.zip'),'dataset':'COCO128 train2017 subset; sanity AP only, possible training overlap; not VOC2007','input':[1,3,640,640],'warmup':30,'repeats_per_image':3,'image_count':128,'seed':20261011,'parameters_fused':sum(p.numel() for p in model.parameters()),'graph_mismatch':'Conv6x6 stem / SiLU / COCO80 vs paper Focus / ReLU / VOC20','timing_scope':'Wall clock with CUDA sync. Includes letterbox+H2D, model sigmoid/decode, GPU NMS + detections D2H. Excludes image disk I/O. Batch1. Sequential frames; no overlapping.'}
    cached=json.loads((out/'fp32.json').read_text()) if (out/'fp32.json').is_file() else None
    if not args.force and cached and cached['checkpoint']==result['checkpoint'] and cached['dataset_archive']==result['dataset_archive'] and cached['torch']==result['torch'] and cached['device']==result['device'] and cached['input']==result['input'] and cached['warmup']==30 and (out/'fp32_samples.csv').is_file():
        result['fp32']=cached['fp32']
    else:
        x=preprocess(images[0])
        for _ in range(30):non_max_suppression(model(x)[0],.25,.45)
        torch.cuda.synchronize()
        rows=[]
        for repeat in range(3):
            for filename,image in zip(files,images):
                torch.cuda.synchronize();t0=time.perf_counter();x=preprocess(image);torch.cuda.synchronize();t1=time.perf_counter()
                pred=model(x)[0];torch.cuda.synchronize();t2=time.perf_counter()
                det=non_max_suppression(pred,.25,.45,max_det=300)[0];det.cpu();torch.cuda.synchronize();t3=time.perf_counter()
                rows.append({'image':filename.name,'repeat':repeat,'preprocess_ms':(t1-t0)*1000,'inference_ms':(t2-t1)*1000,'nms_ms':(t3-t2)*1000,'end_to_end_ms':(t3-t0)*1000,'detections':len(det)})
        with (out/'fp32_samples.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        result['fp32']={key:summary([r[key] for r in rows]) for key in ['preprocess_ms','inference_ms','nms_ms','end_to_end_ms']}
        (out/'fp32.json').write_text(json.dumps(result,indent=2))
    print('FP32 '+json.dumps(result['fp32']),flush=True)
    qref=QuantizedReference(model)
    for image in images[:32]:model(preprocess(image))
    calibration={name:{'min':r[0],'max':r[1],'scale':qref.domain(name)[0],'zero':qref.domain(name)[1]} for name,r in qref.ranges.items()}
    (out/'calibration.json').write_text(json.dumps({'kind':'SIMULATED','scheme':'uint8 affine activations, symmetric per-output-channel signed int8 weights, nearest-even, FP32 bias and convolution, Add/Cat signed affine oracle','images':[p.name for p in files[:32]],'ranges':calibration},indent=2))
    quantized_weights={};scales={};biases={}
    for name,m in model.named_modules():
        if name in qref.weights:
            original=qref.weights[name];scale=original.abs().amax((1,2,3),keepdim=True).clamp_min(1e-12)/127
            quantized_weights[name]=torch.round(original/scale).clamp(-127,127).to(torch.int8).cpu()
            scales[name]=scale.cpu()
            if m.bias is not None:biases[name]=m.bias.cpu()
    torch.save({'weight_codes':quantized_weights,'weight_scales':scales,'fp32_biases':biases,'activation_domains':calibration,'original_checkpoint_sha256':result['checkpoint']['sha256'],'scope':'Reconstructible QDQ weights; not native INT8 engine'},out/'quantized_weights.pt')
    # Separate model timing for QDQ; report simulated arithmetic on CUDA FP32.
    from val import run as validate
    data_yaml=out/'coco128_local.yaml'
    import yaml
    data_yaml.write_text(yaml.safe_dump({'path':str(ROOT/'data/coco128'),'train':'images/train2017','val':'images/train2017','names':model.names}),encoding='utf-8')
    result['accuracy']={};result['qdq_benchmark']={}
    validation_data=check_dataset(str(data_yaml))
    dataloader=create_dataloader(validation_data['val'],640,8,32,rect=False,workers=0,pad=0)[0]
    for mode in ['fp32','mqda','no_mqda']:
        qref.set_mode(mode);x=preprocess(images[64])
        for _ in range(10):model(x)
        times=[]
        for image in images[32:]:
            x=preprocess(image);torch.cuda.synchronize();t=time.perf_counter();model(x);torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
        result['qdq_benchmark'][mode]=summary(times)
        metrics,_,speeds=validate(data=validation_data,dataloader=dataloader,model=model,batch_size=8,imgsz=640,conf_thres=.001,iou_thres=.6,plots=False,save_dir=out/mode,workers=0,half=False)
        result['accuracy'][mode]={'precision':float(metrics[0]),'recall':float(metrics[1]),'map50':float(metrics[2]),'map50_95':float(metrics[3]),'scope':'COCO128 sanity, not independent held-out VOC validation','result_kind':'MEASURED' if mode=='fp32' else 'SIMULATED'}
        (out/'results.json').write_text(json.dumps(result,indent=2))
        print('AP '+mode+' '+json.dumps(result['accuracy'][mode]),flush=True)
    qref.set_mode('error')
    for image in images[32:48]:model(preprocess(image))
    with (out/'layer_errors.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['layer','local_mse','relative_rmse','elements'])
        for name,(e,p,n) in qref.layer_errors.items():writer.writerow([name,e/n,(e/max(p,1e-20))**.5,n])
    result['cpu_int8']=cpu_int8(out)
    result['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print('Experiments saved',flush=True)


def cpu_int8(out):
    """Actual quantized Conv2d backend, isolated kernel workload, not YOLO INT8."""
    supported=torch.backends.quantized.supported_engines
    backend=next((e for e in ['x86','fbgemm','onednn','qnnpack'] if e in supported),None)
    if backend is None:
        return {'result_kind':'NOT_VERIFIED','reason':'No native quantized CPU engine in this PyTorch build','supported_engines':supported}
    torch.backends.quantized.engine=backend;torch.set_num_threads(4)
    results=[]
    for k,co in [(3,8),(1,16)]:
        f=torch.nn.Conv2d(16,co,k,padding=k//2).eval();x=torch.rand(1,16,32,32)
        w=f.weight.detach();scales=w.abs().amax((1,2,3)).clamp_min(1e-12)/127
        qw=torch.quantize_per_channel(w,scales,torch.zeros(co,dtype=torch.int64),0,torch.qint8)
        q=torch.ao.nn.quantized.Conv2d(16,co,k,padding=k//2);q.set_weight_bias(qw,f.bias.detach())
        y=f(x);q.scale=float((y.max()-y.min())/255);q.zero_point=int(torch.clamp(torch.round(-y.min()/q.scale),0,255));qx=torch.quantize_per_tensor(x,1/255,0,torch.quint8)
        qy=q(qx).dequantize();mse=float((qy-y).square().mean());s={}
        for label,fn in [('FP32',lambda:f(x)),('INT8_'+backend,lambda:q(qx))]:
            for _ in range(30):fn()
            values=[]
            for _ in range(200):t=time.perf_counter();fn();values.append((time.perf_counter()-t)*1000)
            s[label]=summary(values)
        results.append({'kernel':k,'backend':backend,'shape':[1,16,32,32],'cout':co,'mse':mse,'timings':s,'result_kind':'MEASURED','scope':'isolated CPU Conv2d; excludes input quantization/output dequantization; not complete network'})
    return results


if __name__=='__main__':main()
