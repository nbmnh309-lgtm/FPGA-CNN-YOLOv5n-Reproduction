"""Content-verified checkpoints; failures recorded and independent work continues."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'outputs/experiment_status.json'


def hash_file(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp():return datetime.now(timezone(timedelta(hours=7))).isoformat()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)


def signature(files):
    h=hashlib.sha256()
    for name in sorted(files):
        p=ROOT/name;h.update(name.encode());h.update(p.read_bytes() if p.is_file() else b'MISSING')
    h.update(sys.version.encode())
    for name in ['torch','torchvision','numpy','pytest']:
        try:h.update(importlib.metadata.version(name).encode())
        except importlib.metadata.PackageNotFoundError:h.update(b'MISSING')
    return h.hexdigest()


def valid_checkpoint(record,sig):
    if record.get('status')!='PASS' or record.get('signature')!=sig or not record.get('artifacts'):return False
    for name,digest in record['artifacts'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file() or hash_file(p)!=digest:return False
    return True


def environment():
    import torch
    result={'date':stamp(),'platform':platform.platform(),'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}
    if torch.cuda.is_available():result.update(gpu=torch.cuda.get_device_name(0),vram_free_total=list(torch.cuda.mem_get_info()))
    result['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    result['branch']=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    result['nvidia_smi']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.free,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=15).stdout.strip()
    write_json(ROOT/'outputs/environment.json',result);return result


def execute_stage(state,name,cmd,sources,artifacts,timeout=600,force=False):
    sig=signature(sources);previous=state['stages'].get(name,{})
    if not force and valid_checkpoint(previous,sig):print('SKIP verified '+name,flush=True);return True
    record={'status':'RUNNING','signature':sig,'started':stamp(),'command':cmd,'artifacts':{}}
    state['stages'][name]=record;write_json(STATE,state)
    logs=ROOT/'outputs/logs';logs.mkdir(parents=True,exist_ok=True)
    logfile=logs/(name+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.log')
    print('RUN '+name,flush=True)
    try:
        env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with logfile.open('w',encoding='utf-8') as log:
            completed=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        if completed.returncode:raise RuntimeError(f'Exit code {completed.returncode}')
        for artifact in artifacts:
            path=ROOT/artifact
            if not path.is_file():raise RuntimeError(f'Missing artifact {artifact}')
            if path.suffix=='.json':json.loads(path.read_text(encoding='utf-8'))
            record['artifacts'][artifact]=hash_file(path)
        record['status']='PASS'
    except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as exc:
        record.update(status='FAIL',error=str(exc));print(f'FAIL {name}: {exc}; log {logfile}',flush=True)
    record.update(ended=stamp(),log=str(logfile.relative_to(ROOT)));write_json(STATE,state)
    print(record['status']+' '+name,flush=True);return record['status']=='PASS'


def main():
    p=argparse.ArgumentParser();p.add_argument('--force',action='store_true');p.add_argument('--skip-gpu',action='store_true');args=p.parse_args()
    env=environment()
    state=json.loads(STATE.read_text(encoding='utf-8')) if STATE.is_file() else {'schema_version':1,'stages':{}}
    state.update(updated=stamp(),environment='outputs/environment.json',scope='Local maximum feasible method reproduction; physical FPGA/VOC paper replication incomplete')
    py=[sys.executable,'-B'];ok=True
    tests=execute_stage(state,'python_tests',py+['-m','pytest','tests/python','-q','--junitxml=outputs/python_tests.xml'],[str(x.relative_to(ROOT)) for x in (ROOT/'tests/python').glob('*.py')]+['reproduction/reference.py','windows_runner/runner.py','scripts/validate_phase_a.py','scripts/collect_reproduction_manifest.py','configs/reproduction/traceability.json','configs/reproduction/paper_metrics.csv','pytest.ini'],['outputs/python_tests.xml'],force=args.force);ok&=tests
    inputs=execute_stage(state,'inputs',py+['-m','windows_runner.prepare_inputs'],['windows_runner/prepare_inputs.py','yolov5n.pt','models/yolov5/models/yolo.py','data/downloads/coco128.zip'],['outputs/inputs.json'],force=args.force);ok&=inputs
    rtl=execute_stage(state,'rtl',py+['-m','windows_runner.rtl_verify'],['windows_runner/rtl_verify.py','reproduction/reference.py',*map(lambda x:str(x.relative_to(ROOT)),(ROOT/'rtl').rglob('*.sv')),'testbench/unit/tb_buffer.sv'],['outputs/rtl/results.json','outputs/rtl/commands.log'],force=args.force);ok&=rtl
    if not args.skip_gpu and inputs and env['cuda_available']:
        src=['benchmarks/local_experiments.py','benchmarks/__init__.py','reproduction/reference.py','yolov5n.pt','data/downloads/coco128.zip','models/yolov5/models/yolo.py']
        src.extend(str(x.relative_to(ROOT)) for x in (ROOT/'data/coco128').rglob('*') if x.suffix in ('.txt','.jpg'))
        src.extend(str(x.relative_to(ROOT)) for x in (ROOT/'models/yolov5').rglob('*.py') if '.git' not in x.parts)
        gpu=execute_stage(state,'gpu_experiments',py+['-m','benchmarks.local_experiments']+(['--force'] if args.force else []),src,['outputs/experiments/results.json','outputs/experiments/fp32_samples.csv','outputs/experiments/layer_errors.csv','outputs/experiments/calibration.json','outputs/experiments/quantized_weights.pt'],force=args.force);ok&=gpu
    elif not args.skip_gpu:
        state['stages']['gpu_experiments']={'status':'NOT_VERIFIED','reason':'inputs or CUDA unavailable'};ok=False
    # Synthesis is a separate optional milestone because XC7Z100 needs a license.
    vivados=sorted(Path('C:/Xilinx/Vivado').glob('*/bin/vivado.bat'))
    if vivados:
        synth=execute_stage(state,'dmp_synthesis_z020',[str(vivados[-1]),'-mode','batch','-source','scripts/synth_dmp.tcl','-tclargs','xc7z020clg484-1','-log','outputs/rtl/synthesis_z020_runner.log','-journal','outputs/rtl/synthesis_z020_runner.jou'],['scripts/synth_dmp.tcl','rtl/dual_multiply_packing/dmp.sv'],['outputs/rtl/dmp_synthesis.json','outputs/rtl/dmp_utilization.rpt'],force=args.force);ok&=synth
    write_json(STATE,state)
    report=execute_stage(state,'report',py+['-m','benchmarks.summarize'],['benchmarks/summarize.py','outputs/experiment_status.json','outputs/experiments/results.json','outputs/rtl/results.json','outputs/rtl/dmp_synthesis.json'],['reports/LOCAL_REPRODUCTION_VI.md','reports/local_results.json','docs/PROGRESS.md','docs/REPRODUCTION_MATRIX.md'],force=True);ok&=report
    state['last_run']=stamp();state['last_exit_code']=0 if ok else 1;write_json(STATE,state)
    print('Checkpoint: '+str(STATE),flush=True);return 0 if ok else 1


if __name__=='__main__':sys.exit(main())
