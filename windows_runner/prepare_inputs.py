"""Acquire small public sanity dataset; preserve existing upstream and checkpoint."""
import hashlib
import json
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
YOLO_COMMIT='f9578796b8a64889aa9e66dc06914d19f40d338e'
DATA_URL='https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip'


def download(url,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix(target.suffix+'.partial')
    with urllib.request.urlopen(url,timeout=60) as src,temp.open('wb') as dst:
        while block:=src.read(1024*1024):dst.write(block)
    temp.replace(target)


def main():
    upstream=ROOT/'models/yolov5'
    if not upstream.exists():
        subprocess.run(['git','clone','https://github.com/ultralytics/yolov5.git',str(upstream)],check=True,timeout=180)
        subprocess.run(['git','-C',str(upstream),'checkout',YOLO_COMMIT],check=True,timeout=30)
    commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    if commit!=YOLO_COMMIT:raise RuntimeError('Upstream commit differs; preserve it and explicitly revalidate compatibility')
    dataset=ROOT/'data/coco128';archive=ROOT/'data/downloads/coco128.zip'
    if not archive.is_file():download(DATA_URL,archive)
    if not dataset.is_dir():
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                if not (ROOT/'data'/item.filename).resolve().is_relative_to((ROOT/'data').resolve()):raise ValueError('Unsafe archive path')
            z.extractall(ROOT/'data')
    if len(list((dataset/'images/train2017').glob('*.jpg')))!=128:raise ValueError('Incomplete COCO128 dataset')
    checkpoint=ROOT/'yolov5n.pt'
    if not checkpoint.is_file():download('https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt',checkpoint)
    metadata={'upstream_commit':commit,'upstream_url':'https://github.com/ultralytics/yolov5','dataset_url':DATA_URL,'dataset_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'checkpoint_provenance':'Existing user checkpoint; hash recorded, authors VOC checkpoint unavailable','dataset_scope':'COCO128 sanity; not paper VOC2007'}
    target=ROOT/'outputs/inputs.json';target.parent.mkdir(exist_ok=True);target.write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata))


if __name__=='__main__':main()
