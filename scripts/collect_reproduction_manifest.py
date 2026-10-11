"""Collect reproducibility metadata only. No inference, download, or training."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_value(root, *args):
    result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise ValueError(f'Git metadata unavailable: {args[0]}')
    return result.stdout.strip()


def file_identity(path):
    if not path.is_file():
        raise ValueError(f'Input file missing: {path.name}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'filename': path.name, 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--model-yaml', type=Path)
    parser.add_argument('--data-yaml', type=Path)
    parser.add_argument('--torch-info', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = {'schema_version': 1, 'result_kind': 'measured', 'scope': 'metadata only; no inference or benchmark', 'collected_at_utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(), 'python': platform.python_version(), 'project_commit': git_value(root, 'rev-parse', 'HEAD'), 'project_dirty': bool(git_value(root, 'status', '--porcelain')), 'packages': {}}
    for package in ('torch', 'torchvision', 'numpy', 'PyYAML'):
        try:
            result['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][package] = None
    if args.upstream:
        result['upstream'] = {'commit': git_value(args.upstream, 'rev-parse', 'HEAD'), 'dirty': bool(git_value(args.upstream, 'status', '--porcelain'))}
    for name in ('weights', 'model_yaml', 'data_yaml'):
        value = getattr(args, name)
        result[name] = file_identity(value) if value else None
    if args.torch_info:
        import torch
        available = torch.cuda.is_available()
        result['cuda'] = {'available': available, 'torch_cuda_version': torch.version.cuda, 'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if available else []}
    else:
        result['cuda'] = None
    result['unresolved'] = ['graph/classes/anchors/activations need inspection', 'training and calibration provenance', 'VOC split and AP/NMS/preprocessing protocol']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Metadata saved: {args.output}')


if __name__ == '__main__':
    main()
