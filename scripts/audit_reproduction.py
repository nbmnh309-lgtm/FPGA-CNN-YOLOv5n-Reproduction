"""Verify retained evidence, then capture the real Python runner exit code.

Does not force new timings: verified runner stages may be cached. Preserves the
pre-audit state and logs in reports/audit_evidence for independent review.
"""
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'reports/audit_evidence'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def preserve(path, name=None):
    target = DEST / (name or str(path.relative_to(ROOT)).replace('\\', '/').replace('/', '__'))
    shutil.copy2(path, target)
    return {'path': str(target.relative_to(ROOT)).replace('\\', '/'),
            'sha256': digest(target), 'bytes': target.stat().st_size}


def junit(path):
    suite = ET.parse(path).getroot().find('testsuite')
    cases = suite.findall('testcase')
    assert not suite.findall('.//failure') and not suite.findall('.//error')
    assert all(int(suite.get(k, '0')) == 0 for k in ('failures', 'errors', 'skipped'))
    return {'suite_counter': int(suite.get('tests')), 'testcase_elements': len(cases),
            'extra_subtest_events': int(suite.get('tests')) - len(cases),
            'failures': 0, 'errors': 0, 'skipped': 0,
            'cases': [c.get('classname') + '.' + c.get('name') for c in cases]}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    original = read('outputs/experiment_status.json')
    proof = {'date_utc': datetime.now(timezone.utc).isoformat(),
             'cwd': str(ROOT), 'platform': platform.platform(), 'python': sys.version,
             'python_executable': sys.executable,
             'source_head_before_audit_commit': subprocess.check_output(
                 ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'packages': {n: importlib.metadata.version(n) for n in
                          ['pytest', 'numpy', 'torch', 'torchvision', 'opencv-python',
                           'PyYAML', 'matplotlib']}, 'retained_files': [], 'stages_before': {}}
    proof['retained_files'].append(preserve(ROOT/'outputs/experiment_status.json', 'state_before.json'))
    proof['junit_before'] = junit(ROOT/'outputs/python_tests.xml')
    proof['retained_files'].append(preserve(ROOT/'outputs/python_tests.xml', 'python_tests_before.xml'))
    for name, stage in original['stages'].items():
        checked = {}
        for file, expected in stage.get('artifacts', {}).items():
            path = ROOT / file
            assert path.is_file() and digest(path) == expected, f'Artifact mismatch: {file}'
            checked[file] = expected
        log = ROOT / stage['log']
        assert log.is_file(), f'Missing log: {name}'
        proof['stages_before'][name] = {'status': stage['status'], 'artifact_hashes': checked,
                                      'log': preserve(log, name + '_before.log')}
        assert stage['status'] == 'PASS'
    gpu = read('outputs/experiments/results.json')
    rows = list(csv.DictReader((ROOT/'outputs/experiments/fp32_samples.csv').open(encoding='utf-8')))
    counts = Counter(r['image'] for r in rows)
    assert len(rows) == 384 and len(counts) == 128 and set(counts.values()) == {3}
    assert len({(r['image'], r['repeat']) for r in rows}) == 384
    assert {r['repeat'] for r in rows} == {'0', '1', '2'}
    stats = {}
    for column in ('preprocess_ms', 'inference_ms', 'nms_ms', 'end_to_end_ms'):
        values = np.array([float(r[column]) for r in rows])
        assert np.all(np.isfinite(values)) and np.all(values > 0)
        actual = {'count': len(values), 'mean_ms': float(values.mean()),
                  'median_ms': float(np.median(values)), 'p95_ms': float(np.percentile(values, 95)),
                  'std_ms': float(values.std()), 'fps_from_mean': float(1000/values.mean())}
        for key, value in actual.items():
            assert math.isclose(value, gpu['fp32'][column][key], rel_tol=1e-10, abs_tol=1e-10), (column, key)
        stats[column] = actual
    # All stages use adjoining perf_counter intervals; E2E is their sum.
    assert max(abs(float(r['end_to_end_ms']) - sum(float(r[k]) for k in
               ('preprocess_ms', 'inference_ms', 'nms_ms'))) for r in rows) < 1e-8
    proof['benchmark'] = {'samples': 384, 'unique_images': 128, 'repeats': 3,
                          'statistics_recomputed': stats, 'accuracy': gpu['accuracy'],
                          'cpu_int8': gpu['cpu_int8'], 'raw_cpu_samples_available': False}
    for key in ('checkpoint', 'dataset_archive'):
        assert digest(Path(gpu[key]['path'])) == gpu[key]['sha256']
        proof[key] = gpu[key]
    proof['upstream_commit'] = subprocess.check_output(
        ['git', '-C', str(ROOT/'models/yolov5'), 'rev-parse', 'HEAD'], text=True).strip()
    assert proof['upstream_commit'] == read('outputs/inputs.json')['upstream_commit']
    members_checked = 0
    with zipfile.ZipFile(gpu['dataset_archive']['path']) as archive:
        for member in archive.infolist():
            if not member.is_dir() and Path(member.filename).suffix in ('.jpg', '.txt'):
                local = ROOT/'data'/member.filename
                assert local.is_file() and digest(local) == hashlib.sha256(archive.read(member)).hexdigest(), member.filename
                members_checked += 1
    proof['extracted_dataset_files_matching_archive'] = members_checked
    rtl = read('outputs/rtl/results.json')
    expected = {'tb_dmp', 'tb_buffer', 'tb_npu', 'tb_dot3', 'tb_dot1', 'tb_stream', 'tb_mqda0', 'tb_mqda1'}
    assert set(rtl['tests']) == expected and len(rtl['operator_cases']) == 27
    for name, test in rtl['tests'].items():
        assert test['status'] == 'PASS' and 'PASS ' in test['output']
        assert 'Vivado Simulator 2020.2' in test['output'] and 'SW Build 3064766' in test['output']
    proof['rtl'] = rtl
    synth = read('outputs/rtl/dmp_synthesis.json')
    util = (ROOT/'outputs/rtl/dmp_utilization.rpt').read_text()
    assert synth['dsp48e1'] == 1 and synth['part'] == 'xc7z020clg484-1'
    assert '7z020clg484-1' in util and '| DSP48E1  |    1 |' in util
    proof['synthesis'] = synth
    license_log = (ROOT/'outputs/rtl/synthesis.log').read_text()
    assert 'xc7z100ffg900-2' in license_log and '[Common 17-345]' in license_log
    proof['paper_target_failure'] = 'xc7z100ffg900-2: Common 17-345 / 17-348 license unavailable'
    for file in ('outputs/rtl/results.json', 'outputs/rtl/commands.log',
                 'outputs/rtl/dmp_utilization.rpt', 'outputs/rtl/synthesis.log',
                 'outputs/rtl/dmp_synth_timing.rpt', 'outputs/experiments/results.json',
                 'outputs/experiments/calibration.json', 'outputs/environment.json'):
        proof['retained_files'].append(preserve(ROOT/file))
    # Capture installed tools and the actual local GPU, not a inferred cloud identity.
    import torch
    proof['cuda_available_now'] = torch.cuda.is_available()
    assert proof['cuda_available_now']
    proof['gpu_now'] = torch.cuda.get_device_name(0)
    proof['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True)
    version = subprocess.run(['C:/Xilinx/Vivado/2020.2/bin/vivado.bat', '-version'],
                             capture_output=True, text=True)
    assert version.returncode == 0 and 'Vivado v2020.2' in version.stdout
    proof['vivado_version'] = version.stdout
    # Invoke the documented Python entry point without changing Windows policy.
    command = [sys.executable, '-B', str(ROOT/'windows_runner/runner.py')]
    started = datetime.now(timezone.utc).isoformat()
    with (DEST/'runner_audit.log').open('w', encoding='utf-8') as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    proof['runner_receipt'] = {'command': command, 'started_utc': started,
                              'ended_utc': datetime.now(timezone.utc).isoformat(),
                              'actual_process_returncode': completed.returncode,
                              'log': {'path': 'reports/audit_evidence/runner_audit.log',
                                      'sha256': digest(DEST/'runner_audit.log')},
                              'scope': 'normal cached runner; not a forced complete rerun'}
    assert completed.returncode == 0
    current = read('outputs/experiment_status.json')
    assert current['last_exit_code'] == 0
    proof['junit_after'] = junit(ROOT/'outputs/python_tests.xml')
    proof['stages_after'] = {}
    for name, stage in current['stages'].items():
        assert stage['status'] == 'PASS'
        for file, expected_hash in stage['artifacts'].items():
            assert digest(ROOT/file) == expected_hash, file
        proof['stages_after'][name] = stage
    proof['retained_files'].append(preserve(ROOT/'outputs/python_tests.xml', 'python_tests_after.xml'))
    proof['retained_files'].append(preserve(ROOT/current['stages']['python_tests']['log'], 'python_tests_after.log'))
    (ROOT/'reports/final_audit_evidence.json').write_text(
        json.dumps(proof, indent=2, ensure_ascii=False), encoding='utf-8')
    print('AUDIT PASS: 24 testcase + 2 subtests; 8 XSim; 384 raw samples; 1 DSP; runner actual exit 0')


if __name__ == '__main__':
    main()
