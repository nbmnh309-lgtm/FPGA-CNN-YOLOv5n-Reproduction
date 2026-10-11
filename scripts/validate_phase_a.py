"""Validate traceability without claiming that planned CNN/RTL modules passed."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KINDS = {'measured', 'software_simulation', 'rtl_testbench', 'estimate', 'paper_reported'}


def safe_path(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('Expected a nonempty relative artifact path')
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Artifact escapes checkout')
    return target


def validate_trace(document, root=ROOT):
    if document.get('schema_version') != 1:
        raise ValueError('Unknown traceability schema')
    entries = document.get('requirements')
    if not isinstance(entries, list) or not entries:
        raise ValueError('No trace requirements')
    seen = set()
    phases = set()
    for item in entries:
        identifier = item.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError('Missing or duplicate requirement ID')
        seen.add(identifier)
        phase = item.get('phase')
        if not isinstance(phase, str) or len(phase) != 1 or phase not in 'ABCDEFGHI':
            raise ValueError('Invalid phase')
        phases.add(phase)
        if not item.get('paper_source') or not item.get('requirement'):
            raise ValueError('Missing source/description')
        status = item.get('status')
        if status not in {'planned', 'implemented'}:
            raise ValueError('Unknown status')
        for field in ('code', 'test'):
            path = safe_path(root, item.get(field))
            if status == 'implemented' and not path.is_file():
                raise ValueError('Implemented artifact does not exist')
        if status == 'planned':
            if item.get('result') is not None or item.get('result_kind') is not None:
                raise ValueError('Planned item must not claim evidence')
        else:
            if item.get('result_kind') not in KINDS:
                raise ValueError('Missing result provenance')
            # The current run writes this report only after validation completes.
            safe_path(root, item.get('result'))
    if phases != set('ABCDEFGHI'):
        raise ValueError('Trace must cover A through I')
    return len(entries)


def validate_metrics(path):
    seen = set()
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['metric'] in seen or not row['metric']:
                raise ValueError('Duplicate/empty metric')
            seen.add(row['metric'])
            if row['result_kind'] != 'paper_reported':
                raise ValueError('Paper table must retain paper provenance')
            if not math.isfinite(float(row['value'])) or not row['source'] or not row['unit']:
                raise ValueError('Invalid metric/source')
    if not seen:
        raise ValueError('No paper metrics')
    return len(seen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    count = validate_trace(json.loads((ROOT / 'configs/reproduction/traceability.json').read_text()))
    metrics = validate_metrics(ROOT / 'configs/reproduction/paper_metrics.csv')
    result = {'schema_version': 1, 'result_kind': 'measured', 'scope': 'Phase A document/tooling validation only; no CNN or RTL execution', 'trace_requirements': count, 'paper_metrics': metrics, 'status': 'passed'}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
