"""Reject unsafe paths and unsupported reproduction evidence."""
import copy
import importlib.util
import json
import hashlib
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('validate_phase_a', ROOT / 'scripts/validate_phase_a.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class TraceabilityTests(unittest.TestCase):
    def setUp(self):
        self.trace = json.loads((ROOT / 'configs/reproduction/traceability.json').read_text())

    def test_real_trace_and_metrics(self):
        self.assertGreater(validator.validate_trace(self.trace), 9)
        self.assertGreater(validator.validate_metrics(ROOT / 'configs/reproduction/paper_metrics.csv'), 10)

    def test_duplicate_ids_rejected(self):
        self.trace['requirements'].append(copy.deepcopy(self.trace['requirements'][0]))
        with self.assertRaises(ValueError):
            validator.validate_trace(self.trace)

    def test_future_module_cannot_claim_a_pass(self):
        self.trace['requirements'][1]['result'] = 'reports/nonexistent.json'
        self.trace['requirements'][1]['result_kind'] = 'rtl_testbench'
        with self.assertRaises(ValueError):
            validator.validate_trace(self.trace)

    def test_missing_implementation_rejected(self):
        self.trace['requirements'][0]['code'] = 'models/not_implemented.py'
        with self.assertRaises(ValueError):
            validator.validate_trace(self.trace)

    def test_path_escape_rejected(self):
        for path in ('../outside.py', '/tmp/outside.py'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                validator.safe_path(ROOT, path)

    def test_invalid_phase_type_rejected(self):
        self.trace['requirements'][0]['phase'] = None
        with self.assertRaises(ValueError):
            validator.validate_trace(self.trace)

    def test_missing_phase_rejected(self):
        self.trace['requirements'] = [r for r in self.trace['requirements'] if r['phase'] != 'H']
        with self.assertRaises(ValueError):
            validator.validate_trace(self.trace)

    def test_paper_values_cannot_be_relabelled_as_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.csv'
            path.write_text('metric,value,unit,source,result_kind,notes\npower,8.536,W,paper,measured,bad\n')
            with self.assertRaises(ValueError):
                validator.validate_metrics(path)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('collector', ROOT / 'scripts/collect_reproduction_manifest.py')
        self.collector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.collector)

    def test_file_hash_without_disclosing_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.yaml'
            content = b'nc: 20\n'
            path.write_bytes(content)
            record = self.collector.file_identity(path)
            self.assertEqual(record['sha256'], hashlib.sha256(content).hexdigest())
            self.assertEqual(record['bytes'], len(content))
            self.assertEqual(set(record), {'filename', 'bytes', 'sha256'})

    def test_missing_input_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self.collector.file_identity(Path(tmp) / 'missing.pt')


if __name__ == '__main__':
    unittest.main()
