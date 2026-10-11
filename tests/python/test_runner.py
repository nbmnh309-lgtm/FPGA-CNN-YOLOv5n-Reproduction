import json
from pathlib import Path
from windows_runner.runner import ROOT,signature,valid_checkpoint,hash_file
import sys
import tempfile
import windows_runner.runner as runner


def test_corrupt_and_missing_artifact_cannot_skip(tmp_path):
    relative='outputs/cache_test.json';path=ROOT/relative;path.parent.mkdir(exist_ok=True)
    try:
        path.write_text('{"passed":true}')
        record={'status':'PASS','signature':'abc','artifacts':{relative:hash_file(path)}}
        assert valid_checkpoint(record,'abc')
        assert not valid_checkpoint(record,'different')
        path.write_text('{"passed":false}')
        assert not valid_checkpoint(record,'abc')
        path.unlink();assert not valid_checkpoint(record,'abc')
        record['status']='RUNNING';assert not valid_checkpoint(record,'abc')
    finally:
        if path.exists():path.unlink()


def test_signature_changes_with_source(tmp_path):
    relative='outputs/signature_test.txt';path=ROOT/relative
    try:
        path.write_text('a');first=signature([relative]);path.write_text('b')
        assert first!=signature([relative])
    finally:
        if path.exists():path.unlink()


def test_failed_stage_is_recorded_and_can_recover(monkeypatch):
    with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temporary:
        work=Path(temporary);monkeypatch.setattr(runner,'STATE',work/'state.json')
        artifact=str((work/'proof.json').relative_to(ROOT));state={'stages':{}}
        assert not runner.execute_stage(state,'test_failure',[sys.executable,'-c','raise SystemExit(7)'],[],[artifact],timeout=10)
        failed_log=state['stages']['test_failure']['log']
        assert state['stages']['test_failure']['status']=='FAIL'
        persisted=json.loads((work/'state.json').read_text())
        assert persisted['stages']['test_failure']['error']=='Exit code 7'
        command=[sys.executable,'-c','import sys;from pathlib import Path;Path(sys.argv[1]).write_text("{}")',str(work/'proof.json')]
        assert runner.execute_stage(state,'test_failure',command,[],[artifact],timeout=10)
        assert state['stages']['test_failure']['log']!=failed_log
        assert (ROOT/failed_log).is_file()
