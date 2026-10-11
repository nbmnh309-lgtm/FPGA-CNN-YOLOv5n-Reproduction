import json
from pathlib import Path
from windows_runner.runner import ROOT,signature,valid_checkpoint,hash_file


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
