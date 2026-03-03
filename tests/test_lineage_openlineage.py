import pytest
from data.lineage_tracker import track_lineage

def test_track_lineage_success(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.called = False
        def emit(self, event):
            self.called = True
    class DummyInputDataset:
        def __init__(self, *args, **kwargs):
            pass
    class DummyRunEvent:
        def __init__(self, *args, **kwargs):
            pass
    class DummyJob:
        def __init__(self, *args, **kwargs):
            pass
    class DummyRun:
        def __init__(self, *args, **kwargs):
            pass
    monkeypatch.setattr("data.lineage_tracker.OpenLineageClient", lambda url: DummyClient())
    monkeypatch.setattr("data.lineage_tracker.set_producer", lambda x: None)
    monkeypatch.setattr("data.lineage_tracker.InputDataset", DummyInputDataset)
    monkeypatch.setattr("data.lineage_tracker.RunEvent", DummyRunEvent)
    monkeypatch.setattr("data.lineage_tracker.Job", DummyJob)
    monkeypatch.setattr("data.lineage_tracker.Run", DummyRun)
    assert track_lineage([1, 2, 3]) is True

def test_track_lineage_failure(monkeypatch):
    def fail_emit(*args, **kwargs):
        raise Exception("fail")
    class DummyClient:
        def emit(self, event):
            raise Exception("fail")
    monkeypatch.setattr("openlineage.client.OpenLineageClient", lambda url: DummyClient())
    monkeypatch.setattr("openlineage.client.set_producer", lambda x: None)
    # Patch uuid to avoid real uuid
    import uuid
    monkeypatch.setattr(uuid, "uuid4", lambda: "id")
    # Patch event to raise
    assert track_lineage([1, 2, 3], event_type="FAIL") is False

def test_track_lineage_env(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.called = False
        def emit(self, event):
            self.called = True
    class DummyInputDataset:
        def __init__(self, *args, **kwargs):
            pass
    class DummyRunEvent:
        def __init__(self, *args, **kwargs):
            pass
    class DummyJob:
        def __init__(self, *args, **kwargs):
            pass
    class DummyRun:
        def __init__(self, *args, **kwargs):
            pass
    monkeypatch.setattr("data.lineage_tracker.OpenLineageClient", lambda url: DummyClient())
    monkeypatch.setattr("data.lineage_tracker.set_producer", lambda x: None)
    monkeypatch.setattr("data.lineage_tracker.InputDataset", DummyInputDataset)
    monkeypatch.setattr("data.lineage_tracker.RunEvent", DummyRunEvent)
    monkeypatch.setattr("data.lineage_tracker.Job", DummyJob)
    monkeypatch.setattr("data.lineage_tracker.Run", DummyRun)
    import os
    monkeypatch.setenv("OPENLINEAGE_URL", "http://test")
    assert track_lineage([1, 2, 3]) is True
