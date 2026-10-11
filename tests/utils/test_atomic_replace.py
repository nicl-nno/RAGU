import os

import pytest

from ragu.utils import atomic_replace


@pytest.mark.parametrize('winerror', [5, 32, 33])
def test_windows_lock_retries_without_losing_previous_value(tmp_path, monkeypatch, winerror):
    source, target = tmp_path / 'new.tmp', tmp_path / 'store.json'
    source.write_text('new')
    target.write_text('old')
    original_replace = os.replace
    attempts = []

    def replace(src, dst):
        assert target.read_text() == 'old'
        attempts.append(1)
        if len(attempts) < 3:
            error = PermissionError('transient sharing lock')
            error.winerror = winerror
            raise error
        original_replace(src, dst)

    monkeypatch.setattr(atomic_replace.os, 'replace', replace)
    monkeypatch.setattr(atomic_replace.time, 'sleep', lambda _: None)
    atomic_replace._replace_with_retry(source, target)
    assert len(attempts) == 3
    assert target.read_text() == 'new'
    assert not source.exists()


def test_persistent_windows_lock_has_bounded_retry_and_preserves_files(tmp_path, monkeypatch):
    source, target = tmp_path / 'new.tmp', tmp_path / 'store.json'
    source.write_text('new')
    target.write_text('old')
    clock = [0.0]
    attempts = []

    def replace(*_):
        attempts.append(1)
        error = PermissionError('persistent lock')
        error.winerror = 32
        raise error

    def advance(seconds):
        clock[0] += seconds

    monkeypatch.setattr(atomic_replace.os, 'replace', replace)
    monkeypatch.setattr(atomic_replace.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(atomic_replace.time, 'sleep', advance)
    with pytest.raises(PermissionError, match='persistent lock'):
        atomic_replace._replace_with_retry(source, target)
    assert 2 <= len(attempts) <= 62
    assert clock[0] == pytest.approx(3.0)
    assert target.read_text() == 'old'
    assert source.read_text() == 'new'


@pytest.mark.parametrize('error', [PermissionError('access denied'), OSError('disk failure')])
def test_other_errors_are_not_retried(monkeypatch, error):
    attempts = []

    def replace(*_):
        attempts.append(1)
        raise error

    monkeypatch.setattr(atomic_replace.os, 'replace', replace)
    with pytest.raises(type(error)):
        atomic_replace._replace_with_retry('source', 'target')
    assert len(attempts) == 1
