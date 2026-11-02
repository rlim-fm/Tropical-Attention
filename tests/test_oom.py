"""The final evaluation waits out, then shrinks past, a transient CUDA out-of-memory error."""
import pytest
import torch

import train


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(train.time, 'sleep', slept.append)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    return slept


def failing(times):
    calls = []

    def call(shrink):
        calls.append(shrink)
        if len(calls) <= times:
            raise torch.cuda.OutOfMemoryError('CUDA out of memory.')
        return shrink
    return call, calls


def test_a_transient_oom_is_waited_out_at_full_size(no_sleep):
    call, calls = failing(2)
    assert train._retry_oom(call) == 1
    assert calls == [1, 1, 1] and no_sleep == list(train.OOM_WAITS[:2])


def test_after_the_waits_the_batch_shrinks(no_sleep):
    waits = len(train.OOM_WAITS)
    call, calls = failing(waits + 2)
    assert train._retry_oom(call) == 4
    assert calls == [1] * (waits + 1) + [2, 4]


def test_a_persistent_oom_is_raised():
    call, calls = failing(10 ** 6)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        train._retry_oom(call)
    assert calls[-1] == 2 ** train.OOM_HALVINGS


def test_other_errors_pass_straight_through():
    def call(shrink):
        raise ValueError('not memory')
    with pytest.raises(ValueError):
        train._retry_oom(call)
