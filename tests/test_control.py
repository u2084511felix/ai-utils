"""Check cancellation through the public wait functions."""

from threading import Event
from unittest.mock import AsyncMock, Mock

import pytest

from aiutils.control import check_non_negative_int, wait, wait_sync
from aiutils.exceptions import CallCancelledError


@pytest.fixture
def cancelled_wait(monkeypatch):
    signal = Event()
    sleep = AsyncMock(side_effect=lambda seconds: signal.set())
    monkeypatch.setattr("aiutils.control.asyncio.sleep", sleep)
    return signal, sleep


@pytest.mark.asyncio
async def test_async_wait_stops_on_cancellation(cancelled_wait):
    """GIVEN a capacity wait. WHEN cancellation is requested. THEN the wait stops promptly."""
    signal, sleep = cancelled_wait
    with pytest.raises(CallCancelledError):
        await wait(1, signal)
    expected = 1
    assert sleep.await_count == expected


@pytest.fixture
def sync_wait(monkeypatch):
    signal = Event()
    times = iter([0, 0, 0.05])
    monkeypatch.setattr("aiutils.control.time.monotonic", lambda: next(times))
    monkeypatch.setattr("aiutils.control.time.sleep", lambda seconds: signal.set())
    return signal


def test_sync_wait_stops_on_cancellation(sync_wait):
    """GIVEN a synchronous wait. WHEN cancellation is requested. THEN the wait stops."""
    with pytest.raises(CallCancelledError) as error:
        wait_sync(1, sync_wait)
    expected = CallCancelledError
    assert error.type is expected


@pytest.mark.parametrize("seconds", [-1, float("nan"), float("inf"), True])
@pytest.mark.asyncio
async def test_wait_rejects_invalid_duration(seconds):
    """GIVEN an invalid duration. WHEN a wait starts. THEN input is rejected."""
    with pytest.raises(ValueError) as error:
        await wait(seconds)
    expected = ValueError
    assert error.type is expected


@pytest.fixture
def async_sleep(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr("aiutils.control.asyncio.sleep", sleep)
    return sleep


@pytest.mark.asyncio
async def test_zero_async_wait_does_not_sleep(async_sleep):
    """GIVEN zero delay. WHEN an async wait runs. THEN no sleep occurs."""
    await wait(0)
    expected = 0
    assert async_sleep.await_count == expected


@pytest.fixture
def sync_sleep(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr("aiutils.control.time.sleep", sleep)
    return sleep


def test_zero_sync_wait_does_not_sleep(sync_sleep):
    """GIVEN zero delay. WHEN a sync wait runs. THEN no sleep occurs."""
    wait_sync(0)
    expected = 0
    assert sync_sleep.call_count == expected


@pytest.fixture
def cancelled_signal():
    signal = Event()
    signal.set()
    return signal


@pytest.mark.asyncio
async def test_zero_async_wait_checks_cancellation(cancelled_signal):
    """GIVEN zero delay and cancellation. WHEN a wait runs. THEN cancellation is raised."""
    with pytest.raises(CallCancelledError) as error:
        await wait(0, cancelled_signal)
    expected = CallCancelledError
    assert error.type is expected


def test_zero_sync_wait_checks_cancellation(cancelled_signal):
    """GIVEN zero delay and cancellation. WHEN a sync wait runs. THEN cancellation is raised."""
    with pytest.raises(CallCancelledError) as error:
        wait_sync(0, cancelled_signal)
    expected = CallCancelledError
    assert error.type is expected


@pytest.mark.parametrize("value", [False, True])
def test_token_count_rejects_booleans(value):
    """GIVEN a boolean count. WHEN it is checked. THEN the count is rejected."""
    with pytest.raises(ValueError) as error:
        check_non_negative_int(value)
    expected = ValueError
    assert error.type is expected
