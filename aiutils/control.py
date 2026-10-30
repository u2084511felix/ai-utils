"""Cancellation signals and wait services for model calls."""

import asyncio
import math
import time
from typing import Any, Protocol

from .exceptions import CallCancelledError


class Cancellation(Protocol):
    """A signal that can stop a model call."""

    def is_set(self) -> bool:
        """Return whether cancellation was requested."""
        ...


def check_cancel(cancel: Cancellation | None) -> None:
    """Stop when the cancellation signal is set.

    Args:
        cancel: Optional cancellation signal.

    Raises:
        CallCancelledError: Cancellation was requested.
    """
    if cancel is not None and cancel.is_set():
        raise CallCancelledError("The model call was cancelled.")


def check_finite_number(value: float | int, name: str, *, minimum: float = 0) -> float | int:
    """Check a finite numeric value against a lower limit.

    Args:
        value: Value to check.
        name: Input name for error messages.
        minimum: Lowest permitted value.

    Returns:
        The checked value.

    Raises:
        ValueError: The value is invalid.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}.")
    return value


def check_non_negative_int(value: Any, name: str = "input_tokens") -> int:
    """Type check a token count.

    Args:
        value: Count to check.
        name: Input name for error messages.

    Returns:
        The checked count.

    Raises:
        ValueError: The count is not a nonnegative integer.
    """
    match value:
        case None:
            raise ValueError(f"{name} must not be None.")
        case bool():
            raise ValueError(f"{name} must not be a boolean, got {value}.")
        case int() as v if v >= 0:
            return v
        case _:
            raise ValueError(f"{name} must be a nonnegative integer, got {value}.")


def _check_wait_args(seconds: float | int, cancel: Cancellation | None = None) -> None:
    check_finite_number(seconds, "seconds")
    if cancel is not None and not callable(getattr(cancel, "is_set", None)):
        raise ValueError("cancel must have an is_set() method if provided.")


async def wait(seconds: float | int, cancel: Cancellation | None = None) -> None:
    """Wait without blocking the event loop.

    Args:
        seconds: Wait duration.
        cancel: Optional cancellation signal.
    """
    min_step = 0.05
    _check_wait_args(seconds, cancel)
    remaining = seconds
    while remaining > 0:
        check_cancel(cancel)
        step = min(remaining, min_step)
        await asyncio.sleep(step)
        remaining -= step
    check_cancel(cancel)


def wait_sync(seconds: float | int, cancel: Cancellation | None = None) -> None:
    """Wait in a synchronous caller.

    Args:
        seconds: Wait duration.
        cancel: Optional cancellation signal.
    """
    min_step = 0.05
    _check_wait_args(seconds, cancel)
    end = time.monotonic() + seconds
    while (remaining := end - time.monotonic()) > 0:
        check_cancel(cancel)
        time.sleep(min(remaining, min_step))
    check_cancel(cancel)
