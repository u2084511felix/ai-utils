"""Retry policy and transport-attempt accounting."""

import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Protocol, Self

from openai import APIConnectionError, APIStatusError
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, NonNegativeInt, model_validator

from aiutils.exceptions import InitializationError

from .budget import SpendBudget
from .control import (
    Cancellation, 
    wait, 
    wait_sync, 
    check_cancel, 
    check_finite_number, 
    check_non_negative_int
)


class RetryPolicy(Protocol):

    def delay(
        self, 
        error: Exception, 
        attempt: int, 
        *, 
        jitter: Callable[[], float] = random.random,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    ) -> float | None:
        """Get the delay before a permitted retry.

        Args:
            error: Transport or provider failure.
            attempt: Number of retries already made.
            jitter: Random value source in the range zero to one.
            now: Current UTC time source.

        Returns:
            Delay in seconds, or None when no retry is permitted.
        """
        ...


class _RetryPolicy(BaseModel):
    """Limits for retries after the initial transport attempt.

    Attributes:
        retries: Maximum retries after the initial attempt.
        min_wait: Initial upper limit for a random delay in seconds.
        max_wait: Maximum delay in seconds. Longer provider delays stop retries.
    """
    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    retries: NonNegativeInt = 2
    min_wait: NonNegativeFloat = 0.25
    max_wait: NonNegativeFloat = 30.0

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.min_wait > self.max_wait:
            raise ValueError(f"min_wait cannot exceed max_wait, got min_wait={self.min_wait}, max_wait={self.max_wait}.")
        return self

    def delay(
        self, 
        error: Exception, 
        attempt: int, 
        *, 
        jitter: Callable[[], float] = random.random,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    ) -> float | None:

        check_non_negative_int(attempt, "attempt")
        transient = isinstance(error, (APIConnectionError, TimeoutError, ConnectionError))
        if isinstance(error, APIStatusError):
            transient = error.status_code == 429 or 500 <= error.status_code < 600
        if not transient or attempt >= self.retries:
            return None
        response = getattr(error, "response", None)
        value = response.headers.get("retry-after") if response is not None else None
        if value is not None:
            delay = _provider_delay(value, now)
            if delay is not None:
                return delay if delay <= self.max_wait else None
        bound = min(self.max_wait, self.min_wait * 2 ** min(attempt, 1000))
        fraction = check_finite_number(jitter(), "jitter")
        if fraction > 1:
            raise ValueError("jitter must be between zero and one.")
        return bound * fraction


def make_retry_policy(
    retries: int = 2,
    min_wait: float = 0.25,
    max_wait: float = 30.0
) -> RetryPolicy:
    try:
        return _RetryPolicy(retries=retries, min_wait=min_wait, max_wait=max_wait)
    except Exception as e:
        raise InitializationError("Failed to create RetryPolicy.") from e


def _provider_delay(value, now):
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                return None
            seconds = max(0.0, (date - now()).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


class RequestRunner:
    """Apply one retry policy and budget to transport attempts."""

    def __init__(
        self, 
        policy: RetryPolicy, 
        *, 
        budget: SpendBudget | None = None,
    ):
        self.policy = policy
        self.budget = budget


    def validate(self, input_tokens: int | None, cancel: Any):
        """Check cancellation and the input estimate before a request.

        Args:
            input_tokens: Optional input estimate.
            cancel: Optional cancellation signal.

        Raises:
            ValueError: A budget has no valid input estimate.
        """
        check_cancel(cancel)
        if self.budget is not None or input_tokens is not None:
            check_non_negative_int(input_tokens)

    async def run(self, operation, *, model, input_tokens=None, cancel=None, usage):
        """Run an asynchronous operation with per-attempt accounting.

        Args:
            operation: Transport operation.
            model: Model name.
            input_tokens: Estimated input count when a budget is used.
            cancel: Optional cancellation signal.
            usage: Function that reads reported usage.

        Returns:
            The transport response.
        """
        self.validate(input_tokens, cancel)
        attempt = 0
        while True:
            charge = None
            if self.budget is not None:
                charge = await self.budget.start_call(
                    model=model, 
                    input_tokens=input_tokens, 
                    cancel=cancel
                )
            try:
                check_cancel(cancel)
            except BaseException:
                if charge is not None and self.budget is not None:
                    self.budget.release_call(charge=charge)
                raise
            try:
                response = await operation()
            except BaseException as error:
                if charge is not None:
                    self._settle(charge, error, usage, failed=True)
                if not isinstance(error, Exception):
                    raise
                delay = self.policy.delay(error, attempt)
                if delay is None:
                    raise
                attempt += 1
                await wait(delay, cancel)
                continue
            self._settle(charge, response, usage)
            return response

    def run_sync(self, op, *, model, input_tokens=None, cancel=None, usage):
        """Run a synchronous operation with per-attempt accounting.

        Args:
            op: Transport operation.
            model: Model name.
            input_tokens: Estimated input count when a budget is used.
            cancel: Optional cancellation signal.
            usage: Function that reads reported usage.

        Returns:
            The transport response.
        """
        self.validate(input_tokens, cancel)
        attempt = 0
        while True:
            charge = None
            if self.budget is not None:
                charge = self.budget.start_sync(
                    model=model, 
                    input_tokens=input_tokens, 
                    cancel=cancel
                )
            try:
                check_cancel(cancel)
            except BaseException:
                if charge is not None and self.budget is not None:
                    self.budget.release_call(charge=charge)
                raise
            try:
                response = op()
            except BaseException as error:
                if charge is not None:
                    self._settle(charge, error, usage, failed=True)
                if not isinstance(error, Exception):
                    raise
                delay = self.policy.delay(error, attempt)
                if delay is None:
                    raise
                attempt += 1
                wait_sync(delay, cancel)
                continue
            self._settle(charge, response, usage)
            return response

    def _settle(self, charge, response, usage, *, failed=False):
        if charge is None:
            return
        try:
            reported = usage(response)
            if self.budget is not None:
                self.budget.finish_call(charge=charge, usage=reported)
        except Exception:
            if self.budget is not None:
                self.budget.finish_call(charge=charge, usage=None)
            if not failed:
                raise


def make_request_runner(
    *,
    policy: RetryPolicy, 
    budget: SpendBudget | None = None,
) -> RequestRunner:
    try:
        return RequestRunner(policy=policy, budget=budget)
    except Exception as e:
        raise InitializationError("Failed to create RequestRunner.") from e
