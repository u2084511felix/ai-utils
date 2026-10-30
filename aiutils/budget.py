"""Shared token capacity and model spend limits."""

from dataclasses import dataclass
from threading import Lock
import time
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict, 
    Field, 
    model_validator,
    NonNegativeInt
)
from .control import (
    Cancellation, wait, wait_sync, check_cancel, check_finite_number, check_non_negative_int
)
from .exceptions import BudgetExceededError, ReservationError
from .pricing import calculate_token_cost, get_price


class TokenUsage(BaseModel):
    """Token counts reported for one model call.

    Attributes:
        input_tokens: Total input tokens, including cached tokens.
        output_tokens: Total output tokens, including reasoning tokens.
        cached_input_tokens: Cached tokens within the input count.
        reasoning_tokens: Reasoning tokens within the output count.
    """
    model_config = ConfigDict(frozen=True, strict=True)
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    cached_input_tokens: NonNegativeInt = 0
    reasoning_tokens: NonNegativeInt = 0

    @model_validator(mode="after")
    def check_counts(self) -> Self:
        """Check that token detail counts do not exceed their totals.

        Returns:
            The checked usage.

        Raises:
            ValueError: A detail count exceeds its total.
        """
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("Reasoning tokens cannot exceed output tokens.")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("Cached input tokens cannot exceed input tokens.")
        return self


@dataclass(frozen=True, eq=False)
class Reservation:
    """An input-cost reservation for one transport attempt.

    Attributes:
        model: Normalized model name.
        input_tokens: Estimated input tokens.
        estimate_usd: Reserved input cost in US dollars.
    """
    model: str
    input_tokens: int
    estimate_usd: float


class SpendBudget(Protocol):
    """Public contract for shared token capacity and spend accounting."""

    @property
    def limit_usd(self) -> float | None:
        """Return the configured spend limit."""
        ...

    @property
    def token_limits(self) -> Mapping[str, int]:
        """Return the immutable per-model token limits."""
        ...

    @property
    def window_seconds(self) -> float:
        """Return the token window duration."""
        ...

    @property
    def spent_usd(self) -> float:
        """Return confirmed cost plus all estimates."""
        ...

    @property
    def confirmed_usd(self) -> float:
        """Return cost from reported usage."""
        ...

    @property
    def estimated_usd(self) -> float:
        """Return pending and unresolved estimates."""
        ...

    @property
    def unresolved_usd(self) -> float:
        """Return estimates retained for unknown billing."""
        ...


    async def start_call(self, *, model: str, input_tokens: int,
                         cancel: Cancellation | None = None) -> Reservation:
        """Reserve capacity and input cost for an async call."""
        ...

    def start_sync(self, *, model: str, input_tokens: int,
                   cancel: Cancellation | None = None) -> Reservation:
        """Reserve capacity and input cost for a sync call."""
        ...

    def finish_call(self, *, charge: Reservation, usage: TokenUsage | None) -> None:
        """Settle one reservation with reported usage or an unresolved estimate."""
        ...

    def release_call(self, *, charge: Reservation) -> None:
        """Release one reservation before its transport attempt starts."""
        ...


@dataclass(eq=False)
class _Capacity:
    expires: float
    tokens: int


class _SpendBudget:
    def __init__(self, limit_usd, token_limits, window_seconds):
        self._limit = limit_usd
        self._limits = MappingProxyType(token_limits)
        self._window = window_seconds
        self._lock = Lock()
        self._confirmed = 0.0
        self._unresolved = 0.0
        self._pending: dict[Reservation, _Capacity] = {}
        self._capacity: dict[str, list[_Capacity]] = {name: [] for name in token_limits}

    @property
    def limit_usd(self):
        return self._limit

    @property
    def token_limits(self):
        return self._limits

    @property
    def window_seconds(self):
        return self._window

    @property
    def spent_usd(self):
        with self._lock:
            return self._total()

    @property
    def confirmed_usd(self):
        with self._lock:
            return self._confirmed

    @property
    def estimated_usd(self):
        with self._lock:
            return self._unresolved + sum(charge.estimate_usd for charge in self._pending)

    @property
    def unresolved_usd(self):
        with self._lock:
            return self._unresolved

    def _total(self):
        return self._confirmed + self._unresolved + sum(charge.estimate_usd for charge in self._pending)

    def _prepare(self, model, input_tokens):
        check_non_negative_int(input_tokens)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string.")
        model = model.strip().lower()
        if self._limits:
            if model not in self._limits:
                raise ValueError(f"No token limit is configured for model '{model}'.")
            if input_tokens > self._limits[model]:
                raise ValueError("input_tokens exceeds the model window limit.")
        return Reservation(model, input_tokens, calculate_token_cost(model, input_tokens))

    def _reserve(self, charge, cancel):
        check_cancel(cancel)
        with self._lock:
            check_cancel(cancel)
            now = time.monotonic()
            entries = self._capacity.get(charge.model)
            if entries is not None:
                entries[:] = [item for item in entries if item.expires > now]
                used = sum(item.tokens for item in entries)
                if used + charge.input_tokens > self._limits[charge.model]:
                    return min(0.05, min(item.expires for item in entries) - now)
            if self._limit is not None and self._total() + charge.estimate_usd >= self._limit:
                raise BudgetExceededError(f"Run spend limit of {self._limit:.2f} USD was reached.")
            capacity = _Capacity(now + self._window, charge.input_tokens)
            self._pending[charge] = capacity
            if entries is not None:
                entries.append(capacity)
            return None

    async def start_call(self, *, model, input_tokens, cancel=None):
        """Reserve capacity and estimated spend before a call.

        Args:
            model: Model name.
            input_tokens: Estimated input count.
            cancel: Optional cancellation signal.

        Returns:
            A reservation for one transport attempt.

        Raises:
            ValueError: An input is invalid.
            BudgetExceededError: The reservation reaches the spend limit.
        """
        check_cancel(cancel)
        charge = self._prepare(model, input_tokens)
        while (delay := self._reserve(charge, cancel)) is not None:
            await wait(delay, cancel)
        return charge

    def start_sync(self, *, model, input_tokens, cancel=None):
        """Reserve capacity and spend for a synchronous call.

        Args:
            model: Model name.
            input_tokens: Estimated input count.
            cancel: Optional cancellation signal.

        Returns:
            A reservation for one transport attempt.
        """
        check_cancel(cancel)
        charge = self._prepare(model, input_tokens)
        while (delay := self._reserve(charge, cancel)) is not None:
            wait_sync(delay, cancel)
        return charge

    def _charge(self, charge):
        if not isinstance(charge, Reservation) or charge not in self._pending:
            raise ReservationError("Reservation is foreign or has already been settled.")
        return self._pending[charge]

    def finish_call(self, *, charge, usage):
        """Settle a call with reported usage or retain an unresolved estimate.

        Args:
            charge: Reservation returned for this budget.
            usage: Reported usage, or None when billing is unknown.

        Raises:
            ReservationError: The reservation is foreign or settled.
            ValueError: The usage or price is invalid.
        """
        with self._lock:
            capacity = self._charge(charge)
            if usage is None:
                self._unresolved += charge.estimate_usd
            else:
                if not isinstance(usage, TokenUsage):
                    raise TypeError("usage must be TokenUsage or None.")
                cost = calculate_token_cost(
                    charge.model, 
                    usage.input_tokens,
                    usage.output_tokens, 
                    usage.cached_input_tokens
                )
                self._confirmed += cost
                capacity.tokens = usage.input_tokens + usage.output_tokens
            del self._pending[charge]

    def release_call(self, *, charge):
        """Release a reservation before its transport attempt starts.

        Args:
            charge: Reservation returned for this budget.

        Raises:
            ReservationError: The reservation is foreign or settled.
        """
        with self._lock:
            capacity = self._charge(charge)
            entries = self._capacity.get(charge.model, [])
            if capacity in entries:
                entries.remove(capacity)
            del self._pending[charge]


def make_budget(
    limit_usd: float | None = None, 
    *, 
    token_limits: Mapping[str, int] | None = None,
    window_seconds: float = 60, 
) -> SpendBudget:
    """Create a shared budget with optional per-model token limits.

    Args:
        limit_usd: Spend limit, or None for no limit.
        token_limits: Positive token counts per model window.
        window_seconds: Positive finite window duration.

    Returns:
        A budget that satisfies the public SpendBudget contract.

    Raises:
        ValueError: Configuration is invalid.
        NotImplementedError: A configured model has no price.
    """
    if limit_usd is not None:
        check_finite_number(limit_usd, "limit_usd")
    check_finite_number(window_seconds, "window_seconds")
    if window_seconds == 0:
        raise ValueError("window_seconds must be positive.")
    if token_limits is not None and not isinstance(token_limits, Mapping):
        raise TypeError("token_limits must be a mapping.")
    limits = {}
    for name, count in (token_limits or {}).items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Model names must be nonempty strings.")
        model = name.strip().lower()
        if model in limits:
            raise ValueError(f"Duplicate model name: {model}.")
        check_non_negative_int(count, "token limit")
        if count == 0:
            raise ValueError("Token limits must be positive.")
        get_price(model)
        limits[model] = count
    return _SpendBudget(limit_usd, limits, window_seconds)
