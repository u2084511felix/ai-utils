"""Thread-safe model-spend limits for one pipeline run."""
from dataclasses import dataclass
from threading import Lock
from typing import Self


from pydantic import BaseModel, NonNegativeInt, model_validator


from .exceptions import BudgetExceededError
from .pricing import calculate_token_cost



class TokenUsage(BaseModel):
    """Token counts reported for one model call.

    Args:
        input_tokens: Tokens sent to the model.
        output_tokens: Completion tokens, including reasoning tokens.
        cached_input_tokens: Cached input tokens from previous calls.
        reasoning_tokens: Reasoning tokens within ``output_tokens``.
    """
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    cached_input_tokens: NonNegativeInt = 0
    reasoning_tokens: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_reasoning_tokens(self) -> Self:
        """Ensure reasoning tokens are included in output tokens.

        Raises:
            ValueError: Reasoning tokens exceed output tokens.
        """
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("Reasoning tokens cannot exceed output tokens.")
        return self


class CallCharge(BaseModel):
    """Input-cost reservation for one model call.

    Args:
        model: Model used for the call.
        input_tokens: Estimated input tokens reserved before the call.
    """
    model: str
    input_tokens: NonNegativeInt


class SpendBudget:
    """Reserve estimated call cost and stop new calls at a USD limit."""

    def __init__(self, limit_usd: float | None):
        """Create one run budget.

        Args:
            limit_usd: Maximum run spend, or ``None`` for no limit.

        Raises:
            ValueError: The limit is negative.
        """
        if limit_usd is not None and limit_usd < 0:
            raise ValueError("The run spend limit cannot be negative.")
        self.limit_usd = limit_usd
        self.spent_usd = 0.0
        self._lock = Lock()

    def begin_call(self, *, model: str, input_tokens: int) -> CallCharge:
        """Reserve input cost or reject the call before it starts.

        Args:
            model: Model used for the call.
            input_tokens: Estimated input tokens for the request.

        Returns:
            The reservation to reconcile after the provider responds.

        Raises:
            BudgetExceededError: The reservation reaches the run limit.
        """
        estimate = calculate_token_cost(model, input_tokens)
        with self._lock:
            if self.limit_usd is not None and self.spent_usd + estimate >= self.limit_usd:
                raise BudgetExceededError(f"Run spend limit of {self.limit_usd:.2f} USD was reached.")
            self.spent_usd += estimate
        return CallCharge(model=model, input_tokens=input_tokens)

    def finish_call(self, *, charge: CallCharge, usage: TokenUsage) -> None:
        """Replace a reservation with provider-reported cost.

        Args:
            charge: Reservation returned before the call.
            usage: Token counts reported by the provider.
        """
        reserved_cost = calculate_token_cost(charge.model, charge.input_tokens)
        actual_cost = calculate_token_cost(
            charge.model,
            usage.input_tokens,
            usage.output_tokens,
        )
        with self._lock:
            self.spent_usd += actual_cost - reserved_cost
