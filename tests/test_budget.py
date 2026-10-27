from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from aiutils import budget
from aiutils.budget import CallCharge, TokenUsage, make_spend_budget
from aiutils.exceptions import BudgetExceededError, InitializationError


@pytest.fixture
def linear_cost(monkeypatch):
    # Use an independent, exact cost function to isolate reservation accounting.
    monkeypatch.setattr(budget, "calculate_token_cost", lambda model, input_tokens, output_tokens=0: input_tokens + output_tokens * 2)


def test_usage_defaults_and_reasoning_boundary():
    assert TokenUsage().model_dump() == dict(input_tokens=0, output_tokens=0, cached_input_tokens=0, reasoning_tokens=0)
    assert TokenUsage(output_tokens=3, reasoning_tokens=3).reasoning_tokens == 3


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens"])
def test_negative_usage_rejected(field):
    with pytest.raises(ValidationError):
        TokenUsage(**{field: -1})


def test_reasoning_cannot_exceed_output():
    with pytest.raises(ValidationError, match="Reasoning tokens cannot exceed"):
        TokenUsage(output_tokens=2, reasoning_tokens=3)


def test_negative_charge_rejected():
    with pytest.raises(ValidationError):
        CallCharge(model="gpt-4o", input_tokens=-1)


@pytest.mark.parametrize("limit", [None, 0, 10])
def test_budget_initial_state(limit):
    instance = make_spend_budget(limit)
    assert instance.limit_usd == limit
    assert instance.spent_usd == 0


def test_negative_limit_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        make_spend_budget(-0.01)


@pytest.mark.parametrize("tokens", [10, 11])
def test_reservation_at_or_above_limit_is_rejected(linear_cost, tokens):
    instance = make_spend_budget(10)
    with pytest.raises(BudgetExceededError):
        instance.begin_call(model="test", input_tokens=tokens)
    assert instance.spent_usd == 0


def test_unlimited_budget_accepts_multiple_calls(linear_cost):
    instance = make_spend_budget()
    for _ in range(3):
        assert instance.begin_call(model="test", input_tokens=100).input_tokens == 100
    assert instance.spent_usd == 300


@pytest.mark.parametrize("actual_input,output,expected", [(2, 1, 4), (8, 3, 14), (0, 0, 0)])
def test_finish_replaces_estimate_with_actual_cost(linear_cost, actual_input, output, expected):
    instance = make_spend_budget(100)
    charge = instance.begin_call(model="test", input_tokens=10)
    instance.finish_call(charge=charge, usage=TokenUsage(input_tokens=actual_input, output_tokens=output))
    assert instance.spent_usd == expected


def test_actual_usage_can_exceed_limit_and_blocks_next_call(linear_cost):
    instance = make_spend_budget(10)
    charge = instance.begin_call(model="test", input_tokens=2)
    instance.finish_call(charge=charge, usage=TokenUsage(input_tokens=2, output_tokens=5))
    assert instance.spent_usd == 12
    with pytest.raises(BudgetExceededError):
        instance.begin_call(model="test", input_tokens=1)


def test_reservations_are_atomic_between_threads(linear_cost):
    instance = make_spend_budget(20)
    def reserve(_):
        try:
            instance.begin_call(model="test", input_tokens=1)
            return True
        except BudgetExceededError:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = list(pool.map(reserve, range(100)))
    assert sum(accepted) == 19
    assert instance.spent_usd == 19


def test_factory_wraps_initialization_errors(monkeypatch):
    def fail(limit):
        raise ValueError("bad setup")
    monkeypatch.setattr(budget, "SpendBudget", fail)
    with pytest.raises(InitializationError, match="bad setup") as exc:
        make_spend_budget(1)
    assert isinstance(exc.value.__cause__, ValueError)
