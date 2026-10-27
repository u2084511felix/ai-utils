from concurrent.futures import ThreadPoolExecutor

from pydantic import ValidationError
import pytest

from aiutils import budget
from aiutils.budget import CallCharge, TokenUsage, make_spend_budget
from aiutils.exceptions import BudgetExceededError, InitializationError


@pytest.fixture
def linear_cost(monkeypatch):
    monkeypatch.setattr(budget, "calculate_token_cost", lambda model, input_tokens, output_tokens=0: input_tokens + output_tokens * 2)


def test_usage_defaults():
    """GIVEN a budget and specified token counts.
    WHEN the budget operation is called.
    THEN all token counts are zero."""
    result = TokenUsage().model_dump()
    expected = dict(input_tokens=0, output_tokens=0, cached_input_tokens=0, reasoning_tokens=0)
    assert result == expected


def test_usage_reasoning_boundary():
    """GIVEN a budget and specified token counts.
    WHEN the budget operation is called.
    THEN the reasoning count can equal the output count."""
    result = TokenUsage(output_tokens=3, reasoning_tokens=3).reasoning_tokens
    expected = 3
    assert result == expected


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens"])
def test_negative_usage_rejected(field):
    """GIVEN a negative token count.
    WHEN the budget operation is called.
    THEN a negative token count is rejected."""
    with pytest.raises(ValidationError) as exc:
        TokenUsage(**{field: -1})
    expected = ValidationError
    assert exc.type == expected


def test_reasoning_cannot_exceed_output():
    """GIVEN a reasoning count above the output count.
    WHEN the budget operation is called.
    THEN the excess reasoning count is rejected."""
    with pytest.raises(ValidationError, match="Reasoning tokens cannot exceed") as exc:
        TokenUsage(output_tokens=2, reasoning_tokens=3)
    expected = ValidationError
    assert exc.type == expected


def test_negative_charge_rejected():
    """GIVEN a negative charge.
    WHEN the budget operation is called.
    THEN a negative charge is rejected."""
    with pytest.raises(ValidationError) as exc:
        CallCharge(model="gpt-4o", input_tokens=-1)
    expected = ValidationError
    assert exc.type == expected


@pytest.mark.parametrize("limit", [None, 0, 10])
def test_budget_retains_limit(limit):
    """GIVEN a budget and specified token counts.
    WHEN the budget operation is called.
    THEN budget retains limit."""
    instance = make_spend_budget(limit)
    expected = limit
    assert instance.limit_usd == expected


@pytest.mark.parametrize("limit", [None, 0, 10])
def test_budget_starts_without_charges(limit):
    """GIVEN a budget and specified token counts.
    WHEN the budget operation is called.
    THEN budget starts without charges."""
    instance = make_spend_budget(limit)
    expected = 0
    assert instance.spent_usd == expected


def test_negative_limit_rejected():
    """GIVEN a negative budget limit.
    WHEN the budget operation is called.
    THEN the negative limit is rejected."""
    with pytest.raises(ValueError, match="cannot be negative") as exc:
        make_spend_budget(-0.01)
    expected = ValueError
    assert exc.type == expected


@pytest.fixture
def limited_budget(linear_cost):
    instance = make_spend_budget(10)
    return instance


@pytest.mark.parametrize("tokens", [10, 11])
def test_reservation_at_or_above_limit_is_rejected(limited_budget, tokens):
    """GIVEN a call cost at or above the budget limit.
    WHEN the budget operation is called.
    THEN the request is rejected and the balance does not change."""
    instance = limited_budget
    with pytest.raises(BudgetExceededError):
        instance.begin_call(model="test", input_tokens=tokens)
    expected = 0
    assert instance.spent_usd == expected


@pytest.fixture
def used_budget(linear_cost):
    instance = make_spend_budget()
    for _ in range(2):
        instance.begin_call(model="test", input_tokens=100)
    return instance


def test_unlimited_budget_returns_charge(used_budget):
    """GIVEN a budget and specified token counts.
    WHEN the budget operation is called.
    THEN unlimited budget returns charge."""
    charge = used_budget.begin_call(model="test", input_tokens=100)
    expected = 100
    assert charge.input_tokens == expected


def test_unlimited_budget_records_total(used_budget):
    """GIVEN a budget and specified token counts.
    WHEN the budget operation is called.
    THEN unlimited budget records total."""
    used_budget.begin_call(model="test", input_tokens=100)
    expected = 300
    assert used_budget.spent_usd == expected


@pytest.fixture
def reserved_budget(linear_cost, actual_input, output):
    instance = make_spend_budget(100)
    charge = instance.begin_call(model="test", input_tokens=10)
    usage = TokenUsage(input_tokens=actual_input, output_tokens=output)
    return (charge, instance, usage)


@pytest.mark.parametrize("actual_input,output,expected", [(2, 1, 4), (8, 3, 14), (0, 0, 0)])
def test_finish_replaces_estimate_with_actual_cost(reserved_budget, actual_input, output, expected):
    """GIVEN a budget and specified token counts.
    WHEN the budget operation is called.
    THEN the balance uses the actual token counts."""
    charge, instance, usage = reserved_budget
    instance.finish_call(charge=charge, usage=usage)
    assert instance.spent_usd == expected


@pytest.fixture
def exceeded_budget(linear_cost):
    instance = make_spend_budget(10)
    charge = instance.begin_call(model="test", input_tokens=2)
    instance.finish_call(charge=charge, usage=TokenUsage(input_tokens=2, output_tokens=5))
    return instance


def test_actual_usage_blocks_next_call(exceeded_budget):
    """GIVEN a budget whose actual usage exceeds its limit.
    WHEN the budget operation is called.
    THEN the next call is rejected and the excess charge is retained."""
    with pytest.raises(BudgetExceededError):
        exceeded_budget.begin_call(model="test", input_tokens=1)
    expected = 12
    assert exceeded_budget.spent_usd == expected


@pytest.fixture
def raced_budget(linear_cost):
    instance = make_spend_budget(20)

    def reserve(_):
        try:
            instance.begin_call(model="test", input_tokens=1)
            return True
        except BudgetExceededError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = list(pool.map(reserve, range(100)))
    return (instance, sum(accepted))


def test_concurrent_budget_limits_accepted_calls(raced_budget):
    """GIVEN a budget after concurrent reservation requests.
    WHEN the budget operation is called.
    THEN concurrent budget limits accepted calls."""
    instance, accepted = raced_budget
    with pytest.raises(BudgetExceededError):
        instance.begin_call(model="test", input_tokens=1)
    expected = 19
    assert accepted == expected


def test_concurrent_budget_records_accepted_charges(raced_budget):
    """GIVEN a budget after concurrent reservation requests.
    WHEN the budget operation is called.
    THEN concurrent budget records accepted charges."""
    instance, accepted = raced_budget
    with pytest.raises(BudgetExceededError):
        instance.begin_call(model="test", input_tokens=1)
    expected = 19
    assert instance.spent_usd == expected


@pytest.fixture
def failed_budget(monkeypatch):

    def fail(limit):
        raise ValueError("bad setup")

    monkeypatch.setattr(budget, "SpendBudget", fail)


def test_factory_wraps_initialization_errors(failed_budget):
    """GIVEN an operation that reports an error.
    WHEN the budget operation is called.
    THEN the error retains its original cause."""
    with pytest.raises(InitializationError, match="bad setup") as exc:
        make_spend_budget(1)
    result = isinstance(exc.value.__cause__, ValueError)
    expected = True
    assert result == expected
