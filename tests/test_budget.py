"""Check public budget contracts."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
import pytest_asyncio
from pydantic import ValidationError

from aiutils.budget import SpendBudget, TokenUsage, make_budget
from aiutils.exceptions import BudgetExceededError, CallCancelledError, ReservationError


class Clock:
    def __init__(self):
        self.now = 0.0
        self.waits = []
        self.action = None

    def __call__(self):
        return self.now

    async def wait(self, seconds, cancel=None):
        self.wait_sync(seconds, cancel)
        await asyncio.sleep(0)

    def wait_sync(self, seconds, cancel=None):
        self.waits.append(seconds)
        self.now += seconds
        if self.action is not None:
            self.action()


@pytest.fixture
def clock(monkeypatch):
    timer = Clock()
    monkeypatch.setattr("aiutils.budget.time", SimpleNamespace(monotonic=timer))
    monkeypatch.setattr("aiutils.budget.wait", timer.wait)
    monkeypatch.setattr("aiutils.budget.wait_sync", timer.wait_sync)
    return timer


@pytest.fixture
def budget(clock):
    return make_budget(
        token_limits={"gpt-5.2": 10}, 
        window_seconds=1, 
    )


@pytest.fixture
def reserved(budget):
    charge = budget.start_sync(model="gpt-5.2", input_tokens=10)
    return budget, charge


@pytest.fixture
def settled(reserved):
    budget, charge = reserved
    budget.finish_call(charge=charge, usage=TokenUsage(input_tokens=5, output_tokens=2))
    return budget, charge


@pytest.fixture
def released(reserved):
    budget, charge = reserved
    budget.release_call(charge=charge)
    return budget, charge


@pytest.fixture
def cancelled(clock, reserved):
    signal = Event()
    clock.action = signal.set
    return reserved[0], signal


@pytest.fixture
def rejected(clock):
    budget = make_budget(0.0001, token_limits={"gpt-5.2": 10}, window_seconds=1)
    charge = budget.start_sync(model="gpt-5.2", input_tokens=10)
    clock.action = lambda: budget.finish_call(charge=charge, usage=TokenUsage(input_tokens=10, output_tokens=10))
    def settle():
        clock.action = None
        budget.finish_call(charge=charge, usage=TokenUsage(input_tokens=10, output_tokens=10))
    clock.action = settle
    return budget


@pytest.mark.parametrize("limit", [None, 0, 10])
def test_factory_returns_public_contract(limit):
    """GIVEN a spend limit. WHEN a budget is created. THEN its configuration is public."""
    budget: SpendBudget = make_budget(limit)
    assert budget.limit_usd == limit


@pytest.mark.parametrize("limit", [-1, float("nan"), float("inf"), True, "1"])
def test_factory_rejects_invalid_limits(limit):
    """GIVEN an invalid limit. WHEN a budget is created. THEN creation fails."""
    with pytest.raises(ValueError) as error:
        make_budget(limit)
    expected = ValueError
    assert error.type is expected


@pytest.mark.parametrize("limits", [{"": 1}, {" ": 1}, {"gpt-5.2": 0}, {"gpt-5.2": True}, {"gpt-5.2": 1.5}, {"GPT-5.2": 1, " gpt-5.2 ": 2}])
def test_factory_rejects_invalid_capacity(limits):
    """GIVEN invalid capacity. WHEN a budget is created. THEN creation fails."""
    with pytest.raises(ValueError) as error:
        make_budget(token_limits=limits)
    expected = ValueError
    assert error.type is expected


@pytest.mark.parametrize("window", [0, -1, float("nan"), float("inf"), True])
def test_factory_rejects_invalid_windows(window):
    """GIVEN an invalid window. WHEN a budget is created. THEN creation fails."""
    with pytest.raises(ValueError) as error:
        make_budget(window_seconds=window)
    expected = ValueError
    assert error.type is expected


@pytest.mark.parametrize("field,value", [("input_tokens", -1), ("input_tokens", True), ("output_tokens", 1.5), ("cached_input_tokens", 1), ("reasoning_tokens", 1)])
def test_usage_rejects_invalid_counts(field, value):
    """GIVEN invalid counts. WHEN usage is created. THEN validation fails."""
    with pytest.raises(ValidationError) as error:
        TokenUsage(**{field: value})
    expected = ValidationError
    assert error.type is expected


@pytest.mark.parametrize("count", [-1, True, 1.5, float("nan"), 11])
@pytest.mark.asyncio
async def test_reservation_rejects_invalid_input(budget, count):
    """GIVEN invalid input. WHEN capacity is requested. THEN spend does not change."""
    with pytest.raises(ValueError):
        await budget.start_call(model="gpt-5.2", input_tokens=count)
    expected = 0
    assert budget.spent_usd == expected


@pytest.mark.asyncio
async def test_capacity_rejects_unconfigured_models(budget):
    """GIVEN model limits. WHEN an unconfigured model is used. THEN the call fails."""
    with pytest.raises(ValueError) as error:
        await budget.start_call(model="gpt-4o", input_tokens=1)
    expected = ValueError
    assert error.type is expected


@pytest.mark.asyncio
async def test_window_expiry_permits_next_call(reserved, clock):
    """GIVEN full capacity. WHEN the window expires. THEN a new reservation succeeds."""
    charge = await reserved[0].start_call(model="gpt-5.2", input_tokens=10)
    expected = 10
    assert charge.input_tokens == expected


@pytest.mark.asyncio
async def test_wait_uses_injected_clock(reserved, clock):
    """GIVEN full capacity. WHEN a caller waits. THEN the window clock advances."""
    await reserved[0].start_call(model="gpt-5.2", input_tokens=10)
    expected = 1
    assert clock.now == pytest.approx(expected)


@pytest.mark.asyncio
async def test_cancel_during_capacity_wait(cancelled):
    """GIVEN full capacity. WHEN cancellation occurs in a wait. THEN no charge is added."""
    budget, signal = cancelled
    with pytest.raises(CallCancelledError):
        await budget.start_call(model="gpt-5.2", input_tokens=1, cancel=signal)
    expected = 0.0000175
    assert budget.spent_usd == pytest.approx(expected)


@pytest.mark.asyncio
async def test_spend_is_checked_after_wait(rejected):
    """GIVEN a waiting caller. WHEN spend reaches the limit. THEN no reservation is added."""
    with pytest.raises(BudgetExceededError):
        await rejected.start_call(model="gpt-5.2", input_tokens=10)
    expected = 0.0001575
    assert rejected.spent_usd == pytest.approx(expected)


@pytest.mark.parametrize("usage,expected", [(TokenUsage(input_tokens=5, output_tokens=2), 0.00003675), (None, 0.0000175)])
def test_settlement_replaces_estimate(reserved, usage, expected):
    """GIVEN a reservation. WHEN it is settled. THEN its charge follows the usage policy."""
    budget, charge = reserved
    budget.finish_call(charge=charge, usage=usage)
    assert budget.spent_usd == pytest.approx(expected)


def test_cached_usage_changes_confirmed_spend(reserved):
    """GIVEN cached input. WHEN the call completes. THEN cached tokens use their rate."""
    budget, charge = reserved
    budget.finish_call(charge=charge, usage=TokenUsage(input_tokens=10, cached_input_tokens=5))
    expected = 0.000009625
    assert budget.confirmed_usd == pytest.approx(expected)


def test_unknown_usage_stays_unresolved(reserved):
    """GIVEN unknown usage. WHEN a call completes. THEN its estimate is unresolved."""
    budget, charge = reserved
    budget.finish_call(charge=charge, usage=None)
    expected = charge.estimate_usd
    assert budget.unresolved_usd == expected


def test_release_refunds_pending_estimate(reserved):
    """GIVEN an unused reservation. WHEN it is released. THEN estimated spend is zero."""
    budget, charge = reserved
    budget.release_call(charge=charge)
    expected = 0
    assert budget.estimated_usd == expected


@pytest.mark.asyncio
async def test_release_restores_capacity(released, clock):
    """GIVEN released capacity. WHEN a new call starts. THEN it does not wait."""
    await released[0].start_call(model="gpt-5.2", input_tokens=10)
    expected = []
    assert clock.waits == expected


@pytest.mark.parametrize("state", ["settled", "released"])
def test_repeated_settlement_fails(request, state):
    """GIVEN a settled reservation. WHEN it is completed again. THEN it is rejected."""
    budget, charge = request.getfixturevalue(state)
    with pytest.raises(ReservationError) as error:
        budget.finish_call(charge=charge, usage=None)
    expected = ReservationError
    assert error.type is expected


@pytest.mark.parametrize("state", ["settled", "released"])
def test_repeated_release_fails(request, state):
    """GIVEN a settled reservation. WHEN it is released again. THEN it is rejected."""
    budget, charge = request.getfixturevalue(state)
    with pytest.raises(ReservationError) as error:
        budget.release_call(charge=charge)
    expected = ReservationError
    assert error.type is expected


@pytest.fixture
def foreign(reserved):
    return make_budget(), reserved[1]


def test_foreign_reservation_is_rejected(foreign):
    """GIVEN another budget's reservation. WHEN it is settled. THEN it is rejected."""
    budget, charge = foreign
    with pytest.raises(ReservationError) as error:
        budget.finish_call(charge=charge, usage=None)
    expected = ReservationError
    assert error.type is expected


@pytest.mark.parametrize("name", ["spent_usd", "limit_usd", "confirmed_usd", "estimated_usd", "token_limits", "window_seconds"])
def test_accounting_properties_are_read_only(budget, name):
    """GIVEN a budget. WHEN public state is assigned. THEN assignment fails."""
    with pytest.raises(AttributeError) as error:
        setattr(budget, name, 0)
    expected = AttributeError
    assert error.type is expected


@pytest.fixture
def raced():
    budget = make_budget(0.00001)
    def reserve(_):
        try:
            budget.start_sync(model="gpt-5.2", input_tokens=1)
        except BudgetExceededError:
            pass
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(reserve, range(100)))
    return budget


def test_threads_share_one_spend_limit(raced):
    """GIVEN concurrent reservations. WHEN another call starts. THEN the shared limit holds."""
    with pytest.raises(BudgetExceededError):
        raced.start_sync(model="gpt-5.2", input_tokens=1)
    expected = 0.00000875
    assert raced.spent_usd == pytest.approx(expected)


@pytest.mark.parametrize("tokens,expected", [(3, 0), (4, 1)])
@pytest.mark.asyncio
async def test_reported_usage_updates_capacity(settled, clock, tokens, expected):
    """GIVEN settled usage. WHEN capacity is requested. THEN reported totals set the wait."""
    await settled[0].start_call(model="gpt-5.2", input_tokens=tokens)
    assert clock.now == pytest.approx(expected)


@pytest.fixture
def pre_cancelled():
    signal = Event()
    signal.set()
    return signal


@pytest.mark.asyncio
async def test_cancellation_before_reservation_preserves_capacity(budget, pre_cancelled):
    """GIVEN cancellation before a call. WHEN capacity is requested. THEN no charge is added."""
    with pytest.raises(CallCancelledError):
        await budget.start_call(model="gpt-5.2", input_tokens=1, cancel=pre_cancelled)
    expected = 0
    assert budget.spent_usd == expected


@pytest.fixture
def reached_limit():
    return make_budget(0.0000175)


@pytest.mark.asyncio
async def test_reservation_at_limit_is_rejected(reached_limit):
    """GIVEN a call at the spend limit. WHEN it starts. THEN the reservation is rejected."""
    with pytest.raises(BudgetExceededError):
        await reached_limit.start_call(model="gpt-5.2", input_tokens=10)
    expected = 0
    assert reached_limit.spent_usd == expected


@pytest_asyncio.fixture
async def shared_capacity(budget):
    await asyncio.gather(
        budget.start_call(model="gpt-5.2", input_tokens=6),
        budget.start_call(model="gpt-5.2", input_tokens=6),
    )
    return budget


@pytest.mark.asyncio
async def test_concurrent_callers_share_capacity(shared_capacity, clock):
    """GIVEN concurrent callers. WHEN another reservation starts. THEN shared windows apply."""
    await shared_capacity.start_call(model="gpt-5.2", input_tokens=5)
    expected = 2
    assert clock.now == pytest.approx(expected)
