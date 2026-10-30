"""Check retries through public clients and mock HTTP transports."""

import asyncio
import json
from threading import Event

import httpx2
import pytest
from openai import AsyncOpenAI, OpenAI

from aiutils.budget import make_budget
from aiutils.clients import ClientReply, make_client
from aiutils.exceptions import CallCancelledError, ClientError, InitializationError
from aiutils.generate import make_generate
from aiutils.retries import make_retry_policy


class Waiter:
    def __init__(self):
        self.delays = []
        self.action = None

    def wait_sync(self, seconds, cancel=None):
        self.delays.append(seconds)
        if self.action is not None:
            self.action()

    async def wait(self, seconds, cancel=None):
        self.wait_sync(seconds, cancel)
        await asyncio.sleep(0)


@pytest.fixture
def payload():
    return {
        "id": "chat-test", "object": "chat.completion", "created": 0, "model": "gpt-5.2",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                  "prompt_tokens_details": {"cached_tokens": 4},
                  "completion_tokens_details": {"reasoning_tokens": 2}},
    }


@pytest.fixture
def waiter(monkeypatch):
    recorder = Waiter()
    monkeypatch.setattr("aiutils.retries.wait", recorder.wait)
    monkeypatch.setattr("aiutils.retries.wait_sync", recorder.wait_sync)
    return recorder


@pytest.fixture
def signal():
    return Event()


@pytest.fixture
def params():
    return {"model": "gpt-5.2", "messages": [{"role": "user", "content": "hello"}]}


@pytest.fixture
def transport(payload, failure):
    calls = []
    def handle(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1 or failure == "exhaust":
            if failure == "timeout":
                raise httpx2.ReadTimeout("timeout", request=request)
            if failure == "connection":
                raise httpx2.ConnectError("connection", request=request)
            status = 429 if failure in ("rate", "long", "invalid-delay", "date") else 401 if failure == "auth" else 503
            header = "31" if failure == "long" else "invalid" if failure == "invalid-delay" else "3"
            if failure == "date":
                header = "Thu, 29 Oct 2099 12:00:00 GMT"
            return httpx2.Response(status, headers={"Retry-After": header}, json={"error": {"message": "provider failed"}, **({"usage": payload["usage"]} if failure == "known" else {})})
        return httpx2.Response(200, json=payload)
    return httpx2.MockTransport(handle), calls


@pytest.fixture
def retry_client(transport, waiter):
    mock, calls = transport
    sync = OpenAI(api_key="test-key", max_retries=0, http_client=httpx2.Client(transport=mock))
    asynchronous = AsyncOpenAI(api_key="test-key", max_retries=0, http_client=httpx2.AsyncClient(transport=mock))
    asynchronous._platform = "Linux"
    client = make_client(client=sync, async_client=asynchronous, budget=make_budget())
    yield client, calls
    sync.close()
    asyncio.run(asynchronous.close())


@pytest.mark.parametrize("failure", ["timeout", "connection", "rate", "server", "invalid-delay"])
@pytest.mark.asyncio
async def test_transient_failure_then_success(retry_client, params, failure):
    """GIVEN one transient failure. WHEN a request runs. THEN the retry returns text."""
    reply = await retry_client[0].async_chat_completions(params, input_tokens=10)
    expected = "hello"
    assert reply.content == expected


@pytest.mark.parametrize("failure", ["timeout", "connection", "rate", "server"])
def test_sync_transient_failure_then_success(retry_client, params, failure):
    """GIVEN one transient failure. WHEN a synchronous request runs. THEN it retries."""
    reply = retry_client[0].chat_completions(params, input_tokens=10)
    expected = "hello"
    assert reply.content == expected


@pytest.mark.parametrize("failure", ["rate"])
@pytest.mark.asyncio
async def test_provider_delay_is_honored(retry_client, params, waiter, failure):
    """GIVEN a valid Retry-After value. WHEN a retry runs. THEN it waits as requested."""
    await retry_client[0].async_chat_completions(params, input_tokens=10)
    expected = [3]
    assert waiter.delays == expected


@pytest.mark.parametrize("failure,expected", [("auth", 1), ("long", 1), ("date", 1), ("exhaust", 3)])
@pytest.mark.asyncio
async def test_retry_limits_stop_transport_calls(retry_client, params, failure, expected):
    """GIVEN a permanent error or retry limit. WHEN a request runs. THEN attempts stop."""
    await retry_client[0].async_chat_completions(params, input_tokens=10)
    assert len(retry_client[1]) == expected


@pytest.mark.parametrize("failure", ["exhaust"])
@pytest.mark.asyncio
async def test_exhausted_failure_retains_status(retry_client, params, failure):
    """GIVEN exhausted retries. WHEN a request fails. THEN the final status is retained."""
    reply = await retry_client[0].async_chat_completions(params, input_tokens=10)
    expected = 503
    assert reply.error.status_code == expected


@pytest.mark.parametrize("failure", ["server"])
@pytest.mark.asyncio
async def test_each_attempt_has_separate_accounting(retry_client, params, failure):
    """GIVEN a failed attempt then success. WHEN a call ends. THEN both charges remain."""
    client, _ = retry_client
    await client.async_chat_completions(params, input_tokens=10)
    expected = 0.0000987
    assert client.budget.spent_usd == pytest.approx(expected)


@pytest.mark.parametrize("failure", ["server"])
@pytest.mark.asyncio
async def test_failed_attempt_retains_unresolved_estimate(retry_client, params, failure):
    """GIVEN a failure without usage. WHEN a retry succeeds. THEN the estimate remains."""
    client, _ = retry_client
    await client.async_chat_completions(params, input_tokens=10)
    expected = 0.0000175
    assert client.budget.unresolved_usd == pytest.approx(expected)


@pytest.fixture
def cancelling(retry_client, waiter, signal):
    waiter.action = signal.set
    return retry_client


@pytest.mark.parametrize("failure", ["server"])
@pytest.mark.asyncio
async def test_cancellation_stops_retry_wait(cancelling, params, signal, failure):
    """GIVEN a retry wait. WHEN cancellation is set. THEN no second attempt starts."""
    client, calls = cancelling
    with pytest.raises(CallCancelledError):
        await client.async_chat_completions(params, input_tokens=10, cancel=signal)
    expected = 1
    assert len(calls) == expected


@pytest.fixture
def presignalled(signal):
    signal.set()
    return signal


@pytest.mark.parametrize("failure", ["server"])
@pytest.mark.asyncio
async def test_cancellation_prevents_reservation(retry_client, params, presignalled, failure):
    """GIVEN cancellation before a call. WHEN a request runs. THEN no estimate is added."""
    client, _ = retry_client
    with pytest.raises(CallCancelledError):
        await client.async_chat_completions(params, input_tokens=10, cancel=presignalled)
    expected = 0
    assert client.budget.spent_usd == expected


@pytest.fixture
def before_transport(retry_client, signal):
    client, calls = retry_client
    budget = client.budget
    class CancelBudget:
        async def start_call(self, **kwargs):
            charge = await budget.start_call(**kwargs)
            signal.set()
            return charge
        def release_call(self, **kwargs):
            return budget.release_call(**kwargs)
    client.budget = CancelBudget()
    return client, calls, budget


@pytest.mark.parametrize("failure", ["server"])
@pytest.mark.asyncio
async def test_cancellation_releases_unused_reservation(before_transport, params, signal, failure):
    """GIVEN a reserved call. WHEN cancellation precedes transport. THEN spend is released."""
    client, _, budget = before_transport
    with pytest.raises(CallCancelledError):
        await client.async_chat_completions(params, input_tokens=10, cancel=signal)
    expected = 0
    assert budget.spent_usd == expected


@pytest.fixture
def generator(retry_client):
    return make_generate(client=retry_client[0])


@pytest.mark.parametrize("failure", ["server"])
@pytest.mark.asyncio
async def test_retries_repeat_prepared_history(generator, retry_client, failure):
    """GIVEN a retry. WHEN generation runs. THEN both requests have the same history."""
    await generator.generate(system_msg="system", prompt="question", input_tokens=10)
    first, second = retry_client[1]
    assert first == second


@pytest.mark.parametrize("failure", ["server"])
@pytest.mark.asyncio
async def test_generation_does_not_add_retry_loop(generator, retry_client, failure):
    """GIVEN client retries. WHEN generation runs. THEN one owner makes two attempts."""
    await generator.generate(system_msg="system", prompt="question", input_tokens=10)
    expected = 2
    assert len(retry_client[1]) == expected


@pytest.mark.parametrize("failure", ["auth"])
@pytest.mark.asyncio
async def test_generation_retains_provider_message(generator, failure):
    """GIVEN a permanent provider error. WHEN generation runs. THEN its message is retained."""
    with pytest.raises(ClientError) as error:
        await generator.generate(system_msg="system", prompt="question", input_tokens=10)
    expected = "provider failed"
    assert expected in str(error.value)


@pytest.mark.parametrize("failure", ["auth"])
@pytest.mark.asyncio
async def test_generation_retains_provider_headers(generator, failure):
    """GIVEN a provider failure. WHEN generation runs. THEN response headers are retained."""
    with pytest.raises(ClientError) as error:
        await generator.generate(system_msg="system", prompt="question", input_tokens=10)
    expected = "3"
    assert error.value.headers["retry-after"] == expected


@pytest.fixture
def nested_clients(mock_sync_client, mock_async_client):
    mock_async_client.max_retries = 2
    return dict(client=mock_sync_client, async_client=mock_async_client)


def test_injected_sdk_retries_are_rejected(nested_clients):
    """GIVEN SDK retries enabled. WHEN a client is created. THEN construction fails."""
    with pytest.raises(ValueError) as error:
        make_client(**nested_clients)
    expected = ValueError
    assert error.type is expected


@pytest.mark.parametrize("options", [{"retries": -1}, {"retries": True}, {"retries": 1.5}, {"min_wait": float("nan")}, {"max_wait": float("inf")}, {"min_wait": 2, "max_wait": 1}])
def test_invalid_retry_policy_is_rejected(options):
    """GIVEN invalid retry limits. WHEN a policy is created. THEN construction fails."""
    with pytest.raises(InitializationError) as error:
        make_retry_policy(**options)
    expected = InitializationError
    assert error.type is expected


def test_success_reply_requires_explicit_usage():
    """GIVEN no usage report. WHEN a successful reply is created. THEN it is rejected."""
    with pytest.raises(ValueError) as error:
        ClientReply(content="hello")
    expected = ValueError
    assert error.type is expected


@pytest.fixture
def dated_error():
    from datetime import datetime, timezone
    from types import SimpleNamespace as NS
    from openai import APIStatusError
    response = NS(request=NS(), status_code=429, headers={"retry-after": "Thu, 29 Oct 2026 12:00:03 GMT"})
    error = APIStatusError("rate limit", response=response, body=None)
    policy = make_retry_policy(max_wait=10)
    now = lambda: datetime(2026, 10, 29, 12, 0, 0, tzinfo=timezone.utc)
    return policy, error, now


def test_http_date_delay_is_honored(dated_error):
    """GIVEN an HTTP date. WHEN retry delay is calculated. THEN the clock sets the delay."""
    policy, error, now = dated_error
    result = policy.delay(error, 0, now=now)
    expected = 3
    assert result == expected


@pytest.fixture
def jitter_error():
    return make_retry_policy(min_wait=2, max_wait=3), TimeoutError("timeout")


def test_exponential_delay_has_bounded_jitter(jitter_error):
    """GIVEN a transient failure. WHEN a delay is calculated. THEN jitter uses the wait bound."""
    policy, error = jitter_error
    result = policy.delay(error, 1, jitter=lambda: 0.5)
    expected = 1.5
    assert result == expected


@pytest.fixture
def permanent_error():
    return make_retry_policy(), ValueError("invalid schema")


def test_validation_errors_have_no_retry(permanent_error):
    """GIVEN invalid input. WHEN a retry is considered. THEN no retry is permitted."""
    policy, error = permanent_error
    result = policy.delay(error, 0)
    expected = None
    assert result is expected


@pytest.fixture
def mutable_request(retry_client, waiter, params):
    waiter.action = lambda: params["messages"].append({"role": "user", "content": "later"})
    return retry_client, params


@pytest.mark.parametrize("failure", ["server"])
@pytest.mark.asyncio
async def test_retry_request_is_independent_of_caller_mutation(mutable_request, failure):
    """GIVEN caller mutation during a retry wait. WHEN the call runs. THEN requests stay equal."""
    (client, calls), params = mutable_request
    await client.async_chat_completions(params, input_tokens=10)
    first, second = calls
    assert first == second


@pytest.mark.parametrize("failure", ["known"])
@pytest.mark.asyncio
async def test_failed_attempt_with_usage_is_confirmed(retry_client, params, failure):
    """GIVEN usage on a failed attempt. WHEN retry succeeds. THEN both charges are confirmed."""
    client, _ = retry_client
    await client.async_chat_completions(params, input_tokens=10)
    expected = 0.0001624
    assert client.budget.confirmed_usd == pytest.approx(expected)


@pytest.fixture
def zero_retries(retry_client):
    client, calls = retry_client
    client.retry_policy = make_retry_policy(retries=0)
    return client, calls


@pytest.mark.parametrize("failure", ["server"])
@pytest.mark.asyncio
async def test_zero_retries_makes_one_attempt(zero_retries, params, failure):
    """GIVEN retries disabled. WHEN a transient failure occurs. THEN only one attempt runs."""
    client, calls = zero_retries
    await client.async_chat_completions(params, input_tokens=10)
    expected = 1
    assert len(calls) == expected


@pytest.fixture
def cancelled_transport(client):
    client.budget = make_budget()
    client.async_client.chat.completions.create.side_effect = asyncio.CancelledError()
    return client


@pytest.mark.asyncio
async def test_task_cancellation_retains_uncertain_charge(cancelled_transport, params):
    """GIVEN cancellation after dispatch. WHEN a task stops. THEN its charge is unresolved."""
    with pytest.raises(asyncio.CancelledError):
        await cancelled_transport.async_chat_completions(params, input_tokens=10)
    expected = 0.0000175
    assert cancelled_transport.budget.unresolved_usd == pytest.approx(expected)


@pytest.fixture
def immediate_retry(retry_client, monkeypatch):
    from aiutils.control import wait, wait_sync
    client, calls = retry_client
    client.retry_policy = make_retry_policy(retries=1, min_wait=0, max_wait=0)
    monkeypatch.setattr("aiutils.retries.wait", wait)
    monkeypatch.setattr("aiutils.retries.wait_sync", wait_sync)
    return client, calls


@pytest.mark.parametrize("failure", ["invalid-delay"])
@pytest.mark.asyncio
async def test_zero_delay_allows_async_retry(immediate_retry, params, failure):
    """GIVEN zero retry delay. WHEN a transient request fails. THEN the retry succeeds."""
    reply = await immediate_retry[0].async_chat_completions(params, input_tokens=10)
    expected = "hello"
    assert reply.content == expected


@pytest.mark.parametrize("failure", ["invalid-delay"])
def test_zero_delay_allows_sync_retry(immediate_retry, params, failure):
    """GIVEN zero retry delay. WHEN a sync request fails. THEN the retry succeeds."""
    reply = immediate_retry[0].chat_completions(params, input_tokens=10)
    expected = "hello"
    assert reply.content == expected


class StopPolicy:
    def delay(self, error, attempt, **kwargs):
        return None


@pytest.fixture
def structural_policy(client):
    policy = StopPolicy()
    client.retry_policy = policy
    return client, policy


def test_generation_accepts_structural_retry_policy(structural_policy):
    """GIVEN a policy protocol implementation. WHEN generation is created. THEN it is retained."""
    client, expected = structural_policy
    result = make_generate(client=client)
    assert result.client.retry_policy is expected


@pytest.fixture
def dictionary_policy(mock_sync_client, mock_async_client):
    mock_sync_client.chat.completions.create.side_effect = TimeoutError("timeout")
    client = make_client(client=mock_sync_client, async_client=mock_async_client, retry_policy={"retries": 0})
    return client, mock_sync_client.chat.completions.create


def test_dictionary_policy_disables_retries(dictionary_policy, params):
    """GIVEN dictionary settings with zero retries. WHEN a request fails. THEN one attempt runs."""
    client, operation = dictionary_policy
    client.chat_completions(params)
    expected = 1
    assert operation.call_count == expected
