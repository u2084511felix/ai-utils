from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aiutils import clients
from aiutils.clients import ClientReply, InputType
from aiutils.exceptions import InitializationError
from aiutils.models import Vendor


@pytest.mark.parametrize("vendor,key,url", [
    (Vendor.OPENAI, "OPENAI_API_KEY", None),
    (Vendor.GOOGLE, "GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    (Vendor.BRAVE, "BRAVE_API_KEY", "https://api.search.brave.com/res/v1"),
])
def test_vendor_configuration(monkeypatch, vendor, key, url):
    monkeypatch.setenv(key, "fake-key")
    assert clients._client_type_kwargs(vendor) == {"api_key": "fake-key", "base_url": url}
    monkeypatch.delenv(key)
    with pytest.raises(EnvironmentError, match="not found"):
        clients._client_type_kwargs(vendor)


def test_unsupported_vendor():
    with pytest.raises(ValueError, match="Unsupported vendor"):
        clients._client_type_kwargs("unknown")


def test_client_constructor_and_errors():
    constructor = Mock()
    assert clients._make_client_type(constructor, api_key="fake", base_url="https://example.invalid") is constructor.return_value
    constructor.assert_called_once_with(api_key="fake", base_url="https://example.invalid")
    with pytest.raises(InitializationError, match="api_key"):
        clients._make_client_type(constructor, api_key="")
    error = ValueError("bad client")
    constructor.side_effect = error
    with pytest.raises(InitializationError) as exc:
        clients._make_client_type(constructor, api_key="fake")
    assert exc.value.__cause__ is error


@pytest.mark.parametrize("factory,sdk_name", [
    ("make_openai_client", "OpenAI"), ("make_async_openai_client", "AsyncOpenAI"),
    ("make_google_client", "OpenAI"), ("make_brave_client", "OpenAI"),
])
def test_client_factories_use_sdk_constructor(monkeypatch, factory, sdk_name):
    constructor = Mock()
    monkeypatch.setattr(clients, sdk_name, constructor)
    assert getattr(clients, factory)() is constructor.return_value
    assert constructor.call_args.kwargs["api_key"] == "pytest-not-a-real-key"


def test_reply_defaults_are_independent():
    first, second = ClientReply(), ClientReply()
    first.usage.input_tokens = 7
    assert second.usage.input_tokens == 0


def test_parse_text_and_usage(response_factory):
    reply = clients._parse_response(response_factory())
    assert reply.content == "hello"
    assert reply.error is None
    assert reply.usage.input_tokens == 10
    assert reply.usage.output_tokens == 5
    assert reply.usage.reasoning_tokens == 2


def test_parse_tool_calls(response_factory):
    calls = [SimpleNamespace(id="call-1")]
    assert clients._parse_response(response_factory(tool_calls=calls)).content is calls


def test_missing_content_is_error(response_factory):
    reply = clients._parse_response(response_factory(content=None))
    assert isinstance(reply.error, RuntimeError)
    assert "no content" in str(reply.error)


def test_missing_response_and_provider_exception():
    assert "No response" in str(clients._parse_response(None).error)
    error = RuntimeError("provider unavailable")
    assert clients._parse_response(error).error is error


def test_usage_requires_provider_counts():
    with pytest.raises(RuntimeError, match="did not report token usage"):
        clients._get_usage(SimpleNamespace())


def test_usage_without_reasoning_details(response_factory):
    response = response_factory()
    del response.usage.completion_tokens_details
    assert clients._get_usage(response).reasoning_tokens == 0


def test_empty_choices_returns_error(response_factory):
    response = response_factory()
    response.choices = []
    assert isinstance(clients._parse_response(response).error, RuntimeError)


@pytest.mark.parametrize("method,operation", [("chat_completions", "create"), ("chat_completions_body", "create")])
def test_sync_chat_forwards_parameters(client, method, operation):
    params = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    reply = getattr(client, method)(params)
    getattr(client.client.chat.completions, operation).assert_called_once_with(**params)
    assert reply.content == "hello"


@pytest.mark.parametrize("input_type,operation", [(InputType.JSON, "create"), (InputType.PYDANTIC, "parse")])
def test_sync_body_dispatch(client, input_type, operation):
    assert client.chat_completions_body({"model": "gpt-4o"}, input_type=input_type).error is None
    getattr(client.client.chat.completions, operation).assert_called_once_with(model="gpt-4o")


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type,operation", [("json", "create"), ("pydantic", "parse")])
async def test_async_body_dispatch(client, input_type, operation):
    reply = await client.async_chat_completions_body({"model": "gpt-4o"}, input_type=input_type)
    getattr(client.async_client.chat.completions, operation).assert_awaited_once_with(model="gpt-4o")
    assert reply.content == "hello"


@pytest.mark.asyncio
async def test_async_chat(client):
    reply = await client.async_chat_completions({"model": "gpt-4o"})
    client.async_client.chat.completions.create.assert_awaited_once_with(model="gpt-4o")
    assert reply.content == "hello"


@pytest.mark.asyncio
async def test_provider_exceptions_are_returned_in_replies(client):
    error = RuntimeError("provider failed")
    client.client.chat.completions.create.side_effect = error
    client.async_client.chat.completions.create.side_effect = error
    assert client.chat_completions({}).error is error
    assert client.chat_completions_body({}).error is error
    assert (await client.async_chat_completions({})).error is error
    assert (await client.async_chat_completions_body({})).error is error


@pytest.mark.asyncio
async def test_invalid_input_type_does_not_call_provider(client):
    assert isinstance(client.chat_completions_body({}, input_type="bad").error, ValueError)
    assert isinstance((await client.async_chat_completions_body({}, input_type="bad")).error, ValueError)
    client.client.chat.completions.create.assert_not_called()
    client.async_client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_embedding_response_and_error(client):
    params = dict(input="text", model="test", encoding_format="float", dimensions=2)
    assert await client.make_embeddings(**params) == [0.1, 0.2]
    client.async_client.embeddings.create.assert_awaited_once_with(**params)
    error = RuntimeError("embeddings failed")
    client.async_client.embeddings.create.side_effect = error
    assert await client.make_embeddings(**params) is error


@pytest.mark.parametrize("vendor", ["OPENAI", "google", "brave"])
def test_make_client_accepts_explicit_clients(client, vendor):
    result = clients.make_client(vendor, client=client.client, async_client=client.async_client)
    assert result.vendor == vendor.lower()
    assert result.client is client.client
    assert result.async_client is client.async_client


@pytest.mark.parametrize("vendor,factory", [("openai", "make_openai_client"), ("google", "make_google_client"), ("brave", "make_brave_client")])
def test_set_vendor_switches_clients(monkeypatch, client, vendor, factory):
    sync, asynchronous = Mock(), Mock()
    monkeypatch.setattr(clients, factory, lambda: sync)
    monkeypatch.setattr(clients, "make_async_openai_client", lambda: asynchronous)
    client.set_vendor(vendor.upper())
    assert client.vendor == vendor
    assert client.client is sync
    assert client.async_client is asynchronous


@pytest.mark.asyncio
async def test_responses_errors(client):
    error = RuntimeError("responses unavailable")
    client.client.responses.create.side_effect = error
    client.async_client.responses.create.side_effect = error
    assert client.responses_create("hi", model="test").error is error
    assert (await client.async_responses_create("hi", model="test")).error is error


def test_responses_api_text(client):
    client.client.responses.create.return_value = SimpleNamespace(
        output_text="hello", output=[],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    assert client.responses_create("hi", model="gpt-4o").content == "hello"


@pytest.mark.parametrize("method", ["chat_completions", "chat_completions_body"])
def test_sync_chat_verbose_logs_provider_response(client, method):
    client.logger = Mock()
    response = client.client.chat.completions.create.return_value
    assert getattr(client, method)({}, verbose=True).error is None
    client.logger.debug.assert_called_once_with(response)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["async_chat_completions", "async_chat_completions_body"])
async def test_async_chat_verbose_logs_provider_response(client, method):
    client.logger = Mock()
    response = client.async_client.chat.completions.create.return_value
    assert (await getattr(client, method)({}, verbose=True)).error is None
    client.logger.debug.assert_called_once_with(response)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_forwards_options_and_logs_dump(client, response_factory, asynchronous):
    # Isolate request forwarding; the real Responses parsing defect has its own
    # regression test above.
    response = response_factory()
    response.model_dump = Mock(return_value={"id": "response-id"})
    client.logger = Mock()
    if asynchronous:
        create = client.async_client.responses.create
        create.return_value = response
        reply = await client.async_responses_create("question", model="test", temperature=0.2, verbose=True)
        create.assert_awaited_once_with(input="question", model="test", temperature=0.2)
    else:
        create = client.client.responses.create
        create.return_value = response
        reply = client.responses_create("question", model="test", temperature=0.2, verbose=True)
        create.assert_called_once_with(input="question", model="test", temperature=0.2)
    assert reply.content == "hello"
    client.logger.debug.assert_called_once_with({"id": "response-id"})


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_forwards_tools(client, response_factory, asynchronous):
    tools = [{"type": "web_search_preview"}]
    client.client.responses.create.return_value = response_factory()
    if asynchronous:
        await client.async_responses_create("q", model="test", tools=tools)
        create = client.async_client.responses.create
    else:
        client.responses_create("q", model="test", tools=tools)
        create = client.client.responses.create
    assert create.call_args.kwargs.get("tools") == tools


@pytest.mark.parametrize("sync_missing", [True, False])
def test_set_vendor_rejects_unavailable_clients(monkeypatch, client, sync_missing):
    monkeypatch.setattr(clients, "make_openai_client", lambda: None if sync_missing else client.client)
    monkeypatch.setattr(clients, "make_async_openai_client", lambda: client.async_client if sync_missing else None)
    with pytest.raises(RuntimeError, match="Sync client" if sync_missing else "Async client"):
        client.set_vendor("openai")


def test_invalid_vendor_does_not_replace_clients(client):
    original_sync, original_async = client.client, client.async_client
    with pytest.raises(ValueError):
        client.set_vendor("unknown")
    assert client.client is original_sync
    assert client.async_client is original_async


def test_make_client_wraps_constructor_error(monkeypatch, client):
    error = RuntimeError("setup failed")
    monkeypatch.setattr(clients, "Client", Mock(side_effect=error))
    with pytest.raises(InitializationError, match="setup failed") as exc:
        clients.make_client(client=client.client, async_client=client.async_client)
    assert exc.value.__cause__ is error


def test_usage_handles_explicitly_absent_completion_details(response_factory):
    response = response_factory()
    response.usage.completion_tokens_details = None
    assert clients._get_usage(response).reasoning_tokens == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_preserves_output_usage_and_accepts_input(client, asynchronous):
    operation = SimpleNamespace(type="apply_patch_call", id="patch-1")
    response = SimpleNamespace(
        output_text="", output=[operation],
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=5,
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        ),
    )
    sdk = client.async_client if asynchronous else client.client
    sdk.responses.create.return_value = response
    tools = [{"type": "apply_patch"}]
    if asynchronous:
        reply = await client.async_responses_create(input="edit", model="test", tools=tools)
    else:
        reply = client.responses_create(input="edit", model="test", tools=tools)
    sdk.responses.create.assert_called_once_with(input="edit", model="test", tools=tools)
    assert reply.error is None
    assert reply.response is response
    assert reply.response.output == [operation]
    assert reply.usage.input_tokens == 10
    assert reply.usage.output_tokens == 5
    assert reply.usage.reasoning_tokens == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_rejects_ambiguous_input(client, asynchronous):
    with pytest.raises(TypeError, match="either prompt or input"):
        if asynchronous:
            await client.async_responses_create("first", input="second", model="test")
        else:
            client.responses_create("first", input="second", model="test")
    client.client.responses.create.assert_not_called()
    client.async_client.responses.create.assert_not_called()
