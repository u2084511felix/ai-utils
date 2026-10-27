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


@pytest.mark.xfail(strict=True, raises=IndexError, reason="Empty choices are indexed before checking for an empty list")
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


@pytest.mark.xfail(strict=True, raises=AttributeError, reason="Responses API output is parsed as a Chat Completions response")
def test_responses_api_text(client):
    client.client.responses.create.return_value = SimpleNamespace(
        output_text="hello", output=[],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    assert client.responses_create("hi", model="gpt-4o").content == "hello"
