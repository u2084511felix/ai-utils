from aiutils.budget import TokenUsage
from pydantic import BaseModel
from types import SimpleNamespace
from unittest.mock import Mock, call as mock_call

import pytest

from aiutils import clients
from aiutils.clients import (
    ClientReply, 
    InputType,
    make_openai_client,
    make_google_client,
    make_brave_client,
    make_async_openai_client,
)
from aiutils.exceptions import InitializationError


@pytest.fixture
def vendor_sdk(monkeypatch, key):
    constructor = Mock()
    monkeypatch.setenv(key, "fake-key")
    monkeypatch.setattr(clients, "OpenAI", constructor)
    return constructor


@pytest.mark.parametrize(
    "factory,key,url", [
        (make_openai_client, "OPENAI_API_KEY", None), 
        (make_google_client, "GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/"), 
        (make_brave_client, "BRAVE_API_KEY", "https://api.search.brave.com/res/v1")
    ]
)
class TestVendorFactory:
    def test_vendor_factory_returns_client(self, vendor_sdk, factory, key, url):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN vendor factory returns client."""
        result = factory()
        assert result is vendor_sdk.return_value

    def test_vendor_factory_passes_configuration(self, vendor_sdk, factory, key, url):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN vendor factory passes configuration."""
        constructor = vendor_sdk
        factory()
        expected = [mock_call(api_key="fake-key", base_url=url, max_retries=0)]
        assert constructor.call_args_list == expected


@pytest.fixture
def missing_credentials(monkeypatch, factory, key, missing):
    if missing:
        monkeypatch.delenv(key)
    else:
        monkeypatch.setenv(key, "")
    return factory


@pytest.mark.parametrize(
    "factory,key", [
        (make_openai_client, "OPENAI_API_KEY"), 
        (make_google_client, "GEMINI_API_KEY"), 
        (make_brave_client, "BRAVE_API_KEY")
    ])
@pytest.mark.parametrize("missing", [False, True])
def test_vendor_requires_credentials(missing_credentials, factory, key, missing):
    """GIVEN a missing or empty API key.
    WHEN the client operation is called.
    THEN the missing or empty key is rejected."""
    factory = missing_credentials
    with pytest.raises(EnvironmentError, match="not found") as exc:
        factory()
    expected = EnvironmentError
    assert exc.type == expected


def test_unsupported_vendor(client):
    """GIVEN an unknown vendor.
    WHEN the client operation is called.
    THEN ValueError is raised."""
    with pytest.raises(ValueError) as exc:
        clients.make_client("unknown", client=client.client, async_client=client.async_client)
    expected = ValueError
    assert exc.type is expected



@pytest.fixture
def failed_sdk(monkeypatch):
    error = ValueError("bad client")
    monkeypatch.setattr(clients, "OpenAI", Mock(side_effect=error))
    return error


def test_sdk_constructor_error_is_chained(failed_sdk):
    """GIVEN an operation that reports an error.
    WHEN the client operation is called.
    THEN the error retains the SDK error as its cause."""
    error = failed_sdk
    with pytest.raises(InitializationError) as exc:
        clients.make_openai_client()
    assert exc.value.__cause__ is error


@pytest.fixture
def sdk_factory(monkeypatch, sdk_name):
    constructor = Mock()
    monkeypatch.setattr(clients, sdk_name, constructor)
    return constructor


@pytest.mark.parametrize(
    "factory,sdk_name", [
        (make_openai_client, "OpenAI"), 
        (make_async_openai_client, "AsyncOpenAI"), 
        (make_google_client, "OpenAI"), 
        (make_brave_client, "OpenAI")
    ])
def test_sdk_factory_returns_client(sdk_factory, factory, sdk_name):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN sdk factory returns client."""
    actual = factory()
    assert actual is sdk_factory.return_value


@pytest.mark.parametrize(
    "factory,sdk_name", [
        (make_openai_client, "OpenAI"), 
        (make_async_openai_client, "AsyncOpenAI"), 
        (make_google_client, "OpenAI"), 
        (make_brave_client, "OpenAI")
    ])
def test_sdk_factory_passes_key(sdk_factory, factory, sdk_name):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN sdk factory passes key."""
    factory()
    actual = sdk_factory.call_args.kwargs["api_key"]
    expected = "pytest-not-a-real-key"
    assert actual == expected


@pytest.fixture
def changed_reply():
    first = ClientReply(usage=TokenUsage(input_tokens=7))
    return first


def test_reply_defaults_are_independent(changed_reply):
    """GIVEN a reply with changed token usage.
    WHEN another reply is created.
    THEN the new reply has no token charges."""
    reply = ClientReply(usage=TokenUsage())
    expected = 0
    assert reply.usage.input_tokens == expected


def test_chat_reply_retains_text(client):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN chat reply retains text."""
    reply = client.chat_completions({})
    expected = "hello"
    assert reply.content == expected


def test_chat_reply_has_no_error(client):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN reply.error is None"""
    reply = client.chat_completions({})
    expected = None
    assert reply.error == expected


def test_chat_reply_counts_input(client):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN chat reply counts input."""
    reply = client.chat_completions({})
    expected = 10
    assert reply.usage.input_tokens == expected


def test_chat_reply_counts_output(client):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN chat reply counts output."""
    reply = client.chat_completions({})
    expected = 5
    assert reply.usage.output_tokens == expected


def test_chat_reply_counts_reasoning(client):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN chat reply counts reasoning."""
    reply = client.chat_completions({})
    expected = 2
    assert reply.usage.reasoning_tokens == expected


@pytest.fixture
def tool_response(client, response_factory):
    calls = [SimpleNamespace(id="call-1")]
    client.client.chat.completions.create.return_value = response_factory(tool_calls=calls)
    return (calls, client)


def test_chat_tool_calls(tool_response):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN the reply contains the provider tool calls."""
    calls, client = tool_response
    reply = client.chat_completions({})
    assert reply.content  is calls


@pytest.fixture
def missing_content_client(client, response_factory):
    client.client.chat.completions.create.return_value = response_factory(content=None)
    return client


def test_missing_content_returns_error(missing_content_client):
    """GIVEN a provider response with no content.
    WHEN the client operation is called.
    THEN missing content returns error."""
    reply = missing_content_client.chat_completions({})
    assert isinstance(reply.error, RuntimeError)


def test_missing_content_explains_error(missing_content_client):
    """GIVEN a provider response with no content.
    WHEN the client operation is called.
    THEN missing content explains error."""
    reply = missing_content_client.chat_completions({})
    error_string = str(reply.error)
    expected = "no content"
    assert expected in error_string


@pytest.fixture
def missing_response_client(client):
    client.client.chat.completions.create.return_value = None
    return client


def test_missing_response_is_error(missing_response_client):
    """GIVEN no provider response.
    WHEN the client operation is called.
    THEN the reply reports that no response was received."""
    reply = missing_response_client.chat_completions({})
    error_str = str(reply.error)
    expected_str = "No response"
    assert expected_str in error_str


@pytest.fixture
def missing_usage_client(client, response_factory):
    response = response_factory()
    del response.usage
    client.client.chat.completions.create.return_value = response
    return client


def test_usage_requires_provider_counts(missing_usage_client):
    """GIVEN a provider response with absent usage fields.
    WHEN the client operation is called.
    THEN the absent token counts are reported."""
    with pytest.raises(RuntimeError, match="did not report token usage") as exc:
        missing_usage_client.chat_completions({})
    expected = RuntimeError
    assert exc.type is expected


@pytest.fixture
def missing_reasoning_client(client, response_factory, absent):
    response = response_factory()
    if absent:
        del response.usage.completion_tokens_details
    else:
        response.usage.completion_tokens_details = None
    client.client.chat.completions.create.return_value = response
    return client


@pytest.mark.parametrize("absent", [True, False])
def test_usage_without_reasoning_details(missing_reasoning_client, absent):
    """GIVEN a provider response with absent usage fields.
    WHEN the client operation is called.
    THEN the reasoning count is zero."""
    reply = missing_reasoning_client.chat_completions({})
    reasoning_tokens = reply.usage.reasoning_tokens
    expected = 0
    assert reasoning_tokens == expected


@pytest.fixture
def empty_choices_client(client, response_factory):
    response = response_factory()
    response.choices = []
    client.client.chat.completions.create.return_value = response
    return client


def test_empty_choices_returns_error(empty_choices_client):
    """GIVEN a response with no choices.
    WHEN the client operation is called.
    THEN the reply reports an error."""
    reply = empty_choices_client.chat_completions({})
    assert isinstance(reply.error, RuntimeError)


@pytest.fixture
def chat_params(client):
    params = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    return (params, client)


@pytest.mark.parametrize(
    "method,op", [
        ("chat_completions", "create"), 
        ("chat_completions_body", "create")
    ]
)
class TestSyncChat:
    def test_sync_chat_passes_parameters(self, chat_params, method, op):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN sync chat passes parameters."""
        params, client = chat_params
        getattr(client, method)(params)
        result = getattr(client.client.chat.completions, op).call_args_list
        expected = [mock_call(**params)]
        assert result == expected

    def test_sync_chat_returns_text(self, chat_params, method, op):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN sync chat returns text."""
        params, client = chat_params
        reply = getattr(client, method)(params)
        expected = "hello"
        assert reply.content == expected

@pytest.fixture
def model():
    return "gpt-4o"

class ParsedResult(BaseModel):
    value: str


@pytest.fixture
def body_params(model, request):
    params = {"model": model}
    case = getattr(request.node, "callspec", None)
    if case is not None and case.params.get("input_type") == "pydantic":
        params["response_format"] = ParsedResult
    return params

@pytest.mark.parametrize(
    "input_type,op", [
        (InputType.JSON, "create"), 
        (InputType.PYDANTIC, "parse")
    ])
class TestSyncBody:
    def test_sync_body_has_no_error(self, client, input_type, body_params, op):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN sync body has no error."""
        reply = client.chat_completions_body(body_params, input_type=input_type)
        assert reply.error is None


    def test_sync_body_selects_endpoint(self, client, input_type, model, body_params, op):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN sync body selects endpoint."""
        client.chat_completions_body(body_params, input_type=input_type)
        expected_call_args = [mock_call(**body_params)]
        call_args_list = getattr(client.client.chat.completions, op).call_args_list
        assert call_args_list == expected_call_args


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_type,op", [
        ("json", "create"), 
        ("pydantic", "parse")
    ])
class TestAsyncBody:
    async def test_async_body_selects_endpoint(self, client, body_params, model, input_type, op):
        """GIVEN a client and a provider response.
        WHEN async_chat_completions is called.
        THEN async body selects endpoint."""
        await client.async_chat_completions_body(body_params, input_type=input_type)
        result = getattr(client.async_client.chat.completions, op)
        expected = [mock_call(**body_params)]
        assert result.await_args_list == expected

    async def test_async_body_returns_text(self, client, body_params, input_type, op):
        """GIVEN a client and a provider response.
        WHEN async_chat_completions_body is called.
        THEN async body returns text."""
        reply = await client.async_chat_completions_body(body_params, input_type=input_type)
        expected = "hello"
        assert reply.content == expected


@pytest.mark.asyncio
async def test_async_chat_passes_parameters(client, body_params, model):
    """GIVEN a client and a provider response.
    WHEN async_chat_completions is called.
    THEN async chat passes parameters."""
    await client.async_chat_completions(body_params)
    expected = [mock_call(model=model)]
    await_args_list = client.async_client.chat.completions.create.await_args_list
    assert await_args_list == expected


@pytest.mark.asyncio
async def test_async_chat_returns_text(client, body_params):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN async chat returns text."""
    reply = await client.async_chat_completions(body_params)
    expected = "hello"
    assert reply.content == expected


@pytest.fixture
def sync_failure(client):
    error = RuntimeError("provider failed")
    client.client.chat.completions.create.side_effect = error
    return (error, client)


@pytest.mark.parametrize("method", ["chat_completions", "chat_completions_body"])
def test_sync_provider_exceptions_are_returned_in_replies(sync_failure, method):
    """GIVEN an operation that reports an error.
    WHEN the client operation is called.
    THEN the reply retains the provider error."""
    error, client = sync_failure
    reply = getattr(client, method)({})
    assert reply.error is error


@pytest.fixture
def async_failure(client):
    error = RuntimeError("provider failed")
    client.async_client.chat.completions.create.side_effect = error
    return (error, client)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["async_chat_completions", "async_chat_completions_body"])
async def test_async_provider_exceptions_are_returned_in_replies(async_failure, method):
    """GIVEN an operation that reports an error.
    WHEN the client operation is called.
    THEN the reply retains the provider error."""
    error, client = async_failure
    reply = await getattr(client, method)({})
    assert reply.error is error


@pytest.fixture
def bad_input():
    return "bad"

def test_invalid_sync_body_returns_error(client, bad_input):
    """GIVEN an invalid input type.
    WHEN the client operation is called.
    THEN invalid sync body returns error."""
    with pytest.raises(ValueError) as error:
        client.chat_completions_body({}, input_type=bad_input)
    expected = ValueError
    assert error.type is expected


def test_invalid_sync_body_avoids_provider(client, bad_input):
    """GIVEN an invalid input type.
    WHEN the client operation is called.
    THEN invalid sync body avoids provider."""
    with pytest.raises(ValueError):
        client.chat_completions_body({}, input_type=bad_input)
    call_args_list = client.client.chat.completions.create.call_args_list
    expected = []
    assert call_args_list == expected


@pytest.mark.asyncio
async def test_invalid_async_body_returns_error(client, bad_input):
    """GIVEN an invalid input type.
    WHEN the client operation is called.
    THEN invalid async body returns ValueError."""
    with pytest.raises(ValueError) as error:
        await client.async_chat_completions_body({}, input_type=bad_input)
    expected = ValueError
    assert error.type is expected


@pytest.mark.asyncio
async def test_invalid_async_body_avoids_provider(client, bad_input):
    """GIVEN an invalid input type.
    WHEN the client operation is called.
    THEN invalid async body avoids provider."""
    with pytest.raises(ValueError):
        await client.async_chat_completions_body({}, input_type=bad_input)
    await_args_list = client.async_client.chat.completions.create.await_args_list
    expected = []
    assert await_args_list == expected

@pytest.fixture
def embedding_params():
    return {
        "input": "text",
        "model": "test",
        "encoding_format": "float",
        "dimensions": 2
    }

@pytest.mark.asyncio
async def test_client_embedding_returns_vector(client, embedding_params):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN client embedding returns vector."""
    result = await client.make_embeddings(**embedding_params)
    expected = [0.1, 0.2]
    assert result == expected


@pytest.mark.asyncio
async def test_client_embedding_passes_parameters(client, embedding_params):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN client embedding passes parameters."""
    await client.make_embeddings(**embedding_params)
    expected = [mock_call(**embedding_params)]
    await_args_list = client.async_client.embeddings.create.await_args_list
    assert await_args_list == expected

@pytest.fixture
def embedding_error():
    return RuntimeError("embeddings failed")

@pytest.fixture
def failed_embedding_client(client, embedding_error):
    client.async_client.embeddings.create.side_effect = embedding_error
    return client


@pytest.mark.asyncio
async def test_embedding_error(
    failed_embedding_client, 
    embedding_error,
    embedding_params
):
    """GIVEN an operation that reports an error.
    WHEN the client operation is called.
    THEN the result retains the provider error."""
    result = await failed_embedding_client.make_embeddings(**embedding_params) 
    assert result is embedding_error


@pytest.fixture
def make_client_kwargs(client):
    return {
        "client": client.client,
        "async_client": client.async_client
    }

@pytest.mark.parametrize("vendor", ["OPENAI", "google", "brave"])
class TestClientFactory:
    def test_client_factory_normalizes_vendor(self, vendor, make_client_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN client factory normalizes vendor."""
        result = clients.make_client(vendor, **make_client_kwargs)
        expected_vendor = vendor.lower()
        assert result.vendor == expected_vendor

    def test_client_factory_retains_sync_client(self, vendor, make_client_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN client factory retains sync client."""
        result = clients.make_client(vendor, **make_client_kwargs)
        expected_client = make_client_kwargs["client"]
        assert result.client is expected_client

    def test_client_factory_retains_async_client(self, vendor, make_client_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN client factory retains async client."""
        result = clients.make_client(vendor, **make_client_kwargs)
        expected_async_client = make_client_kwargs["async_client"]
        assert result.async_client is expected_async_client


@pytest.fixture
def vendor_clients(monkeypatch, client, factory):
    sync, asynchronous = (Mock(), Mock())
    monkeypatch.setattr(clients, factory, lambda: sync)
    monkeypatch.setattr(clients, "make_async_openai_client", lambda: asynchronous)
    return {
        "asynchronous": asynchronous,
        "sync": sync,
        "client": client
    }

@pytest.fixture
def vendor_sync_client(vendor_clients):
    return vendor_clients["sync"]

@pytest.fixture
def vendor_async_client(vendor_clients):
    return vendor_clients["asynchronous"]

@pytest.fixture
def vendor_client(vendor_clients):
    return vendor_clients["client"]


@pytest.mark.parametrize(
    "vendor,factory", [
        ("openai", "make_openai_client"),
        ("google", "make_google_client"),
        ("brave", "make_brave_client"),
    ])
class TestVendorChange:
    def test_vendor_change_sets_vendor(self, vendor_client, vendor, factory):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN vendor change sets vendor."""
        vendor_client.set_vendor(vendor.upper())
        assert vendor_client.vendor == vendor

    def test_vendor_change_replaces_sync_client(self, vendor_sync_client, vendor_client, vendor, factory):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN vendor change replaces sync client."""
        vendor_client.set_vendor(vendor.upper())
        assert vendor_client.client is vendor_sync_client

    def test_vendor_change_replaces_async_client(self, vendor_async_client, vendor_client, vendor, factory):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN vendor change replaces async client."""
        vendor_client.set_vendor(vendor.upper())
        assert vendor_client.async_client is vendor_async_client


@pytest.fixture
def sync_error(client):
    error = RuntimeError("responses unavailable")
    client.client.responses.create.side_effect = error
    return (error, client)


def test_sync_responses_errors(sync_error):
    """GIVEN an operation that reports an error.
    WHEN the client operation is called.
    THEN the reply retains the provider error."""
    error, client = sync_error
    reply = client.responses_create("hi", model="test")
    assert reply.error is error


@pytest.fixture
def async_error(client):
    error = RuntimeError("responses unavailable")
    client.async_client.responses.create.side_effect = error
    return (error, client)


@pytest.mark.asyncio
async def test_async_responses_errors(async_error):
    """GIVEN an operation that reports an error.
    WHEN the client operation is called.
    THEN the reply retains the provider error."""
    error, client = async_error
    reply = await client.async_responses_create("hi", model="test")
    assert reply.error is error


@pytest.fixture
def response_text(client):
    return_value = SimpleNamespace(
        output_text="hello", 
        output=[], 
        usage=SimpleNamespace(
            input_tokens=10, 
            output_tokens=5
        )
    )
    client.client.responses.create.return_value = return_value
    return client


def test_responses_api_text(response_text):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN the reply contains the response text."""
    client = response_text
    result = client.responses_create("hi", model="gpt-4o").content
    expected = "hello"
    assert result == expected


@pytest.fixture
def sync_logger(client):
    client.logger = Mock()
    response = client.client.chat.completions.create.return_value
    return (response, client)


@pytest.mark.parametrize("method", ["chat_completions", "chat_completions_body"])
def test_verbose_sync_chat_has_no_error(sync_logger, method):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN verbose sync chat has no error."""
    response, client = sync_logger
    reply = getattr(client, method)({}, verbose=True)
    assert reply.error is None


@pytest.mark.parametrize("method", ["chat_completions", "chat_completions_body"])
def test_verbose_sync_chat_logs_response(sync_logger, method):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN verbose sync chat logs response."""
    response, client = sync_logger
    getattr(client, method)({}, verbose=True)
    expected = [mock_call(response)]
    assert client.logger.debug.call_args_list == expected


@pytest.fixture
def async_logger(client):
    client.logger = Mock()
    response = client.async_client.chat.completions.create.return_value
    return (response, client)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["async_chat_completions", "async_chat_completions_body"])
async def test_verbose_async_chat_has_no_error(async_logger, method):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN verbose async chat has no error."""
    response, client = async_logger
    reply = await getattr(client, method)({}, verbose=True)
    assert reply.error is None


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["async_chat_completions", "async_chat_completions_body"])
async def test_verbose_async_chat_logs_response(async_logger, method):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN verbose async chat logs response."""
    response, client = async_logger
    await getattr(client, method)({}, verbose=True)
    expected = [mock_call(response)]
    assert client.logger.debug.call_args_list == expected

@pytest.fixture
def model_dump_return():
    return {"id": "response-id"}



@pytest.fixture
def sync_response_logger(client, response_factory, model_dump_return):
    client.logger = Mock()
    response = response_factory()
    response.model_dump = Mock(return_value=model_dump_return)
    create = client.client.responses.create
    create.return_value = response
    return (create, client)

@pytest.fixture
def async_response_logger(client, response_factory, model_dump_return):
    client.logger = Mock()
    response = response_factory()
    response.model_dump = Mock(return_value=model_dump_return)
    create = client.async_client.responses.create
    create.return_value = response
    return (create, client)


@pytest.fixture
def response_create_kwargs():
    return {
        "input": "question", 
        "model": "test", 
        "temperature": 0.2,
    }


class TestResponsesCreate:

    def test_responses_pass_options(self, sync_response_logger, response_create_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN responses pass options."""
        create, client = sync_response_logger
        reply = client.responses_create(**response_create_kwargs, verbose=True)
        expected = [mock_call(**response_create_kwargs)]
        assert create.call_args_list == expected

    def test_responses_return_text(self, sync_response_logger, response_create_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN responses return text."""
        create, client = sync_response_logger
        reply = client.responses_create(**response_create_kwargs, verbose=True)
        expected = "hello"
        assert reply.content == expected


    def test_responses_log_output(self, sync_response_logger, response_create_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN responses log output."""
        create, client = sync_response_logger
        reply = client.responses_create(**response_create_kwargs, verbose=True)
        expected = [mock_call({"id": "response-id"})]
        assert client.logger.debug.call_args_list == expected


class TestAsyncResponsesCreate:

    @pytest.mark.asyncio
    async def test_responses_pass_options(self, async_response_logger, response_create_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN responses pass options."""
        create, client = async_response_logger
        reply = await client.async_responses_create(**response_create_kwargs, verbose=True)
        expected = [mock_call(**response_create_kwargs)]
        assert create.call_args_list == expected

    @pytest.mark.asyncio
    async def test_responses_return_text(self, async_response_logger, response_create_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN responses return text."""
        create, client = async_response_logger
        reply = await client.async_responses_create(**response_create_kwargs, verbose=True)
        expected = "hello"
        assert reply.content == expected

    @pytest.mark.asyncio
    async def test_responses_log_output(self, async_response_logger, response_create_kwargs, model_dump_return):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN responses log output."""
        create, client = async_response_logger
        reply = await client.async_responses_create(**response_create_kwargs, verbose=True)
        expected = [mock_call(model_dump_return)]
        assert client.logger.debug.call_args_list == expected


@pytest.fixture
def search_tools():
    return [{"type": "web_search_preview"}]

@pytest.fixture
def sync_response_tools(client, search_tools, response_factory):
    sdk =  client.client
    sdk.responses.create.return_value = response_factory()
    return (sdk, search_tools, client)

@pytest.fixture
def async_response_tools(client, search_tools, response_factory):
    sdk =  client.async_client
    sdk.responses.create.return_value = response_factory()
    return (sdk, search_tools, client)



def test_sync_responses_forwards_tools(sync_response_tools):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN the request retains the specified tools."""
    sdk, search_tools, client = sync_response_tools
    reply = client.responses_create("q", model="test", tools=search_tools)
    actual = sdk.responses.create.call_args.kwargs.get("tools")
    assert actual == search_tools

@pytest.mark.asyncio
async def test_async_responses_forwards_tools(async_response_tools):
    """GIVEN an asynchronous client and a provider response.
    WHEN the client operation is called.
    THEN the request retains the specified tools."""
    sdk, search_tools, client = async_response_tools
    reply = await client.async_responses_create("q", model="test", tools=search_tools)
    actual = sdk.responses.create.call_args.kwargs.get("tools")
    assert actual == search_tools


@pytest.fixture
def missing_sync_client(monkeypatch, client):
    monkeypatch.setattr(clients, "make_openai_client", lambda: None)
    return client

@pytest.fixture
def missing_async_client(monkeypatch, client):
    monkeypatch.setattr(clients, "make_async_openai_client", lambda: None)
    return client


def test_set_vendor_rejects_unavailable_sync_clients(missing_sync_client):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN the unavailable client is reported."""
    with pytest.raises(RuntimeError, match="Sync client") as exc:
        missing_sync_client.set_vendor("openai")
    expected = RuntimeError
    assert exc.type is expected

def test_set_vendor_rejects_unavailable_async_client(missing_async_client):
    """GIVEN an asynchronous client and a provider response.
    WHEN the client operation is called.
    THEN the unavailable client is reported."""
    with pytest.raises(RuntimeError, match="Async client") as exc:
        missing_async_client.set_vendor("openai")
    expected = RuntimeError
    assert exc.type is expected


def test_invalid_vendor_retains_sync_client(client):
    """GIVEN a configured client and an unknown vendor.
    WHEN the client operation is called.
    THEN invalid vendor retains sync client."""
    original_sync = client.client
    with pytest.raises(ValueError):
        client.set_vendor("unknown")
    assert client.client is original_sync


def test_invalid_vendor_retains_async_client(client):
    """GIVEN a configured client and an unknown vendor.
    WHEN the client operation is called.
    THEN invalid vendor retains async client."""
    original_async = client.async_client
    with pytest.raises(ValueError):
        client.set_vendor("unknown")
    assert client.async_client is original_async


@pytest.fixture
def client_failure(monkeypatch, client):
    error = RuntimeError("setup failed")
    monkeypatch.setattr(clients, "Client", Mock(side_effect=error))
    return (error, client)


def test_make_client_wraps_constructor_error(client_failure):
    """GIVEN an operation that reports an error.
    WHEN the client operation is called.
    THEN the error retains the original client error."""
    error, client = client_failure
    with pytest.raises(InitializationError, match="setup failed") as exc:
        clients.make_client(client=client.client, async_client=client.async_client)
    assert exc.value.__cause__ is error


@pytest.fixture
def operation():
    return SimpleNamespace(type="apply_patch_call", id="patch-1")

@pytest.fixture
def tokens():
    return {
        "reasoning_tokens": 3,
        "input_tokens": 10,
        "output_tokens": 5
    }

@pytest.fixture
def response(operation, tokens):
    details = SimpleNamespace(reasoning_tokens=tokens["reasoning_tokens"])
    return SimpleNamespace(
        output_text="", 
        output=[operation], 
        usage=SimpleNamespace(
            input_tokens=tokens["input_tokens"],
            output_tokens=tokens["output_tokens"], 
            output_tokens_details=details
        )
    )

@pytest.fixture
def apply_patch_tools():
    return [{"type": "apply_patch"}]

@pytest.fixture
def sync_patch_response(
    client, 
    operation,
    response,
    apply_patch_tools,
):
    sdk = client.client
    sdk.responses.create.return_value = response
    return (operation, response, sdk, apply_patch_tools, client)

@pytest.fixture
def async_sdk(client, response):
    sdk = client.async_client
    sdk.responses.create.return_value = response
    return sdk, client

@pytest.fixture
def async_sdk_client(async_sdk):
    sdk, client = async_sdk
    return client

@pytest.fixture
def sync_sdk(client, response):
    sdk = client.client
    sdk.responses.create.return_value = response
    return sdk, client

@pytest.fixture
def sync_sdk_client(sync_sdk):
    sdk, client = sync_sdk
    return client

@pytest.fixture
def patch_kwargs(apply_patch_tools):
    return {
        "input": "edit", 
        "model": "test", 
        "tools": apply_patch_tools
    }

@pytest.fixture
def ambiguous_kwargs():
    return {
        "prompt": "first", 
        "input": "second", 
        "model": "test"
    }

@pytest.mark.asyncio
class TestAsyncResponses:

    async def test_responses_pass_input(self, async_sdk, patch_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN responses pass input."""
        sdk, async_client = async_sdk
        reply = await async_client.async_responses_create(**patch_kwargs)
        expected = [mock_call(**patch_kwargs)]
        assert sdk.responses.create.call_args_list == expected

    async def test_responses_have_no_error(self, async_sdk_client, patch_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN responses have no error."""
        reply = await async_sdk_client.async_responses_create(**patch_kwargs)
        assert reply.error is None

    async def test_responses_retain_provider_reply(self, async_sdk_client, response, patch_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN responses retain provider reply."""
        reply = await async_sdk_client.async_responses_create(**patch_kwargs)
        assert reply.response is response

    async def test_responses_retain_operations(self, async_sdk_client, operation, patch_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN responses retain operations."""
        reply = await async_sdk_client.async_responses_create(**patch_kwargs)
        expected = [operation]
        assert reply.response.output == expected

    async def test_responses_count_input(self, async_sdk_client, tokens, patch_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN responses count input."""
        reply = await async_sdk_client.async_responses_create(**patch_kwargs)
        expected_num_tokens = tokens["input_tokens"]
        assert reply.usage.input_tokens == expected_num_tokens

    async def test_responses_count_output(self, async_sdk_client, tokens, patch_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN responses count output."""
        reply = await async_sdk_client.async_responses_create(**patch_kwargs)
        expected_num_tokens = tokens["output_tokens"]
        assert reply.usage.output_tokens == expected_num_tokens

    async def test_responses_count_reasoning(self, async_sdk_client, tokens, patch_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN responses count reasoning."""
        reply = await async_sdk_client.async_responses_create(**patch_kwargs)
        expected_num_tokens = tokens["reasoning_tokens"]
        assert reply.usage.reasoning_tokens == expected_num_tokens

    async def test_ambiguous_input_avoids_async_provider(self, client, ambiguous_kwargs):
        """GIVEN both prompt and input values.
        WHEN the client operation is called.
        THEN ambiguous input avoids async provider."""
        with pytest.raises(TypeError, match="either prompt or input"):
            reply = await client.async_responses_create(**ambiguous_kwargs)
        expected = []
        assert client.async_client.responses.create.call_args_list == expected


class TestSyncResponses:

    def test_responses_pass_input(self, sync_sdk, patch_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN responses pass input."""
        sdk, client = sync_sdk
        reply = client.responses_create(**patch_kwargs)
        expected = [mock_call(**patch_kwargs)]
        assert sdk.responses.create.call_args_list == expected

    def test_responses_have_no_error(self, sync_sdk_client, patch_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN responses have no error."""
        reply = sync_sdk_client.responses_create(**patch_kwargs)
        assert reply.error is None

    def test_responses_retain_provider_reply(self, sync_sdk_client, response, patch_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN responses retain provider reply."""
        reply = sync_sdk_client.responses_create(**patch_kwargs)
        assert reply.response is response

    def test_responses_retain_operations(self, sync_sdk_client, operation, patch_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN responses retain operations."""
        reply = sync_sdk_client.responses_create(**patch_kwargs)
        expected = [operation]
        assert reply.response.output == expected

    def test_responses_count_input(self, sync_sdk_client, tokens, patch_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN responses count input."""
        reply = sync_sdk_client.responses_create(**patch_kwargs)
        expected_num_tokens = tokens["input_tokens"]
        assert reply.usage.input_tokens == expected_num_tokens

    def test_responses_count_output(self, sync_sdk_client, tokens, patch_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN responses count output."""
        reply = sync_sdk_client.responses_create(**patch_kwargs)
        expected_num_tokens = tokens["output_tokens"]
        assert reply.usage.output_tokens == expected_num_tokens

    def test_responses_count_reasoning(self, sync_sdk_client, tokens, patch_kwargs):
        """GIVEN a client and a provider response.
        WHEN the client operation is called.
        THEN responses count reasoning."""
        reply = sync_sdk_client.responses_create(**patch_kwargs)
        expected_num_tokens = tokens["reasoning_tokens"]
        assert reply.usage.reasoning_tokens == expected_num_tokens

    def test_ambiguous_input_avoids_sync_provider(self, client, ambiguous_kwargs):
        """GIVEN both prompt and input values.
        WHEN the client operation is called.
        THEN ambiguous input avoids sync provider."""
        with pytest.raises(TypeError, match="either prompt or input"):
            reply = client.responses_create(**ambiguous_kwargs)
        expected = []
        assert client.client.responses.create.call_args_list == expected

