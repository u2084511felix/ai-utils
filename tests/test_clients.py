from types import SimpleNamespace
from unittest.mock import Mock, call as mock_call

import pytest

from aiutils import clients
from aiutils.clients import ClientReply, InputType
from aiutils.exceptions import InitializationError


@pytest.fixture
def vendor_sdk(monkeypatch, key):
    constructor = Mock()
    monkeypatch.setenv(key, "fake-key")
    monkeypatch.setattr(clients, "OpenAI", constructor)
    return constructor


@pytest.mark.parametrize("factory,key,url", [("make_openai_client", "OPENAI_API_KEY", None), ("make_google_client", "GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/"), ("make_brave_client", "BRAVE_API_KEY", "https://api.search.brave.com/res/v1")])
def test_vendor_factory_returns_client(vendor_sdk, factory, key, url):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN vendor factory returns client."""
    constructor = vendor_sdk
    result = getattr(clients, factory)()
    expected = constructor.return_value
    assert result is expected


@pytest.mark.parametrize("factory,key,url", [("make_openai_client", "OPENAI_API_KEY", None), ("make_google_client", "GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/"), ("make_brave_client", "BRAVE_API_KEY", "https://api.search.brave.com/res/v1")])
def test_vendor_factory_passes_configuration(vendor_sdk, factory, key, url):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN vendor factory passes configuration."""
    constructor = vendor_sdk
    getattr(clients, factory)()
    expected = [mock_call(api_key="fake-key", base_url=url)]
    assert constructor.call_args_list == expected


@pytest.fixture
def missing_credentials(monkeypatch, factory, key, missing):
    if missing:
        monkeypatch.delenv(key)
    else:
        monkeypatch.setenv(key, "")
    return factory


@pytest.mark.parametrize("factory,key", [("make_openai_client", "OPENAI_API_KEY"), ("make_google_client", "GEMINI_API_KEY"), ("make_brave_client", "BRAVE_API_KEY")])
@pytest.mark.parametrize("missing", [False, True])
def test_vendor_requires_credentials(missing_credentials, factory, key, missing):
    """GIVEN a missing or empty API key.
    WHEN the client operation is called.
    THEN the missing or empty key is rejected."""
    factory = missing_credentials
    with pytest.raises(EnvironmentError, match="not found") as exc:
        getattr(clients, factory)()
    expected = EnvironmentError
    assert exc.type == expected


def test_unsupported_vendor(client):
    """GIVEN an unknown vendor.
    WHEN the client operation is called.
    THEN the unknown vendor is rejected."""
    with pytest.raises(ValueError) as exc:
        clients.make_client("unknown", client=client.client, async_client=client.async_client)
    expected = ValueError
    assert exc.type == expected


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
    expected = True
    assert (exc.value.__cause__ is error) == expected


@pytest.fixture
def sdk_factory(monkeypatch, sdk_name):
    constructor = Mock()
    monkeypatch.setattr(clients, sdk_name, constructor)
    return constructor


@pytest.mark.parametrize("factory,sdk_name", [("make_openai_client", "OpenAI"), ("make_async_openai_client", "AsyncOpenAI"), ("make_google_client", "OpenAI"), ("make_brave_client", "OpenAI")])
def test_sdk_factory_returns_client(sdk_factory, factory, sdk_name):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN sdk factory returns client."""
    constructor = sdk_factory
    expected = constructor.return_value
    assert getattr(clients, factory)() is expected


@pytest.mark.parametrize("factory,sdk_name", [("make_openai_client", "OpenAI"), ("make_async_openai_client", "AsyncOpenAI"), ("make_google_client", "OpenAI"), ("make_brave_client", "OpenAI")])
def test_sdk_factory_passes_key(sdk_factory, factory, sdk_name):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN sdk factory passes key."""
    constructor = sdk_factory
    getattr(clients, factory)()
    expected = "pytest-not-a-real-key"
    assert constructor.call_args.kwargs["api_key"] == expected


@pytest.fixture
def changed_reply():
    first = ClientReply()
    first.usage.input_tokens = 7
    return first


def test_reply_defaults_are_independent(changed_reply):
    """GIVEN a reply with changed token usage.
    WHEN another reply is created.
    THEN the new reply has no token charges."""
    reply = ClientReply()
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
    THEN chat reply has no error."""
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
    result = client.chat_completions({}).content is calls
    expected = True
    assert result == expected


@pytest.fixture
def missing_content(client, response_factory):
    client.client.chat.completions.create.return_value = response_factory(content=None)
    return client


def test_missing_content_returns_error(missing_content):
    """GIVEN a provider response with no content.
    WHEN the client operation is called.
    THEN missing content returns error."""
    client = missing_content
    reply = client.chat_completions({})
    result = isinstance(reply.error, RuntimeError)
    expected = True
    assert result == expected


def test_missing_content_explains_error(missing_content):
    """GIVEN a provider response with no content.
    WHEN the client operation is called.
    THEN missing content explains error."""
    client = missing_content
    reply = client.chat_completions({})
    expected = "no content"
    assert expected in str(reply.error)


@pytest.fixture
def missing_response(client):
    client.client.chat.completions.create.return_value = None
    return client


def test_missing_response_is_error(missing_response):
    """GIVEN no provider response.
    WHEN the client operation is called.
    THEN the reply reports that no response was received."""
    client = missing_response
    result = "No response" in str(client.chat_completions({}).error)
    expected = True
    assert result == expected


@pytest.fixture
def missing_usage(client, response_factory):
    response = response_factory()
    del response.usage
    client.client.chat.completions.create.return_value = response
    return client


def test_usage_requires_provider_counts(missing_usage):
    """GIVEN a provider response with absent usage fields.
    WHEN the client operation is called.
    THEN the absent token counts are reported."""
    client = missing_usage
    with pytest.raises(RuntimeError, match="did not report token usage") as exc:
        client.chat_completions({})
    expected = RuntimeError
    assert exc.type == expected


@pytest.fixture
def missing_reasoning(client, response_factory, absent):
    response = response_factory()
    if absent:
        del response.usage.completion_tokens_details
    else:
        response.usage.completion_tokens_details = None
    client.client.chat.completions.create.return_value = response
    return client


@pytest.mark.parametrize("absent", [True, False])
def test_usage_without_reasoning_details(missing_reasoning, absent):
    """GIVEN a provider response with absent usage fields.
    WHEN the client operation is called.
    THEN the reasoning count is zero."""
    client = missing_reasoning
    result = client.chat_completions({}).usage.reasoning_tokens
    expected = 0
    assert result == expected


@pytest.fixture
def empty_choices(client, response_factory):
    response = response_factory()
    response.choices = []
    client.client.chat.completions.create.return_value = response
    return client


def test_empty_choices_returns_error(empty_choices):
    """GIVEN a response with no choices.
    WHEN the client operation is called.
    THEN the reply reports an error."""
    client = empty_choices
    result = isinstance(client.chat_completions({}).error, RuntimeError)
    expected = True
    assert result == expected


@pytest.fixture
def chat_params(client):
    params = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    return (params, client)


@pytest.mark.parametrize("method,operation", [("chat_completions", "create"), ("chat_completions_body", "create")])
def test_sync_chat_passes_parameters(chat_params, method, operation):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN sync chat passes parameters."""
    params, client = chat_params
    getattr(client, method)(params)
    result = getattr(client.client.chat.completions, operation).call_args_list
    expected = [mock_call(**params)]
    assert result == expected


@pytest.mark.parametrize("method,operation", [("chat_completions", "create"), ("chat_completions_body", "create")])
def test_sync_chat_returns_text(chat_params, method, operation):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN sync chat returns text."""
    params, client = chat_params
    reply = getattr(client, method)(params)
    expected = "hello"
    assert reply.content == expected


@pytest.mark.parametrize("input_type,operation", [(InputType.JSON, "create"), (InputType.PYDANTIC, "parse")])
def test_sync_body_has_no_error(client, input_type, operation):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN sync body has no error."""
    expected = None
    assert client.chat_completions_body({"model": "gpt-4o"}, input_type=input_type).error is expected


@pytest.mark.parametrize("input_type,operation", [(InputType.JSON, "create"), (InputType.PYDANTIC, "parse")])
def test_sync_body_selects_endpoint(client, input_type, operation):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN sync body selects endpoint."""
    client.chat_completions_body({"model": "gpt-4o"}, input_type=input_type)
    expected = [mock_call(model="gpt-4o")]
    assert getattr(client.client.chat.completions, operation).call_args_list == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type,operation", [("json", "create"), ("pydantic", "parse")])
async def test_async_body_selects_endpoint(client, input_type, operation):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN async body selects endpoint."""
    await client.async_chat_completions_body({"model": "gpt-4o"}, input_type=input_type)
    result = getattr(client.async_client.chat.completions, operation).await_args_list
    expected = [mock_call(model="gpt-4o")]
    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type,operation", [("json", "create"), ("pydantic", "parse")])
async def test_async_body_returns_text(client, input_type, operation):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN async body returns text."""
    reply = await client.async_chat_completions_body({"model": "gpt-4o"}, input_type=input_type)
    expected = "hello"
    assert reply.content == expected


@pytest.mark.asyncio
async def test_async_chat_passes_parameters(client):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN async chat passes parameters."""
    await client.async_chat_completions({"model": "gpt-4o"})
    expected = [mock_call(model="gpt-4o")]
    assert client.async_client.chat.completions.create.await_args_list == expected


@pytest.mark.asyncio
async def test_async_chat_returns_text(client):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN async chat returns text."""
    reply = await client.async_chat_completions({"model": "gpt-4o"})
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
    result = getattr(client, method)({}).error is error
    expected = True
    assert result == expected


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
    result = (await getattr(client, method)({})).error is error
    expected = True
    assert result == expected


def test_invalid_sync_body_returns_error(client):
    """GIVEN an invalid input type.
    WHEN the client operation is called.
    THEN invalid sync body returns error."""
    result = isinstance(client.chat_completions_body({}, input_type="bad").error, ValueError)
    expected = True
    assert result == expected


def test_invalid_sync_body_avoids_provider(client):
    """GIVEN an invalid input type.
    WHEN the client operation is called.
    THEN invalid sync body avoids provider."""
    isinstance(client.chat_completions_body({}, input_type="bad").error, ValueError)
    expected = []
    assert client.client.chat.completions.create.call_args_list == expected


@pytest.mark.asyncio
async def test_invalid_async_body_returns_error(client):
    """GIVEN an invalid input type.
    WHEN the client operation is called.
    THEN invalid async body returns error."""
    result = isinstance((await client.async_chat_completions_body({}, input_type="bad")).error, ValueError)
    expected = True
    assert result == expected


@pytest.mark.asyncio
async def test_invalid_async_body_avoids_provider(client):
    """GIVEN an invalid input type.
    WHEN the client operation is called.
    THEN invalid async body avoids provider."""
    isinstance((await client.async_chat_completions_body({}, input_type="bad")).error, ValueError)
    expected = []
    assert client.async_client.chat.completions.create.await_args_list == expected


@pytest.fixture
def embedding_params(client):
    params = dict(input="text", model="test", encoding_format="float", dimensions=2)
    return (params, client)


@pytest.mark.asyncio
async def test_client_embedding_returns_vector(embedding_params):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN client embedding returns vector."""
    params, client = embedding_params
    result = await client.make_embeddings(**params)
    expected = [0.1, 0.2]
    assert result == expected


@pytest.mark.asyncio
async def test_client_embedding_passes_parameters(embedding_params):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN client embedding passes parameters."""
    params, client = embedding_params
    await client.make_embeddings(**params)
    expected = [mock_call(**params)]
    assert client.async_client.embeddings.create.await_args_list == expected


@pytest.fixture
def failed_embedding(client):
    error = RuntimeError("embeddings failed")
    client.async_client.embeddings.create.side_effect = error
    return (error, client)


@pytest.mark.asyncio
async def test_embedding_error(failed_embedding):
    """GIVEN an operation that reports an error.
    WHEN the client operation is called.
    THEN the result retains the provider error."""
    error, client = failed_embedding
    result = await client.make_embeddings(input="text", model="test", encoding_format="float", dimensions=2) is error
    expected = True
    assert result == expected


@pytest.mark.parametrize("vendor", ["OPENAI", "google", "brave"])
def test_client_factory_normalizes_vendor(client, vendor):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN client factory normalizes vendor."""
    result = clients.make_client(vendor, client=client.client, async_client=client.async_client)
    expected = vendor.lower()
    assert result.vendor == expected


@pytest.mark.parametrize("vendor", ["OPENAI", "google", "brave"])
def test_client_factory_retains_sync_client(client, vendor):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN client factory retains sync client."""
    result = clients.make_client(vendor, client=client.client, async_client=client.async_client)
    expected = client.client
    assert result.client is expected


@pytest.mark.parametrize("vendor", ["OPENAI", "google", "brave"])
def test_client_factory_retains_async_client(client, vendor):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN client factory retains async client."""
    result = clients.make_client(vendor, client=client.client, async_client=client.async_client)
    expected = client.async_client
    assert result.async_client is expected


@pytest.fixture
def vendor_clients(monkeypatch, client, factory):
    sync, asynchronous = (Mock(), Mock())
    monkeypatch.setattr(clients, factory, lambda: sync)
    monkeypatch.setattr(clients, "make_async_openai_client", lambda: asynchronous)
    return (asynchronous, sync, client)


@pytest.mark.parametrize("vendor,factory", [("openai", "make_openai_client"), ("google", "make_google_client"), ("brave", "make_brave_client")])
def test_vendor_change_sets_vendor(vendor_clients, vendor, factory):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN vendor change sets vendor."""
    asynchronous, sync, client = vendor_clients
    client.set_vendor(vendor.upper())
    expected = vendor
    assert client.vendor == expected


@pytest.mark.parametrize("vendor,factory", [("openai", "make_openai_client"), ("google", "make_google_client"), ("brave", "make_brave_client")])
def test_vendor_change_replaces_sync_client(vendor_clients, vendor, factory):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN vendor change replaces sync client."""
    asynchronous, sync, client = vendor_clients
    client.set_vendor(vendor.upper())
    expected = sync
    assert client.client is expected


@pytest.mark.parametrize("vendor,factory", [("openai", "make_openai_client"), ("google", "make_google_client"), ("brave", "make_brave_client")])
def test_vendor_change_replaces_async_client(vendor_clients, vendor, factory):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN vendor change replaces async client."""
    asynchronous, sync, client = vendor_clients
    client.set_vendor(vendor.upper())
    expected = asynchronous
    assert client.async_client is expected


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
    result = client.responses_create("hi", model="test").error is error
    expected = True
    assert result == expected


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
    result = (await client.async_responses_create("hi", model="test")).error is error
    expected = True
    assert result == expected


@pytest.fixture
def response_text(client):
    client.client.responses.create.return_value = SimpleNamespace(output_text="hello", output=[], usage=SimpleNamespace(input_tokens=10, output_tokens=5))
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
    expected = None
    assert getattr(client, method)({}, verbose=True).error is expected


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
    expected = None
    assert (await getattr(client, method)({}, verbose=True)).error is expected


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
def response_logger(client, response_factory, asynchronous):
    response = response_factory()
    response.model_dump = Mock(return_value={"id": "response-id"})
    client.logger = Mock()
    sdk = client.async_client if asynchronous else client.client
    create = sdk.responses.create
    create.return_value = response
    return (create, client)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_pass_options(response_logger, asynchronous):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN responses pass options."""
    create, client = response_logger
    reply = await client.async_responses_create("question", model="test", temperature=0.2, verbose=True) if asynchronous else client.responses_create("question", model="test", temperature=0.2, verbose=True)
    expected = [mock_call(input="question", model="test", temperature=0.2)]
    assert create.call_args_list == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_return_text(response_logger, asynchronous):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN responses return text."""
    create, client = response_logger
    reply = await client.async_responses_create("question", model="test", temperature=0.2, verbose=True) if asynchronous else client.responses_create("question", model="test", temperature=0.2, verbose=True)
    expected = "hello"
    assert reply.content == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_log_output(response_logger, asynchronous):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN responses log output."""
    create, client = response_logger
    reply = await client.async_responses_create("question", model="test", temperature=0.2, verbose=True) if asynchronous else client.responses_create("question", model="test", temperature=0.2, verbose=True)
    expected = [mock_call({"id": "response-id"})]
    assert client.logger.debug.call_args_list == expected


@pytest.fixture
def response_tools(client, response_factory, asynchronous):
    tools = [{"type": "web_search_preview"}]
    sdk = client.async_client if asynchronous else client.client
    sdk.responses.create.return_value = response_factory()
    return (sdk, tools, client)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_forwards_tools(response_tools, asynchronous):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN the request retains the specified tools."""
    sdk, tools, client = response_tools
    reply = await client.async_responses_create("q", model="test", tools=tools) if asynchronous else client.responses_create("q", model="test", tools=tools)
    result = sdk.responses.create.call_args.kwargs.get("tools")
    expected = tools
    assert result == expected


@pytest.fixture
def missing_clients(monkeypatch, client, sync_missing):
    monkeypatch.setattr(clients, "make_openai_client", lambda: None if sync_missing else client.client)
    monkeypatch.setattr(clients, "make_async_openai_client", lambda: client.async_client if sync_missing else None)
    return client


@pytest.mark.parametrize("sync_missing", [True, False])
def test_set_vendor_rejects_unavailable_clients(missing_clients, sync_missing):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN the unavailable client is reported."""
    client = missing_clients
    with pytest.raises(RuntimeError, match="Sync client" if sync_missing else "Async client") as exc:
        client.set_vendor("openai")
    expected = RuntimeError
    assert exc.type == expected


@pytest.fixture
def original_clients(client):
    original_sync, original_async = (client.client, client.async_client)
    return (original_async, original_sync, client)


def test_invalid_vendor_retains_sync_client(original_clients):
    """GIVEN a configured client and an unknown vendor.
    WHEN the client operation is called.
    THEN invalid vendor retains sync client."""
    original_async, original_sync, client = original_clients
    with pytest.raises(ValueError):
        client.set_vendor("unknown")
    expected = original_sync
    assert client.client is expected


def test_invalid_vendor_retains_async_client(original_clients):
    """GIVEN a configured client and an unknown vendor.
    WHEN the client operation is called.
    THEN invalid vendor retains async client."""
    original_async, original_sync, client = original_clients
    with pytest.raises(ValueError):
        client.set_vendor("unknown")
    expected = original_async
    assert client.async_client is expected


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
    expected = True
    assert (exc.value.__cause__ is error) == expected


@pytest.fixture
def patch_response(client, asynchronous):
    operation = SimpleNamespace(type="apply_patch_call", id="patch-1")
    response = SimpleNamespace(output_text="", output=[operation], usage=SimpleNamespace(input_tokens=10, output_tokens=5, output_tokens_details=SimpleNamespace(reasoning_tokens=3)))
    sdk = client.async_client if asynchronous else client.client
    sdk.responses.create.return_value = response
    tools = [{"type": "apply_patch"}]
    return (operation, response, sdk, tools, client)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_pass_input(patch_response, asynchronous):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN responses pass input."""
    operation, response, sdk, tools, client = patch_response
    reply = await client.async_responses_create(input="edit", model="test", tools=tools) if asynchronous else client.responses_create(input="edit", model="test", tools=tools)
    expected = [mock_call(input="edit", model="test", tools=tools)]
    assert sdk.responses.create.call_args_list == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_have_no_error(patch_response, asynchronous):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN responses have no error."""
    operation, response, sdk, tools, client = patch_response
    reply = await client.async_responses_create(input="edit", model="test", tools=tools) if asynchronous else client.responses_create(input="edit", model="test", tools=tools)
    expected = None
    assert reply.error is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_retain_provider_reply(patch_response, asynchronous):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN responses retain provider reply."""
    operation, response, sdk, tools, client = patch_response
    reply = await client.async_responses_create(input="edit", model="test", tools=tools) if asynchronous else client.responses_create(input="edit", model="test", tools=tools)
    expected = response
    assert reply.response is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_retain_operations(patch_response, asynchronous):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN responses retain operations."""
    operation, response, sdk, tools, client = patch_response
    reply = await client.async_responses_create(input="edit", model="test", tools=tools) if asynchronous else client.responses_create(input="edit", model="test", tools=tools)
    expected = [operation]
    assert reply.response.output == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_count_input(patch_response, asynchronous):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN responses count input."""
    operation, response, sdk, tools, client = patch_response
    reply = await client.async_responses_create(input="edit", model="test", tools=tools) if asynchronous else client.responses_create(input="edit", model="test", tools=tools)
    expected = 10
    assert reply.usage.input_tokens == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_count_output(patch_response, asynchronous):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN responses count output."""
    operation, response, sdk, tools, client = patch_response
    reply = await client.async_responses_create(input="edit", model="test", tools=tools) if asynchronous else client.responses_create(input="edit", model="test", tools=tools)
    expected = 5
    assert reply.usage.output_tokens == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_count_reasoning(patch_response, asynchronous):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN responses count reasoning."""
    operation, response, sdk, tools, client = patch_response
    reply = await client.async_responses_create(input="edit", model="test", tools=tools) if asynchronous else client.responses_create(input="edit", model="test", tools=tools)
    expected = 3
    assert reply.usage.reasoning_tokens == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_ambiguous_input_avoids_sync_provider(client, asynchronous):
    """GIVEN both prompt and input values.
    WHEN the client operation is called.
    THEN ambiguous input avoids sync provider."""
    with pytest.raises(TypeError, match="either prompt or input"):
        reply = await client.async_responses_create("first", input="second", model="test") if asynchronous else client.responses_create("first", input="second", model="test")
    expected = []
    assert client.client.responses.create.call_args_list == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_ambiguous_input_avoids_async_provider(client, asynchronous):
    """GIVEN both prompt and input values.
    WHEN the client operation is called.
    THEN ambiguous input avoids async provider."""
    with pytest.raises(TypeError, match="either prompt or input"):
        reply = await client.async_responses_create("first", input="second", model="test") if asynchronous else client.responses_create("first", input="second", model="test")
    expected = []
    assert client.async_client.responses.create.call_args_list == expected
