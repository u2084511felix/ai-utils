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
def test_vendor_configuration(vendor_sdk, factory, key, url):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN the vendor client uses the specified key and service URL."""
    constructor = vendor_sdk
    result = getattr(clients, factory)()
    result = result is constructor.return_value
    actual = (result, constructor.call_args_list)
    expected = (True, [mock_call(api_key="fake-key", base_url=url)])
    assert actual == expected


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
def test_client_factories_use_sdk_constructor(sdk_factory, factory, sdk_name):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN the result is the SDK client with the configured key."""
    constructor = sdk_factory
    result = getattr(clients, factory)() is constructor.return_value
    actual = (result, constructor.call_args.kwargs["api_key"])
    expected = (True, "pytest-not-a-real-key")
    assert actual == expected


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


def test_chat_text_and_usage(client):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN the reply contains the provider text and token counts."""
    reply = client.chat_completions({})
    expected = ("hello", None, 10, 5, 2)
    assert (reply.content, reply.error, reply.usage.input_tokens, reply.usage.output_tokens, reply.usage.reasoning_tokens) == expected


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


def test_missing_content_is_error(missing_content):
    """GIVEN a provider response with no content.
    WHEN the client operation is called.
    THEN the reply reports that content is absent."""
    client = missing_content
    reply = client.chat_completions({})
    result = isinstance(reply.error, RuntimeError)
    actual = (result, "no content" in str(reply.error))
    expected = (True, True)
    assert actual == expected


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
def test_sync_chat_forwards_parameters(chat_params, method, operation):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN the provider receives the specified parameters and the reply contains text."""
    params, client = chat_params
    reply = getattr(client, method)(params)
    result = getattr(client.client.chat.completions, operation).call_args_list
    actual = (result, reply.content)
    expected = ([mock_call(**params)], "hello")
    assert actual == expected


@pytest.mark.parametrize("input_type,operation", [(InputType.JSON, "create"), (InputType.PYDANTIC, "parse")])
def test_sync_body_dispatch(client, input_type, operation):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN the selected endpoint receives the request."""
    result = client.chat_completions_body({"model": "gpt-4o"}, input_type=input_type).error is None
    actual = (result, getattr(client.client.chat.completions, operation).call_args_list)
    expected = (True, [mock_call(model="gpt-4o")])
    assert actual == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type,operation", [("json", "create"), ("pydantic", "parse")])
async def test_async_body_dispatch(client, input_type, operation):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN the selected endpoint receives the request and the reply contains text."""
    reply = await client.async_chat_completions_body({"model": "gpt-4o"}, input_type=input_type)
    result = getattr(client.async_client.chat.completions, operation).await_args_list
    actual = (result, reply.content)
    expected = ([mock_call(model="gpt-4o")], "hello")
    assert actual == expected


@pytest.mark.asyncio
async def test_async_chat(client):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN the provider receives the request and the reply contains text."""
    reply = await client.async_chat_completions({"model": "gpt-4o"})
    actual = (client.async_client.chat.completions.create.await_args_list, reply.content)
    expected = ([mock_call(model="gpt-4o")], "hello")
    assert actual == expected


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


def test_sync_invalid_input_type_does_not_call_provider(client):
    """GIVEN an invalid input type.
    WHEN the client operation is called.
    THEN the input type is rejected before a provider request."""
    result = isinstance(client.chat_completions_body({}, input_type="bad").error, ValueError)
    actual = (result, client.client.chat.completions.create.call_args_list)
    expected = (True, [])
    assert actual == expected


@pytest.mark.asyncio
async def test_async_invalid_input_type_does_not_call_provider(client):
    """GIVEN an invalid input type.
    WHEN the client operation is called.
    THEN the input type is rejected before a provider request."""
    result = isinstance((await client.async_chat_completions_body({}, input_type="bad")).error, ValueError)
    actual = (result, client.async_client.chat.completions.create.await_args_list)
    expected = (True, [])
    assert actual == expected


@pytest.fixture
def embedding_params(client):
    params = dict(input="text", model="test", encoding_format="float", dimensions=2)
    return (params, client)


@pytest.mark.asyncio
async def test_embedding_response(embedding_params):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN the provider receives the specified parameters and the result contains the embedding."""
    params, client = embedding_params
    result = await client.make_embeddings(**params)
    actual = (result, client.async_client.embeddings.create.await_args_list)
    expected = ([0.1, 0.2], [mock_call(**params)])
    assert actual == expected


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
def test_make_client_accepts_explicit_clients(client, vendor):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN the client retains the supplied SDK clients and normalizes the vendor name."""
    result = clients.make_client(vendor, client=client.client, async_client=client.async_client)
    actual = (result.vendor, result.client is client.client, result.async_client is client.async_client)
    expected = (vendor.lower(), True, True)
    assert actual == expected


@pytest.fixture
def vendor_clients(monkeypatch, client, factory):
    sync, asynchronous = (Mock(), Mock())
    monkeypatch.setattr(clients, factory, lambda: sync)
    monkeypatch.setattr(clients, "make_async_openai_client", lambda: asynchronous)
    return (asynchronous, sync, client)


@pytest.mark.parametrize("vendor,factory", [("openai", "make_openai_client"), ("google", "make_google_client"), ("brave", "make_brave_client")])
def test_set_vendor_switches_clients(vendor_clients, vendor, factory):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN the vendor and both SDK clients change."""
    asynchronous, sync, client = vendor_clients
    client.set_vendor(vendor.upper())
    actual = (client.vendor, client.client is sync, client.async_client is asynchronous)
    expected = (vendor, True, True)
    assert actual == expected


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
def test_sync_chat_verbose_logs_provider_response(sync_logger, method):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN the response is logged and returned without an error."""
    response, client = sync_logger
    result = getattr(client, method)({}, verbose=True).error is None
    actual = (result, client.logger.debug.call_args_list)
    expected = (True, [mock_call(response)])
    assert actual == expected


@pytest.fixture
def async_logger(client):
    client.logger = Mock()
    response = client.async_client.chat.completions.create.return_value
    return (response, client)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["async_chat_completions", "async_chat_completions_body"])
async def test_async_chat_verbose_logs_provider_response(async_logger, method):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN the response is logged and returned without an error."""
    response, client = async_logger
    result = (await getattr(client, method)({}, verbose=True)).error is None
    actual = (result, client.logger.debug.call_args_list)
    expected = (True, [mock_call(response)])
    assert actual == expected


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
async def test_responses_forwards_options_and_logs_dump(response_logger, asynchronous):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN the request retains its options and the response is logged."""
    create, client = response_logger
    reply = await client.async_responses_create("question", model="test", temperature=0.2, verbose=True) if asynchronous else client.responses_create("question", model="test", temperature=0.2, verbose=True)
    actual = (create.call_args_list, reply.content, client.logger.debug.call_args_list)
    expected = ([mock_call(input="question", model="test", temperature=0.2)], "hello", [mock_call({"id": "response-id"})])
    assert actual == expected


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


def test_invalid_vendor_does_not_replace_clients(original_clients):
    """GIVEN a configured client and an unknown vendor.
    WHEN the client operation is called.
    THEN the vendor is rejected and both clients are retained."""
    original_async, original_sync, client = original_clients
    with pytest.raises(ValueError):
        client.set_vendor("unknown")
    actual = (client.client is original_sync, client.async_client is original_async)
    expected = (True, True)
    assert actual == expected


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
async def test_responses_preserves_output_usage_and_accepts_input(patch_response, asynchronous):
    """GIVEN a client and a provider response.
    WHEN the client operation is called.
    THEN the reply retains the provider output and token counts."""
    operation, response, sdk, tools, client = patch_response
    reply = await client.async_responses_create(input="edit", model="test", tools=tools) if asynchronous else client.responses_create(input="edit", model="test", tools=tools)
    actual = (sdk.responses.create.call_args_list, reply.error is None, reply.response is response, reply.response.output, reply.usage.input_tokens, reply.usage.output_tokens, reply.usage.reasoning_tokens)
    expected = ([mock_call(input="edit", model="test", tools=tools)], True, True, [operation], 10, 5, 3)
    assert actual == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_rejects_ambiguous_input(client, asynchronous):
    """GIVEN both prompt and input values.
    WHEN the client operation is called.
    THEN the two input values are rejected before a provider request."""
    with pytest.raises(TypeError, match="either prompt or input"):
        reply = await client.async_responses_create("first", input="second", model="test") if asynchronous else client.responses_create("first", input="second", model="test")
    actual = (client.client.responses.create.call_args_list, client.async_client.responses.create.call_args_list)
    expected = ([], [])
    assert actual == expected
