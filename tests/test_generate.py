import json
from unittest.mock import AsyncMock, Mock, call as mock_call

from pydantic import BaseModel
import pytest
import pytest_asyncio

from aiutils.clients import ClientReply, InputType
from aiutils.exceptions import ClientError, InitializationError
from aiutils.generate import make_generate
from aiutils.models import TextModels, Vendor
from aiutils.modules.gpt_module import GPTModule, make_req_body


@pytest.fixture
def generator(client, monkeypatch):
    monkeypatch.setattr(client, "async_chat_completions", AsyncMock(return_value=ClientReply(content="answer")))
    monkeypatch.setattr(client, "async_chat_completions_body", AsyncMock(return_value=ClientReply(content='{"ok": true}')))
    return make_generate(client=client)


@pytest.fixture
def request_module():
    module = GPTModule(model="gpt-4o", temperature=0, stream=False, max_tokens=20, request_body={"old": True})
    return module


def test_request_body_keeps_zero_and_false_but_omits_empty_values(request_module):
    """GIVEN zero, false, and empty request fields.
    WHEN the request body is created.
    THEN the request retains zero and false values and omits empty fields."""
    module = request_module
    result = make_req_body(module)
    expected = {"model": "gpt-4o", "temperature": 0, "stream": False, "max_tokens": 20}
    assert result == expected


@pytest.fixture
def changed_module():
    first = GPTModule()
    first.messages.append({"role": "user", "content": "hi"})
    first.logit_bias["1"] = 1
    return first


def test_module_mutable_defaults_are_independent(changed_module):
    """GIVEN a module with changed messages and bias.
    WHEN another module is created.
    THEN the new module has empty message and bias fields."""
    second = GPTModule()
    actual = (second.messages, second.logit_bias)
    expected = ([], {})
    assert actual == expected


@pytest.fixture
def vendor_generator(generator, vendor):
    generator.client.vendor = vendor
    return generator


@pytest.mark.asyncio
@pytest.mark.parametrize("model,vendor", [("gpt-4o", Vendor.OPENAI), ("gemini-2.5-pro", Vendor.GOOGLE)])
async def test_model_vendor_match(vendor_generator, model, vendor):
    """GIVEN a generator and specified request options.
    WHEN a response is requested.
    THEN the request uses the specified model."""
    generator = vendor_generator
    await generator.generate("s", "p", model=model)
    expected = model
    assert generator.request_body["model"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("model,vendor", [("missing", Vendor.OPENAI), ("gpt-4o", Vendor.GOOGLE), ("gemini-2.5-pro", Vendor.OPENAI)])
async def test_wrong_model_or_vendor_rejected(vendor_generator, model, vendor):
    """GIVEN a model that is not available for the vendor.
    WHEN a response is requested.
    THEN the model is rejected before a provider request."""
    generator = vendor_generator
    with pytest.raises(ValueError, match="Invalid model"):
        await generator.generate("s", "p", model=model)
    expected = []
    assert generator.client.async_chat_completions.await_args_list == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [True, False])
async def test_generate_builds_request_and_selects_endpoint(generator, chat):
    """GIVEN a generator and specified request options.
    WHEN a response is requested.
    THEN the selected endpoint receives the specified messages and options."""
    reply = await generator.generate("instructions", "question", model="gpt-4o", temperature=0.4, chat=chat)
    expected = dict(model="gpt-4o", temperature=0.4, messages=[{"role": "system", "content": "instructions"}, {"role": "user", "content": "question"}])
    method = generator.client.async_chat_completions if chat else generator.client.async_chat_completions_body
    actual = (method.await_args_list, reply is method.return_value)
    expected = ([mock_call(expected)], True)
    assert actual == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("system,prompt,temp,error", [("", "q", 0, "system_message"), ("s", "", 0, "prompt"), ("s", "q", -1, "temperature")])
async def test_generate_validates_before_calling_client(generator, system, prompt, temp, error):
    """GIVEN an empty message or a negative temperature.
    WHEN a response is requested.
    THEN the input is rejected before messages or provider requests are added."""
    with pytest.raises(ValueError, match=error):
        await generator.generate(system, prompt, temperature=temp)
    actual = (generator.client.async_chat_completions.await_args_list, generator.messages)
    expected = ([], [])
    assert actual == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("model,effort,expected,temp", [(TextModels.gpt_5_2, "default", "none", 1), (TextModels.gpt_5, "default", "minimal", 1), (TextModels.gpt_5_2, "high", "high", 1), (TextModels.gpt_5, "low", "low", 1), (TextModels.gpt_4o, "default", None, 0.3)])
async def test_reasoning_settings(generator, model, effort, expected, temp):
    """GIVEN a generator and specified request options.
    WHEN a response is requested.
    THEN the request uses the specified reasoning and temperature values."""
    await generator.generate("s", "p", model=model, reasoning_effort=effort, temperature=0.3)
    actual = (generator.reasoning_effort, generator.temperature)
    expected = (expected, temp)
    assert actual == expected


@pytest_asyncio.fixture
async def conversation(generator):
    await generator.generate("s", "first", model="gpt-4o")
    return generator


@pytest.mark.asyncio
async def test_continued_response_keeps_conversation(conversation):
    """GIVEN an existing conversation and a new exchange.
    WHEN a response is requested.
    THEN the request includes the previous messages and the new exchange."""
    generator = conversation
    await generator.continued_response("answer", "second")
    actual = (generator.request_body["messages"], generator.client.async_chat_completions.await_count)
    expected = ([{"role": "system", "content": "s"}, {"role": "user", "content": "first"}, {"role": "assistant", "content": "answer"}, {"role": "user", "content": "second"}], 2)
    assert actual == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("answer,prompt", [("", "p"), ("a", "")])
async def test_continuation_validates_input(generator, answer, prompt):
    """GIVEN an empty assistant response or prompt.
    WHEN a response is requested.
    THEN the input is rejected before a provider request."""
    with pytest.raises(ValueError):
        await generator.continued_response(answer, prompt)
    expected = []
    assert generator.client.async_chat_completions.await_args_list == expected


@pytest.fixture
def json_schema(generator):
    schema = {"name": "result", "schema": {"type": "object"}}
    return (schema, generator)


@pytest.mark.asyncio
async def test_json_structured_output(json_schema):
    """GIVEN a generator and specified request options.
    WHEN a response is requested.
    THEN the request uses the JSON schema and returns the provider reply."""
    schema, generator = json_schema
    reply = await generator.structured_output("s", "p", schema=json.dumps(schema), model="gpt-4o")
    actual = (reply.content, generator.request_body["response_format"], generator.client.async_chat_completions_body.await_args_list)
    expected = ('{"ok": true}', {"type": "json_schema", "json_schema": schema}, [mock_call(generator.request_body, input_type=InputType.JSON)])
    assert actual == expected


@pytest.fixture
def pydantic_schema(generator):

    class Result(BaseModel):
        ok: bool

    return (Result, generator)


@pytest.mark.asyncio
async def test_pydantic_structured_output_and_assistant_context(pydantic_schema):
    """GIVEN a generator and specified request options.
    WHEN a response is requested.
    THEN the request uses the model class and previous assistant message."""
    Result, generator = pydantic_schema
    await generator.structured_output("s", "p", schema=Result, input_type="pydantic", model="gpt-4o", assistant_response="previous")
    actual = (generator.response_format is Result, generator.messages[0], generator.client.async_chat_completions_body.await_args_list)
    expected = (True, {"role": "assistant", "content": "previous"}, [mock_call(generator.request_body, input_type=InputType.PYDANTIC)])
    assert actual == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("schema,error", [({}, TypeError), ("not json", json.JSONDecodeError)])
async def test_invalid_json_schema(generator, schema, error):
    """GIVEN an invalid JSON schema.
    WHEN a response is requested.
    THEN the schema is rejected before a provider request."""
    with pytest.raises(error):
        await generator.structured_output("s", "p", schema=schema)
    expected = []
    assert generator.client.async_chat_completions_body.await_args_list == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("system,prompt,input_type", [("", "p", "json"), ("s", "", "json"), ("s", "p", "invalid")])
async def test_structured_output_invalid_arguments(generator, system, prompt, input_type):
    """GIVEN an empty message or invalid input type.
    WHEN a response is requested.
    THEN the input is rejected before a provider request."""
    with pytest.raises(ValueError):
        await generator.structured_output(system, prompt, schema="{}", input_type=input_type)
    expected = []
    assert generator.client.async_chat_completions_body.await_args_list == expected


@pytest.fixture
def failed_generator(generator, method):
    error = RuntimeError("provider failed")
    generator.client.async_chat_completions.return_value = ClientReply(error=error)
    generator.client.async_chat_completions_body.return_value = ClientReply(error=error)
    kwargs = {"schema": "{}"} if method == "structured_output" else {}
    return (error, kwargs, generator)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "continued_response", "structured_output"])
async def test_provider_error_preserves_cause(failed_generator, method):
    """GIVEN an operation that reports an error.
    WHEN a response is requested.
    THEN the error retains the provider error as its cause."""
    error, kwargs, generator = failed_generator
    with pytest.raises(ClientError) as exc:
        await getattr(generator, method)("s", "p", **kwargs)
    expected = True
    assert (exc.value.__cause__ is error) == expected


def test_factory_applies_options(client):
    """GIVEN a generator and specified request options.
    WHEN a generator is created.
    THEN the new generator has the specified token limit."""
    result = make_generate(client=client, max_tokens=42).max_tokens
    expected = 42
    assert result == expected


def test_factory_wraps_errors(client):
    """GIVEN an operation that reports an error.
    WHEN a generator is created.
    THEN the error retains the original validation error."""
    with pytest.raises(InitializationError) as exc:
        make_generate(client=client, nonexistent_option=True)
    result = isinstance(exc.value.__cause__, ValueError)
    expected = True
    assert result == expected


@pytest_asyncio.fixture
async def reasoning_generator(generator):
    await generator.generate("s", "p", model="gpt-5.2")
    return generator


@pytest.mark.asyncio
async def test_switching_models_clears_reasoning_settings(reasoning_generator):
    """GIVEN a generator that used a reasoning model.
    WHEN a response is requested.
    THEN the new request has no reasoning setting."""
    generator = reasoning_generator
    await generator.generate("s", "p", model="gpt-4o")
    expected = True
    assert ("reasoning_effort" not in generator.request_body) == expected


@pytest.fixture
def logged_generator(generator):
    generator.logger = Mock()
    return generator


@pytest.mark.asyncio
@pytest.mark.parametrize("assistant", [None, "previous"])
async def test_structured_debug_logs_response_and_context(logged_generator, assistant):
    """GIVEN a generator and specified request options.
    WHEN a response is requested.
    THEN the response and any previous context are logged."""
    generator = logged_generator
    reply = await generator.structured_output("s", "p", schema="{}", assistant_response=assistant, debug=True)
    result = mock_call(reply) in generator.logger.debug.call_args_list
    actual = (result, generator.logger.debug.call_count)
    expected = (True, 1 if assistant is None else 2)
    assert actual == expected


@pytest.fixture
def schema_module(tmp_path, monkeypatch):
    import sys

    module_name = "pytest_generated_schema"
    (tmp_path / f"{module_name}.py").write_text("from pydantic import BaseModel\nclass Result(BaseModel):\n    value: str\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, module_name, None)
    monkeypatch.delitem(sys.modules, module_name)
    return module_name


@pytest.mark.asyncio
async def test_pydantic_schema_can_be_loaded_from_module(generator, schema_module):
    """GIVEN a generator and specified request options.
    WHEN a response is requested.
    THEN the request uses the imported schema class."""
    await generator.structured_output("s", "p", schema="Result", input_type="pydantic", module_name=schema_module)
    expected = ("Result", str, True)
    assert (generator.response_format.__name__, generator.response_format.model_fields["value"].annotation, generator.request_body["response_format"] is generator.response_format) == expected


@pytest.mark.asyncio
async def test_missing_schema_module_does_not_call_provider(generator):
    """GIVEN a schema module that does not exist.
    WHEN a response is requested.
    THEN the missing module is reported before a provider request."""
    with pytest.raises(ModuleNotFoundError):
        await generator.structured_output("s", "p", schema="Result", input_type="pydantic", module_name="nonexistent_aiutils_schema_module")
    expected = []
    assert generator.client.async_chat_completions_body.await_args_list == expected
