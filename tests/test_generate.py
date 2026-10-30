from aiutils.budget import TokenUsage
import json
from unittest.mock import AsyncMock, Mock, call as mock_call

from pydantic import BaseModel
import pytest
import pytest_asyncio

from aiutils.clients import ClientReply, InputType
from aiutils.exceptions import ClientError, InitializationError
from aiutils.generate import make_generate, Generate
from aiutils.models import TextModels, Vendor
from aiutils.modules.gpt_module import GPTModule, make_req_body


@pytest.fixture
def gen(client, monkeypatch) -> Generate:
    monkeypatch.setattr(client, "async_chat_completions", AsyncMock(return_value=ClientReply(usage=TokenUsage(), content="answer")))
    monkeypatch.setattr(client, "async_chat_completions_body", AsyncMock(return_value=ClientReply(usage=TokenUsage(), content='{"ok": true}')))
    return make_generate(client=client)

@pytest.fixture
def request_body():
    return {"old": True}

@pytest.fixture
def gpt_module_kwargs():
    return {
        "model": "gpt-4o",
        "temperature": 0,
        "stream": False,
        "max_tokens": 20,
    }

@pytest.fixture
def request_module(gpt_module_kwargs, request_body):
    module = GPTModule(**gpt_module_kwargs, request_body=request_body)
    return module


def test_request_body_keeps_zero_and_false_but_omits_empty_values(
    request_module,
    gpt_module_kwargs
):
    """GIVEN zero, false, and empty request fields.
    WHEN the request body is created.
    THEN the request retains zero and false values and omits empty fields."""
    module = request_module
    result = make_req_body(module)
    assert result == gpt_module_kwargs


@pytest.fixture
def changed_module():
    first = GPTModule()
    first.messages.append({"role": "user", "content": "hi"})
    if first.logit_bias is None:
        first.logit_bias = {}
    first.logit_bias["1"] = 1
    return first


def test_module_has_independent_messages(changed_module):
    """GIVEN a module with changed messages and bias.
    WHEN another module is created.
    THEN module has independent messages."""
    second = GPTModule()
    expected = []
    assert second.messages == expected


def test_module_has_independent_bias(changed_module):
    """GIVEN a module with changed messages and bias.
    WHEN another module is created.
    THEN module has independent bias."""
    second = GPTModule()
    expected = {}
    assert second.logit_bias == expected


@pytest.fixture
def vendor_generator(gen, vendor):
    gen.client.vendor = vendor
    return gen


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,vendor", [
        ("gpt-4o", Vendor.OPENAI),
        ("gemini-2.5-pro", Vendor.GOOGLE)
    ])
async def test_model_vendor_match(vendor_generator, model, vendor):
    """GIVEN a gen and specified request options.
    WHEN a response is requested.
    THEN the request uses the specified model."""
    gen = vendor_generator
    await gen.generate(system_msg="s", prompt="p", model=model)
    assert gen.request_body["model"] == model


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,vendor", [
        ("missing", Vendor.OPENAI),
        ("gpt-4o", Vendor.GOOGLE),
        ("gemini-2.5-pro", Vendor.OPENAI)
    ])
async def test_wrong_model_or_vendor_rejected(vendor_generator, model, vendor):
    """GIVEN a model that is not available for the vendor.
    WHEN a response is requested.
    THEN the model is rejected before a provider request."""
    gen = vendor_generator
    with pytest.raises(ValueError, match="Invalid model"):
        await gen.generate(system_msg="s", prompt="p", model=model)
    await_args_list = gen.client.async_chat_completions.await_args_list
    expected = []
    assert await_args_list == expected


@pytest.fixture
def gen_kwargs():
    return {
        "system_msg": "instructions",
        "prompt": "question",
        "model": "gpt-4o",
        "temperature": 0.4
    }


@pytest.fixture
def expected_request_body(gen_kwargs):
    return {
        "model": gen_kwargs["model"],
        "temperature": gen_kwargs["temperature"],
        "messages": [
            {"role": "system", "content": gen_kwargs["system_msg"]},
            {"role": "user", "content": gen_kwargs["prompt"]}
        ]
    }


@pytest.mark.asyncio
class TestGenerationAsync:

    async def test_generation_passes_request_chat(self, gen, gen_kwargs, expected_request_body):
        """GIVEN a gen and specified request options.
        WHEN a response is requested using chat.
        THEN generation passes request."""
        await gen.generate(**gen_kwargs, chat=True)
        expected = [mock_call(expected_request_body)]
        await_args_list = gen.client.async_chat_completions.await_args_list
        assert await_args_list == expected

    async def test_generation_passes_request_body(self, gen, gen_kwargs, expected_request_body):
        """GIVEN a gen and specified request options.
        WHEN a response is requested using body.
        THEN generation passes request."""
        await gen.generate(**gen_kwargs, chat=False)
        expected = [mock_call(expected_request_body)]
        await_args_list = gen.client.async_chat_completions_body.await_args_list
        assert await_args_list == expected

    async def test_generation_returns_reply_chat(self, gen, gen_kwargs):
        """GIVEN a gen and specified request options.
        WHEN a response is requested using chat.
        THEN generation returns reply."""
        reply = await gen.generate(**gen_kwargs, chat=True)
        expected = gen.client.async_chat_completions.return_value
        assert reply is expected

    async def test_generation_returns_reply_body(self, gen, gen_kwargs):
        """GIVEN a gen and specified request options.
        WHEN a response is requested using body.
        THEN generation returns reply."""
        reply = await gen.generate(**gen_kwargs, chat=False)
        expected = gen.client.async_chat_completions_body.return_value
        assert reply is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "system_msg,prompt,temp,error", [
        ("", "q", 0, "system_msg"),
        ("s", "", 0, "prompt"),
        ("s", "q", -1, "temperature")
    ])
class TestInvalidGeneration:
    async def test_invalid_generation_avoids_provider(self, gen, system_msg, prompt, temp, error):
        """GIVEN an empty message or a negative temperature.
        WHEN a response is requested.
        THEN invalid generation avoids provider."""
        with pytest.raises(ValueError, match=error):
            await gen.generate(system_msg=system_msg, prompt=prompt, temperature=temp)
        expected = []
        assert gen.client.async_chat_completions.await_args_list == expected

    async def test_invalid_generation_preserves_messages(self, gen, system_msg, prompt, temp, error):
        """GIVEN an empty message or a negative temperature.
        WHEN a response is requested.
        THEN invalid generation preserves messages."""
        with pytest.raises(ValueError, match=error):
            await gen.generate(system_msg=system_msg, prompt=prompt, temperature=temp)
        expected = []
        assert gen.messages == expected

@pytest.fixture
def base_kwargs():
    return {
        "system_msg": "s",
        "prompt": "p",
    }

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,effort,expected,temp", [
        (TextModels.gpt_5_2, "default", "none", 1),
        (TextModels.gpt_5, "default", "minimal", 1),
        (TextModels.gpt_5_2, "high", "high", 1),
        (TextModels.gpt_5, "low", "low", 1),
        (TextModels.gpt_4o, "default", None, 0.3)
    ])
class TestGenerationSetsRequestOptions:
    async def test_generation_sets_reasoning(self, gen, model, effort, expected, temp, base_kwargs):
        """GIVEN a gen and specified request options.
        WHEN a response is requested.
        THEN generation sets reasoning."""
        await gen.generate(**base_kwargs, model=model, reasoning_effort=effort, temperature=temp)
        assert gen.reasoning_effort == expected

    async def test_generation_sets_temperature(self, gen, model, effort, expected, temp, base_kwargs):
        """GIVEN a gen and specified request options.
        WHEN a response is requested.
        THEN generation sets temperature."""
        await gen.generate(**base_kwargs, model=model, reasoning_effort=effort, temperature=temp)
        assert gen.temperature == temp


@pytest_asyncio.fixture
async def gen_with_conversation(gen):
    await gen.generate(system_msg="s", prompt="first", model="gpt-4o")
    return gen


@pytest.fixture
def expected_messages():
    return [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"}
    ]

@pytest.mark.asyncio
async def test_continuation_retains_messages(
    gen_with_conversation,
    expected_messages
):
    """GIVEN an existing conversation and a new exchange.
    WHEN a response is requested.
    THEN continuation retains messages."""
    gen = gen_with_conversation
    await gen.continued_response(assistant_response="answer", prompt="second")
    assert gen.request_body["messages"] == expected_messages


@pytest.mark.asyncio
async def test_continuation_sends_once(gen_with_conversation):
    """GIVEN an existing conversation and a new exchange.
    WHEN a response is requested.
    THEN continuation sends once."""
    gen = gen_with_conversation
    await gen.continued_response(assistant_response="answer", prompt="second")
    expected = 2
    assert gen.client.async_chat_completions.await_count == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer,prompt", [
        ("", "p"),
        ("a", "")
    ])
async def test_continuation_validates_input(gen, answer, prompt):
    """GIVEN an empty assistant response or prompt.
    WHEN a response is requested.
    THEN the input is rejected before a provider request."""
    with pytest.raises(ValueError):
        await gen.continued_response(assistant_response=answer, prompt=prompt)
    expected = []
    assert gen.client.async_chat_completions.await_args_list == expected


@pytest.fixture
def json_schema():
    return {"name": "result", "schema": {"type": "object"}}


@pytest.fixture
def structured_output_kwargs(json_schema):
    return {
        "system_msg": "s",
        "prompt": "p",
        "schema": json.dumps(json_schema),
        "model": "gpt-4o"
    }


@pytest.mark.asyncio
async def test_json_generation_returns_content(gen, structured_output_kwargs):
    """GIVEN a gen and specified request options.
    WHEN a response is requested.
    THEN json generation returns content."""
    reply = await gen.structured_output(**structured_output_kwargs)
    expected = '{"ok": true}'
    assert reply.content == expected


@pytest.mark.asyncio
async def test_json_generation_passes_schema(gen, structured_output_kwargs):
    """GIVEN a gen and specified request options.
    WHEN a response is requested.
    THEN json generation passes schema."""
    await gen.structured_output(**structured_output_kwargs)
    expected = {
        "type": "json_schema",
        "json_schema": json.loads(structured_output_kwargs["schema"])
    }
    assert gen.request_body["response_format"] == expected


@pytest.mark.asyncio
async def test_json_generation_selects_endpoint(gen, structured_output_kwargs):
    """GIVEN a gen and specified request options.
    WHEN a response is requested.
    THEN json generation selects endpoint."""
    await gen.structured_output(**structured_output_kwargs)
    expected = [mock_call(gen.request_body, input_type=InputType.JSON)]
    assert gen.client.async_chat_completions_body.await_args_list == expected


@pytest.fixture
def pydantic_schema():
    class Result(BaseModel):
        ok: bool
    return Result


@pytest.fixture
def structured_output_kwargs_pydantic(pydantic_schema):
    return {
        "system_msg": "s",
        "prompt": "p",
        "schema": pydantic_schema,
        "input_type": "pydantic",
        "model": "gpt-4o",
        "assistant_response": "previous"
    }



@pytest.mark.asyncio
async def test_pydantic_generation_retains_schema(gen, structured_output_kwargs_pydantic):
    """GIVEN a gen and specified request options.
    WHEN a response is requested.
    THEN pydantic generation retains schema."""
    await gen.structured_output(**structured_output_kwargs_pydantic)
    expected = structured_output_kwargs_pydantic["schema"]
    assert gen.response_format is expected


@pytest.mark.asyncio
async def test_pydantic_generation_retains_context(gen, structured_output_kwargs_pydantic):
    """GIVEN a gen and specified request options.
    WHEN a response is requested.
    THEN pydantic generation retains context."""
    await gen.structured_output(**structured_output_kwargs_pydantic)
    expected = {"role": "assistant", "content": "previous"}
    assert gen.messages[0] == expected


@pytest.mark.asyncio
async def test_pydantic_generation_selects_endpoint(gen, structured_output_kwargs_pydantic):
    """GIVEN a gen and specified request options.
    WHEN a response is requested.
    THEN pydantic generation selects endpoint."""
    await gen.structured_output(**structured_output_kwargs_pydantic)
    expected = [mock_call(gen.request_body, input_type=InputType.PYDANTIC)]
    assert gen.client.async_chat_completions_body.await_args_list == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "schema,error", [
        ({}, ValueError),
        ("not json", json.JSONDecodeError)
    ])
async def test_invalid_json_schema(gen, schema, error):
    """GIVEN an invalid JSON schema.
    WHEN a response is requested.
    THEN the schema is rejected before a provider request."""
    with pytest.raises(error):
        await gen.structured_output(
            system_msg="s", prompt="p", schema=schema
        )
    expected = []
    assert gen.client.async_chat_completions_body.await_args_list == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "system,prompt,input_type", [
        ("", "p", "json"),
        ("s", "", "json"),
        ("s", "p", "invalid")
    ])
async def test_structured_output_invalid_arguments(gen, system, prompt, input_type):
    """GIVEN an empty message or invalid input type.
    WHEN a response is requested.
    THEN the input is rejected before a provider request."""
    with pytest.raises(ValueError):
        await gen.structured_output(
            system_msg=system, prompt=prompt, schema='{"name":"result","schema":{"type":"object"}}', input_type=input_type
        )
    expected = []
    assert gen.client.async_chat_completions_body.await_args_list == expected


@pytest.fixture
def failed_generator(gen):
    error = RuntimeError("provider failed")
    gen.client.async_chat_completions.return_value = ClientReply(error=error)
    gen.client.async_chat_completions_body.return_value = ClientReply(error=error)
    return error, gen


@pytest.mark.asyncio
async def test_generate_error_preserves_cause(failed_generator):
    """GIVEN a generate operation that reports an error.
    WHEN a response is requested.
    THEN the error retains the provider error as its cause."""
    error, gen = failed_generator
    with pytest.raises(ClientError) as exc:
        await gen.generate(system_msg="s", prompt="p")
    assert exc.value.__cause__ is error

@pytest.mark.asyncio
async def test_continued_response_error_preserves_cause(failed_generator):
    """GIVEN a continued_response operation that reports an error.
    WHEN a response is requested.
    THEN the error retains the provider error as its cause."""
    error, gen = failed_generator
    with pytest.raises(ClientError) as exc:
        await gen.continued_response(assistant_response="s", prompt="p")
    assert exc.value.__cause__ is error


@pytest.mark.asyncio
async def test_structured_output_error_preserves_cause(failed_generator):
    """GIVEN a structured_output operation that reports an error.
    WHEN a response is requested.
    THEN the error retains the provider error as its cause."""
    error, gen = failed_generator
    with pytest.raises(ClientError) as exc:
        await gen.structured_output(system_msg="s", prompt="p", schema='{"name":"result","schema":{"type":"object"}}')
    assert exc.value.__cause__ is error





def test_factory_applies_options(client):
    """GIVEN a gen and specified request options.
    WHEN a gen is created.
    THEN the new gen has the specified token limit."""
    expected_tokens = 42
    gen = make_generate(client=client, max_tokens=expected_tokens)
    assert gen.max_tokens == expected_tokens


def test_factory_wraps_errors(client):
    """GIVEN an operation that reports an error.
    WHEN a gen is created.
    THEN the error retains the original validation error."""
    with pytest.raises(InitializationError) as exc:
        make_generate(client=client, nonexistent_option=True)
    assert isinstance(exc.value.__cause__, ValueError)

@pytest.fixture
def reasoning_generate_kwargs():
    return {
        "system_msg": "s",
        "prompt": "p",
        "model": "gpt-5.2"
    }

@pytest_asyncio.fixture
async def reasoning_generator(gen, reasoning_generate_kwargs):
    await gen.generate(**reasoning_generate_kwargs)
    return gen


@pytest.fixture
def nonreasoning_kwargs(reasoning_generate_kwargs):
    return {**reasoning_generate_kwargs, "model": "gpt-4o"}


@pytest.mark.asyncio
async def test_switching_models_clears_reasoning_settings(reasoning_generator, nonreasoning_kwargs):
    """GIVEN a gen that used a reasoning model.
    WHEN a response is requested.
    THEN the new request has no reasoning setting."""
    gen = reasoning_generator
    not_expected_str = "reasoning_effort"
    await gen.generate(**nonreasoning_kwargs)
    assert not_expected_str not in gen.request_body


@pytest.fixture
def logged_generator(gen):
    gen.logger = Mock()
    return gen

@pytest.fixture
def log_gen_kwargs():
    return {
        "system_msg": "s",
        "prompt": "p",
        "schema": '{"name":"result","schema":{"type":"object"}}',
        "debug": True
    }

@pytest.mark.asyncio
@pytest.mark.parametrize("assistant", [None, "previous"])
async def test_debug_generation_logs_reply(logged_generator, assistant, log_gen_kwargs):
    """GIVEN a gen and specified request options.
    WHEN a response is requested.
    THEN debug generation logs reply."""
    gen = logged_generator
    reply = await gen.structured_output(
        **log_gen_kwargs, assistant_response=assistant
    )
    expected = mock_call(reply)
    assert expected in gen.logger.debug.call_args_list


@pytest.mark.asyncio
@pytest.mark.parametrize("assistant", [None, "previous"])
async def test_debug_generation_logs_context(logged_generator, assistant, log_gen_kwargs):
    """GIVEN a gen and specified request options.
    WHEN a response is requested.
    THEN debug generation logs context."""
    gen = logged_generator
    await gen.structured_output(**log_gen_kwargs, assistant_response=assistant)
    expected = 1 if assistant is None else 2
    assert gen.logger.debug.call_count == expected


@pytest.fixture
def schema_module(tmp_path, monkeypatch):
    import sys
    module_name = "pytest_generated_schema"
    (tmp_path / f"{module_name}.py").write_text("from pydantic import BaseModel\nclass Result(BaseModel):\n    value: str\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, module_name, None)
    monkeypatch.delitem(sys.modules, module_name)
    return module_name

@pytest.fixture
def imported_schema_kwargs(schema_module):
    return {
        "system_msg": "s",
        "prompt": "p",
        "schema": "Result",
        "input_type": "pydantic",
        "module_name": schema_module
    }

@pytest.mark.asyncio
async def test_imported_schema_retains_name(gen: Generate, imported_schema_kwargs):
    """GIVEN a gen and specified request options.
    WHEN a response is requested.
    THEN imported schema retains name."""
    await gen.structured_output(**imported_schema_kwargs)
    expected = imported_schema_kwargs["schema"]
    assert gen.response_format.__name__ == expected


@pytest.mark.asyncio
async def test_imported_schema_retains_field_type(gen, imported_schema_kwargs):
    """GIVEN a gen and specified request options.
    WHEN a response is requested.
    THEN imported schema retains field type."""
    await gen.structured_output(**imported_schema_kwargs)
    annotation = gen.response_format.model_fields["value"].annotation
    assert annotation == str


@pytest.mark.asyncio
async def test_imported_schema_enters_request(gen, imported_schema_kwargs):
    """GIVEN a gen and specified request options.
    WHEN a response is requested.
    THEN imported schema enters request."""
    await gen.structured_output(**imported_schema_kwargs)
    expected_resp_format = gen.response_format
    assert gen.request_body["response_format"] is expected_resp_format


@pytest.mark.asyncio
async def test_missing_schema_module_does_not_call_provider(gen, imported_schema_kwargs):
    """GIVEN a schema module that does not exist.
    WHEN a response is requested.
    THEN the missing module is reported before a provider request."""
    kwargs = imported_schema_kwargs.copy()
    kwargs["module_name"] = "nonexistent_aiutils_schema_module"
    with pytest.raises(ValueError):
        await gen.structured_output(**kwargs)
    expected = []
    assert gen.client.async_chat_completions_body.await_args_list == expected
