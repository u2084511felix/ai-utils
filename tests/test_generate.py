import json
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from aiutils.clients import ClientReply, InputType
from aiutils.exceptions import ClientError, InitializationError
from aiutils.generate import Generate, _check_model, make_generate
from aiutils.models import TextModels, Vendor
from aiutils.modules.gpt_module import GPTModule, make_req_body


@pytest.fixture
def generator(client, monkeypatch):
    monkeypatch.setattr(client, "async_chat_completions", AsyncMock(return_value=ClientReply(content="answer")))
    monkeypatch.setattr(client, "async_chat_completions_body", AsyncMock(return_value=ClientReply(content='{"ok": true}')))
    return make_generate(client=client)


def test_request_body_keeps_zero_and_false_but_omits_empty_values():
    module = GPTModule(model="gpt-4o", temperature=0, stream=False, max_tokens=20, request_body={"old": True})
    module.__dict__["_internal"] = "secret"
    assert make_req_body(module) == {"model": "gpt-4o", "temperature": 0, "stream": False, "max_tokens": 20}


def test_module_mutable_defaults_are_independent():
    first, second = GPTModule(), GPTModule()
    first.messages.append({"role": "user", "content": "hi"})
    first.logit_bias["1"] = 1
    assert second.messages == []
    assert second.logit_bias == {}


@pytest.mark.parametrize("model,vendor", [("gpt-4o", Vendor.OPENAI), ("gemini-2.5-pro", Vendor.GOOGLE)])
def test_model_vendor_match(model, vendor):
    assert _check_model(model, vendor) == model


@pytest.mark.parametrize("model,vendor", [("missing", Vendor.OPENAI), ("gpt-4o", Vendor.GOOGLE), ("gemini-2.5-pro", Vendor.OPENAI)])
def test_wrong_model_or_vendor_rejected(model, vendor):
    with pytest.raises(ValueError, match="Invalid model"):
        _check_model(model, vendor)


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [True, False])
async def test_generate_builds_request_and_selects_endpoint(generator, chat):
    reply = await generator.generate("instructions", "question", model="gpt-4o", temperature=0.4, chat=chat)
    expected = dict(model="gpt-4o", temperature=0.4, messages=[
        {"role": "system", "content": "instructions"}, {"role": "user", "content": "question"},
    ])
    method = generator.client.async_chat_completions if chat else generator.client.async_chat_completions_body
    method.assert_awaited_once_with(expected)
    assert reply is method.return_value


@pytest.mark.asyncio
@pytest.mark.parametrize("system,prompt,temp,error", [("", "q", 0, "system_message"), ("s", "", 0, "prompt"), ("s", "q", -1, "temperature")])
async def test_generate_validates_before_calling_client(generator, system, prompt, temp, error):
    with pytest.raises(ValueError, match=error):
        await generator.generate(system, prompt, temperature=temp)
    generator.client.async_chat_completions.assert_not_awaited()
    assert generator.messages == []


@pytest.mark.asyncio
@pytest.mark.parametrize("model,effort,expected,temp", [
    (TextModels.gpt_5_2, "default", "none", 1),
    (TextModels.gpt_5, "default", "minimal", 1),
    (TextModels.gpt_5_2, "high", "high", 1),
    (TextModels.gpt_5, "low", "low", 1),
    (TextModels.gpt_4o, "default", None, 0.3),
])
async def test_reasoning_settings(generator, model, effort, expected, temp):
    await generator.generate("s", "p", model=model, reasoning_effort=effort, temperature=0.3)
    assert generator.reasoning_effort == expected
    assert generator.temperature == temp


@pytest.mark.asyncio
async def test_continued_response_keeps_conversation(generator):
    await generator.generate("s", "first", model="gpt-4o")
    await generator.continued_response("answer", "second")
    assert generator.request_body["messages"] == [
        {"role": "system", "content": "s"}, {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"}, {"role": "user", "content": "second"},
    ]
    assert generator.client.async_chat_completions.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("answer,prompt", [("", "p"), ("a", "")])
async def test_continuation_validates_input(generator, answer, prompt):
    with pytest.raises(ValueError):
        await generator.continued_response(answer, prompt)
    generator.client.async_chat_completions.assert_not_awaited()


@pytest.mark.asyncio
async def test_json_structured_output(generator):
    schema = {"name": "result", "schema": {"type": "object"}}
    reply = await generator.structured_output("s", "p", schema=json.dumps(schema), model="gpt-4o")
    assert reply.content == '{"ok": true}'
    assert generator.request_body["response_format"] == {"type": "json_schema", "json_schema": schema}
    generator.client.async_chat_completions_body.assert_awaited_once_with(generator.request_body, input_type=InputType.JSON)


@pytest.mark.asyncio
async def test_pydantic_structured_output_and_assistant_context(generator):
    class Result(BaseModel):
        ok: bool
    await generator.structured_output("s", "p", schema=Result, input_type="pydantic", model="gpt-4o", assistant_response="previous")
    assert generator.response_format is Result
    assert generator.messages[0] == {"role": "assistant", "content": "previous"}
    generator.client.async_chat_completions_body.assert_awaited_once_with(generator.request_body, input_type=InputType.PYDANTIC)


@pytest.mark.asyncio
@pytest.mark.parametrize("schema,error", [({}, TypeError), ("not json", json.JSONDecodeError)])
async def test_invalid_json_schema(generator, schema, error):
    with pytest.raises(error):
        await generator.structured_output("s", "p", schema=schema)
    generator.client.async_chat_completions_body.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("system,prompt,input_type", [("", "p", "json"), ("s", "", "json"), ("s", "p", "invalid")])
async def test_structured_output_invalid_arguments(generator, system, prompt, input_type):
    with pytest.raises(ValueError):
        await generator.structured_output(system, prompt, schema="{}", input_type=input_type)
    generator.client.async_chat_completions_body.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "continued_response", "structured_output"])
async def test_provider_error_preserves_cause(generator, method):
    error = RuntimeError("provider failed")
    generator.client.async_chat_completions.return_value = ClientReply(error=error)
    generator.client.async_chat_completions_body.return_value = ClientReply(error=error)
    kwargs = {"schema": "{}"} if method == "structured_output" else {}
    with pytest.raises(ClientError) as exc:
        await getattr(generator, method)("s", "p", **kwargs)
    assert exc.value.__cause__ is error


def test_factory_applies_options_and_wraps_errors(client):
    assert make_generate(client=client, max_tokens=42).max_tokens == 42
    with pytest.raises(InitializationError) as exc:
        make_generate(client=client, nonexistent_option=True)
    assert isinstance(exc.value.__cause__, ValueError)


@pytest.mark.asyncio
async def test_switching_models_clears_reasoning_settings(generator):
    await generator.generate("s", "p", model="gpt-5.2")
    await generator.generate("s", "p", model="gpt-4o")
    assert "reasoning_effort" not in generator.request_body


@pytest.mark.asyncio
@pytest.mark.parametrize("assistant", [None, "previous"])
async def test_structured_debug_logs_response_and_context(generator, assistant):
    from unittest.mock import Mock
    generator.logger = Mock()
    reply = await generator.structured_output("s", "p", schema="{}", assistant_response=assistant, debug=True)
    generator.logger.debug.assert_any_call(reply)
    assert generator.logger.debug.call_count == (1 if assistant is None else 2)


@pytest.mark.asyncio
async def test_pydantic_schema_can_be_loaded_from_module(generator, tmp_path, monkeypatch):
    import sys
    module_name = "pytest_generated_schema"
    (tmp_path / f"{module_name}.py").write_text("from pydantic import BaseModel\nclass Result(BaseModel):\n    value: str\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    # Restore the import registry even if the test fails.
    with monkeypatch.context() as scoped:
        scoped.setitem(sys.modules, module_name, None)
        scoped.delitem(sys.modules, module_name)
        await generator.structured_output("s", "p", schema="Result", input_type="pydantic", module_name=module_name)
        assert generator.response_format.__name__ == "Result"
        assert generator.response_format(value="answer").value == "answer"
        assert generator.request_body["response_format"] is generator.response_format


@pytest.mark.asyncio
async def test_missing_schema_module_does_not_call_provider(generator):
    with pytest.raises(ModuleNotFoundError):
        await generator.structured_output("s", "p", schema="Result", input_type="pydantic", module_name="nonexistent_aiutils_schema_module")
    generator.client.async_chat_completions_body.assert_not_awaited()
