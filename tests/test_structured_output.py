"""Exercise schema orchestration with provider and codegen boundaries mocked."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aiutils.clients import ClientReply
from aiutils.exceptions import ClientError
from aiutils.modules import structured_output as structured
from aiutils.modules.function_call import FunctionCall


@pytest.fixture
def function_module(client, monkeypatch):
    module = FunctionCall(client=client)
    arguments = '{"value": "answer"}'
    response = SimpleNamespace(output=[SimpleNamespace(function=SimpleNamespace(arguments=arguments))])
    method = AsyncMock(return_value=ClientReply(response=response))
    monkeypatch.setattr(FunctionCall, "function_call_legacy_structured_output", method)
    monkeypatch.setattr(structured, "FunctionCall", lambda: module)
    return module, method, arguments


@pytest.mark.asyncio
@pytest.mark.parametrize("system", ["default", "custom instructions"])
async def test_legacy_output_configures_tool_and_returns_json(function_module, system):
    module, method, arguments = function_module
    schema = {"name": "result", "parameters": {"type": "object"}}
    result = await structured.legacy_structured_output("question", json.dumps(schema), system)
    assert result == arguments
    assert module.tools == [{"type": "function", "function": schema}]
    assert module.tool_choice == {"type": "function", "function": {"name": "result"}}
    instructions = method.await_args.args[0]
    assert instructions == ("Supply the function variables for the given function according to the instruction." if system == "default" else system)
    assert method.await_args.args[1] == "question"


@pytest.mark.asyncio
async def test_legacy_output_chains_provider_error(function_module):
    _, method, _ = function_module
    error = RuntimeError("provider failed")
    method.return_value = ClientReply(error=error)
    with pytest.raises(ClientError) as exc:
        await structured.legacy_structured_output("question", '{"name":"result"}')
    assert exc.value.__cause__ is error


@pytest.mark.asyncio
async def test_legacy_output_rejects_missing_response(function_module):
    _, method, _ = function_module
    method.return_value = ClientReply()
    with pytest.raises(ClientError, match="No response"):
        await structured.legacy_structured_output("question", '{"name":"result"}')


@pytest.mark.asyncio
async def test_legacy_output_rejects_invalid_schema_before_provider_call(function_module):
    _, method, _ = function_module
    with pytest.raises(json.JSONDecodeError):
        await structured.legacy_structured_output("question", "invalid")
    method.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_output_rejects_invalid_provider_json(function_module):
    _, method, _ = function_module
    method.return_value.response.output[0].function.arguments = "invalid"
    with pytest.raises(json.JSONDecodeError):
        await structured.legacy_structured_output("question", '{"name":"result"}')


@pytest.mark.asyncio
@pytest.mark.parametrize("function,field", [
    (structured.generate_legacy_structured_output_schema, "parameters"),
    (structured.generate_structured_output_schema, "schema"),
])
@pytest.mark.parametrize("system", ["default", "custom instructions"])
async def test_schema_generation_requests_schema_tool(function_module, function, field, system):
    module, method, arguments = function_module
    assert await function({"value": "string"}, system) == arguments
    tool = module.tools[0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "Tool_Schema"
    assert field in tool["function"]["parameters"]["properties"]
    assert field in tool["function"]["parameters"]["required"]
    assert module.tool_choice["function"]["name"] == "Tool_Schema"
    method.assert_awaited_once()
    instructions, prompt = method.await_args.args
    assert instructions == ("Generate a JSON schema for the given instruction converting from an existing data model outline, using the function." if system == "default" else system)
    assert prompt == "Transform this JSON object: {'value': 'string'}"


@pytest.fixture
def output_generator(monkeypatch):
    generate = SimpleNamespace(structured_output=AsyncMock(return_value=ClientReply(content='{"value":42}')))
    factory = Mock(return_value=generate)
    monkeypatch.setattr(structured, "make_generate", factory)
    return generate, factory


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor,model", [("openai", ""), ("openai", "gpt-5.2"), ("openai", "gpt-4o"), ("google", "gemini-2.5-pro")])
async def test_output_generator_decodes_reply_and_forwards_explicit_options(output_generator, vendor, model):
    generate, factory = output_generator
    client = Mock()
    params = {"max_tokens": 100, "client": client}
    result = await structured.structured_outputs_generator(
        "prompt", "Result", system_msg="custom", input_type="pydantic", module_name="models",
        params=params, model=model, vendor=vendor, client=client,
    )
    assert result == {"value": 42}
    client.set_vendor.assert_called_once_with(vendor)
    factory.assert_called_once_with(**params)
    expected = dict(schema="Result", input_type="pydantic", module_name="models")
    if model:
        expected["model"] = model
    generate.structured_output.assert_awaited_once_with("custom", "prompt", **expected)


@pytest.mark.asyncio
async def test_output_generator_chains_reply_error(output_generator):
    generate, _ = output_generator
    error = RuntimeError("provider failed")
    generate.structured_output.return_value = ClientReply(error=error)
    with pytest.raises(ClientError) as exc:
        await structured.structured_outputs_generator("p", "{}", client=Mock())
    assert exc.value.__cause__ is error


@pytest.mark.asyncio
async def test_output_generator_rejects_invalid_provider_json(output_generator):
    generate, _ = output_generator
    generate.structured_output.return_value = ClientReply(content="invalid")
    with pytest.raises(json.JSONDecodeError):
        await structured.structured_outputs_generator("p", "{}", client=Mock())


@pytest.mark.asyncio
async def test_output_generator_returns_raised_exception(output_generator):
    generate, _ = output_generator
    error = RuntimeError("request failed")
    generate.structured_output.side_effect = error
    assert await structured.structured_outputs_generator("p", "{}", client=Mock()) is error


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor,model,limit", [("openai", "gpt-5.2", 128000), ("openai", "gpt-4o", 32768), ("google", "gemini-2.5-pro", 65536)])
async def test_output_generator_applies_vendor_and_token_defaults(output_generator, vendor, model, limit):
    _, factory = output_generator
    client = Mock()
    await structured.structured_outputs_generator("p", "{}", model=model, vendor=vendor, client=client)
    assert factory.call_args.kwargs.get("max_tokens") == limit
    assert factory.call_args.kwargs.get("client") is client


@pytest.mark.asyncio
async def test_generate_pydantic_model_reads_schema_and_orchestrates(tmp_path, monkeypatch):
    schema = {"name": "Result", "schema": {"type": "object"}}
    source = tmp_path / "outline.json"
    source.write_text(json.dumps(schema))
    reformatted = tmp_path / "schema.json"
    model_path = tmp_path / "generated_models.py"
    reformat = Mock(return_value=True)
    generate = AsyncMock(return_value={"answer": 42})
    monkeypatch.setattr(structured, "reformat_generated_schema", reformat)
    monkeypatch.setattr(structured, "structured_outputs_generator", generate)
    assert await structured.GeneratePydanticModelStructuredOutput("question", str(source), str(reformatted), str(model_path)) == {"answer": 42}
    reformat.assert_called_once_with(schema, str(reformatted), str(model_path))
    generate.assert_awaited_once_with("question", "Result", input_type="pydantic", module_name="generated_models")


@pytest.mark.parametrize("frames,expected", [
    (["/app/main.py", "/app/modules.py"], "/app"),
    (["/app/modules.py"], None), ([], None),
])
def test_caller_directory_skips_modules_frames(monkeypatch, tmp_path, frames, expected):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(structured.inspect, "stack", lambda: [SimpleNamespace(filename=name) for name in frames])
    assert structured.get_caller_script_dir() == (expected or str(tmp_path))


def test_recursive_patterns_handles_nonmodel_collection_annotations(tmp_path):
    source = tmp_path / "model.py"
    source.write_text('''from pydantic import BaseModel
import typing
class Child(BaseModel):
    name: str
class Parent(BaseModel):
    children: typing.list[Child]
    indexed: typing.dict[str, Child]
    optional: list[Child | None]
    other: set[Child]
    values: dict[str, int]
''')
    assert structured.find_recursive_patterns(str(source)) == {"Parent": ["children", "indexed"]}


@pytest.mark.asyncio
async def test_responses_structured_output_propagates_provider_error():
    error = RuntimeError("parse failed")
    client = SimpleNamespace(responses=SimpleNamespace(parse=AsyncMock(side_effect=error)))
    with pytest.raises(RuntimeError) as exc:
        await structured.responses_structured_output("model", "s", "p", object(), client=client)
    assert exc.value is error
