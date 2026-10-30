from aiutils.budget import TokenUsage
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call as mock_call

import pytest

from aiutils.clients import ClientReply
from aiutils.exceptions import ClientError
from aiutils.modules import structured_output as structured
from aiutils.modules.function_call import FunctionCall


@pytest.fixture
def output_client():
    return Mock(vendor="openai")


@pytest.fixture
def function_module(client, monkeypatch):
    module = FunctionCall(client=client)
    arguments = '{"value": "answer"}'
    tools = [SimpleNamespace(function=SimpleNamespace(arguments=arguments))]
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=tools))])
    method = AsyncMock(return_value=ClientReply(usage=TokenUsage(), content=tools, response=response))
    monkeypatch.setattr(FunctionCall, "function_call_legacy_structured_output", method)
    monkeypatch.setattr(structured, "FunctionCall", lambda: module)
    return (module, method, arguments)


@pytest.fixture
def legacy_output_configures_tool_and_returns_json_case(function_module):
    module, method, arguments = function_module
    schema = {"name": "result", "parameters": {"type": "object"}}
    return (arguments, method, module, schema)


@pytest.mark.asyncio
@pytest.mark.parametrize("system", ["default", "custom instructions"])
async def test_legacy_output_returns_json(legacy_output_configures_tool_and_returns_json_case, system):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN legacy output returns json."""
    arguments, method, module, schema = legacy_output_configures_tool_and_returns_json_case
    result = await structured.legacy_structured_output("question", json.dumps(schema), system)
    expected = arguments
    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("system", ["default", "custom instructions"])
async def test_legacy_output_configures_tools(legacy_output_configures_tool_and_returns_json_case, system):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN legacy output configures tools."""
    arguments, method, module, schema = legacy_output_configures_tool_and_returns_json_case
    await structured.legacy_structured_output("question", json.dumps(schema), system)
    expected = [{"type": "function", "function": schema}]
    assert module.tools == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("system", ["default", "custom instructions"])
async def test_legacy_output_selects_tool(legacy_output_configures_tool_and_returns_json_case, system):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN legacy output selects tool."""
    arguments, method, module, schema = legacy_output_configures_tool_and_returns_json_case
    await structured.legacy_structured_output("question", json.dumps(schema), system)
    expected = {"type": "function", "function": {"name": "result"}}
    assert module.tool_choice == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("system", ["default", "custom instructions"])
async def test_legacy_output_passes_instructions(legacy_output_configures_tool_and_returns_json_case, system):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN legacy output passes instructions."""
    arguments, method, module, schema = legacy_output_configures_tool_and_returns_json_case
    await structured.legacy_structured_output("question", json.dumps(schema), system)
    instructions = method.await_args.args[0]
    expected = "Supply the function variables for the given function according to the instruction." if system == "default" else system
    assert instructions == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("system", ["default", "custom instructions"])
async def test_legacy_output_passes_prompt(legacy_output_configures_tool_and_returns_json_case, system):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN legacy output passes prompt."""
    arguments, method, module, schema = legacy_output_configures_tool_and_returns_json_case
    await structured.legacy_structured_output("question", json.dumps(schema), system)
    expected = "question"
    assert method.await_args.args[1] == expected


@pytest.fixture
def failed_output(function_module):
    _, method, _ = function_module
    error = RuntimeError("provider failed")
    method.return_value = ClientReply(error=error)
    return error


@pytest.mark.asyncio
async def test_legacy_output_chains_provider_error(failed_output):
    """GIVEN an operation that reports an error.
    WHEN the schema operation is called.
    THEN the error retains the provider error as its cause."""
    error = failed_output
    with pytest.raises(ClientError) as exc:
        await structured.legacy_structured_output("question", '{"name":"result"}')
    expected = True
    assert (exc.value.__cause__ is error) == expected


@pytest.fixture
def missing_output(function_module):
    _, method, _ = function_module
    method.return_value = ClientReply(usage=TokenUsage())


@pytest.mark.asyncio
async def test_legacy_output_rejects_missing_response(missing_output):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN the absent response is reported."""
    with pytest.raises(ClientError, match="No response") as exc:
        await structured.legacy_structured_output("question", '{"name":"result"}')
    expected = ClientError
    assert exc.type == expected


@pytest.mark.asyncio
async def test_legacy_output_rejects_invalid_schema_before_provider_call(function_module):
    """GIVEN an invalid JSON schema.
    WHEN the schema operation is called.
    THEN the invalid schema is rejected before a provider request."""
    _, method, _ = function_module
    with pytest.raises(json.JSONDecodeError):
        await structured.legacy_structured_output("question", "invalid")
    expected = []
    assert method.await_args_list == expected


@pytest.fixture
def invalid_output(function_module):
    _, method, _ = function_module
    method.return_value.content[0].function.arguments = "invalid"


@pytest.mark.asyncio
async def test_legacy_output_rejects_invalid_provider_json(invalid_output):
    """GIVEN invalid JSON from the provider.
    WHEN the schema operation is called.
    THEN the invalid provider JSON is reported."""
    with pytest.raises(json.JSONDecodeError) as exc:
        await structured.legacy_structured_output("question", '{"name":"result"}')
    expected = json.JSONDecodeError
    assert exc.type == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("function,field", [(structured.generate_legacy_structured_output_schema, "parameters"), (structured.generate_structured_output_schema, "schema")])
@pytest.mark.parametrize("system", ["default", "custom instructions"])
async def test_schema_generation_returns_json(function_module, function, field, system):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN schema generation returns json."""
    module, method, arguments = function_module
    result = await function({"value": "string"}, system)
    expected = arguments
    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("function,field", [(structured.generate_legacy_structured_output_schema, "parameters"), (structured.generate_structured_output_schema, "schema")])
@pytest.mark.parametrize("system", ["default", "custom instructions"])
async def test_schema_generation_sets_tool_type(function_module, function, field, system):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN schema generation sets tool type."""
    module, method, arguments = function_module
    await function({"value": "string"}, system)
    tool = module.tools[0]
    expected = "function"
    assert tool["type"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("function,field", [(structured.generate_legacy_structured_output_schema, "parameters"), (structured.generate_structured_output_schema, "schema")])
@pytest.mark.parametrize("system", ["default", "custom instructions"])
async def test_schema_generation_names_tool(function_module, function, field, system):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN schema generation names tool."""
    module, method, arguments = function_module
    await function({"value": "string"}, system)
    tool = module.tools[0]
    expected = "Tool_Schema"
    assert tool["function"]["name"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("function,field", [(structured.generate_legacy_structured_output_schema, "parameters"), (structured.generate_structured_output_schema, "schema")])
@pytest.mark.parametrize("system", ["default", "custom instructions"])
async def test_schema_generation_defines_field(function_module, function, field, system):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN schema generation defines field."""
    module, method, arguments = function_module
    await function({"value": "string"}, system)
    tool = module.tools[0]
    expected = field
    assert expected in tool["function"]["parameters"]["properties"]


@pytest.mark.asyncio
@pytest.mark.parametrize("function,field", [(structured.generate_legacy_structured_output_schema, "parameters"), (structured.generate_structured_output_schema, "schema")])
@pytest.mark.parametrize("system", ["default", "custom instructions"])
async def test_schema_generation_requires_field(function_module, function, field, system):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN schema generation requires field."""
    module, method, arguments = function_module
    await function({"value": "string"}, system)
    tool = module.tools[0]
    expected = field
    assert expected in tool["function"]["parameters"]["required"]


@pytest.mark.asyncio
@pytest.mark.parametrize("function,field", [(structured.generate_legacy_structured_output_schema, "parameters"), (structured.generate_structured_output_schema, "schema")])
@pytest.mark.parametrize("system", ["default", "custom instructions"])
async def test_schema_generation_selects_tool(function_module, function, field, system):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN schema generation selects tool."""
    module, method, arguments = function_module
    await function({"value": "string"}, system)
    expected = "Tool_Schema"
    assert module.tool_choice["function"]["name"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("function,field", [(structured.generate_legacy_structured_output_schema, "parameters"), (structured.generate_structured_output_schema, "schema")])
@pytest.mark.parametrize("system", ["default", "custom instructions"])
async def test_schema_generation_sends_once(function_module, function, field, system):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN schema generation sends once."""
    module, method, arguments = function_module
    await function({"value": "string"}, system)
    expected = 1
    assert method.await_count == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("function,field", [(structured.generate_legacy_structured_output_schema, "parameters"), (structured.generate_structured_output_schema, "schema")])
@pytest.mark.parametrize("system", ["default", "custom instructions"])
async def test_schema_generation_passes_instructions(function_module, function, field, system):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN schema generation passes instructions."""
    module, method, arguments = function_module
    await function({"value": "string"}, system)
    instructions, prompt = method.await_args.args
    expected = "Generate a JSON schema for the given instruction converting from an existing data model outline, using the function." if system == "default" else system
    assert instructions == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("function,field", [(structured.generate_legacy_structured_output_schema, "parameters"), (structured.generate_structured_output_schema, "schema")])
@pytest.mark.parametrize("system", ["default", "custom instructions"])
async def test_schema_generation_passes_prompt(function_module, function, field, system):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN schema generation passes prompt."""
    module, method, arguments = function_module
    await function({"value": "string"}, system)
    instructions, prompt = method.await_args.args
    expected = "Transform this JSON object: {'value': 'string'}"
    assert prompt == expected


@pytest.fixture
def output_generator(monkeypatch):
    generate = SimpleNamespace(structured_output=AsyncMock(return_value=ClientReply(usage=TokenUsage(), content='{"value":42}')))
    factory = Mock(return_value=generate)
    monkeypatch.setattr(structured, "make_generate", factory)
    return (generate, factory)


@pytest.fixture
def explicit_options(output_generator, model, vendor):
    generate, factory = output_generator
    client = Mock(vendor=vendor)
    params = {"max_tokens": 100, "client": client}
    expected_call = dict(schema="Result", input_type="pydantic", module_name="models")
    if model:
        expected_call["model"] = model
    return (client, factory, generate, params, expected_call)


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor,model", [("openai", ""), ("openai", "gpt-5.2"), ("openai", "gpt-4o"), ("google", "gemini-2.5-pro")])
async def test_output_generation_decodes_json(explicit_options, vendor, model):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN output generation decodes json."""
    client, factory, generate, params, expected_call = explicit_options
    result = await structured.structured_outputs_generator("prompt", "Result", system_msg="custom", input_type="pydantic", module_name="models", params=params, model=model, vendor=vendor, client=client)
    expected = {"value": 42}
    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor,model", [("openai", ""), ("openai", "gpt-5.2"), ("openai", "gpt-4o"), ("google", "gemini-2.5-pro")])
async def test_output_generation_selects_vendor(explicit_options, vendor, model):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN output generation selects vendor."""
    client, factory, generate, params, expected_call = explicit_options
    await structured.structured_outputs_generator("prompt", "Result", system_msg="custom", input_type="pydantic", module_name="models", params=params, model=model, vendor=vendor, client=client)
    assert client.vendor == vendor


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor,model", [("openai", ""), ("openai", "gpt-5.2"), ("openai", "gpt-4o"), ("google", "gemini-2.5-pro")])
async def test_output_generation_passes_options(explicit_options, vendor, model):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN output generation passes options."""
    client, factory, generate, params, expected_call = explicit_options
    await structured.structured_outputs_generator("prompt", "Result", system_msg="custom", input_type="pydantic", module_name="models", params=params, model=model, vendor=vendor, client=client)
    expected = [mock_call(**params)]
    assert factory.call_args_list == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor,model", [("openai", ""), ("openai", "gpt-5.2"), ("openai", "gpt-4o"), ("google", "gemini-2.5-pro")])
async def test_output_generation_passes_schema(explicit_options, vendor, model):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN output generation passes schema."""
    client, factory, generate, params, expected_call = explicit_options
    await structured.structured_outputs_generator("prompt", "Result", system_msg="custom", input_type="pydantic", module_name="models", params=params, model=model, vendor=vendor, client=client)
    expected = [mock_call(system_msg="custom", prompt="prompt", **expected_call)]
    assert generate.structured_output.await_args_list == expected


@pytest.fixture
def reply_error(output_generator):
    generate, _ = output_generator
    error = RuntimeError("provider failed")
    generate.structured_output.return_value = ClientReply(error=error)
    return error


@pytest.mark.asyncio
async def test_output_generator_chains_reply_error(reply_error, output_client):
    """GIVEN an operation that reports an error.
    WHEN the schema operation is called.
    THEN the error retains the provider error as its cause."""
    error = reply_error
    with pytest.raises(ClientError) as exc:
        await structured.structured_outputs_generator("p", "{}", client=output_client)
    expected = True
    assert (exc.value.__cause__ is error) == expected


@pytest.fixture
def invalid_json(output_generator):
    generate, _ = output_generator
    generate.structured_output.return_value = ClientReply(usage=TokenUsage(), content="invalid")


@pytest.mark.asyncio
async def test_output_generator_rejects_invalid_provider_json(invalid_json, output_client):
    """GIVEN invalid JSON from the provider.
    WHEN the schema operation is called.
    THEN the invalid provider JSON is reported."""
    with pytest.raises(json.JSONDecodeError) as exc:
        await structured.structured_outputs_generator("p", "{}", client=output_client)
    expected = json.JSONDecodeError
    assert exc.type == expected


@pytest.fixture
def request_error(output_generator):
    generate, _ = output_generator
    error = RuntimeError("request failed")
    generate.structured_output.side_effect = error
    return error


@pytest.mark.asyncio
async def test_output_generator_raises_provider_exception(request_error, output_client):
    """GIVEN an operation that reports an error.
    WHEN the schema operation is called.
    THEN the request error is raised."""
    error = request_error
    with pytest.raises(RuntimeError) as raised:
        await structured.structured_outputs_generator("p", "{}", client=output_client)
    assert raised.value is error


@pytest.fixture
def default_options(output_generator, vendor):
    _, factory = output_generator
    client = Mock(vendor=vendor)
    return (client, factory)


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor,model,limit", [("openai", "gpt-5.2", 128000), ("openai", "gpt-4o", 32768), ("google", "gemini-2.5-pro", 65536)])
async def test_output_generation_sets_token_limit(default_options, vendor, model, limit):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN output generation sets token limit."""
    client, factory = default_options
    await structured.structured_outputs_generator("p", "{}", model=model, vendor=vendor, client=client)
    result = factory.call_args.kwargs.get("max_completion_tokens" if vendor == "openai" else "max_tokens")
    expected = limit
    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor,model,limit", [("openai", "gpt-5.2", 128000), ("openai", "gpt-4o", 32768), ("google", "gemini-2.5-pro", 65536)])
async def test_output_generation_retains_client(default_options, vendor, model, limit):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN output generation retains client."""
    client, factory = default_options
    await structured.structured_outputs_generator("p", "{}", model=model, vendor=vendor, client=client)
    expected = client
    assert factory.call_args.kwargs.get("client") is expected


@pytest.fixture
def schema_files(tmp_path, monkeypatch):
    schema = {"name": "Result", "schema": {"type": "object"}}
    source = tmp_path / "outline.json"
    source.write_text(json.dumps(schema))
    reformatted = tmp_path / "schema.json"
    model_path = tmp_path / "generated_models.py"
    reformat = Mock(return_value=True)
    generate = AsyncMock(return_value={"answer": 42})
    monkeypatch.setattr(structured, "reformat_generated_schema", reformat)
    monkeypatch.setattr(structured, "structured_outputs_generator", generate)
    return (generate, model_path, reformat, reformatted, schema, source)


@pytest.mark.asyncio
async def test_model_generation_returns_answer(schema_files):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN model generation returns answer."""
    generate, model_path, reformat, reformatted, schema, source = schema_files
    result = await structured.GeneratePydanticModelStructuredOutput("question", str(source), str(reformatted), str(model_path))
    expected = {"answer": 42}
    assert result == expected


@pytest.mark.asyncio
async def test_model_generation_converts_schema(schema_files):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN model generation converts schema."""
    generate, model_path, reformat, reformatted, schema, source = schema_files
    await structured.GeneratePydanticModelStructuredOutput("question", str(source), str(reformatted), str(model_path))
    expected = [mock_call(schema, str(reformatted), str(model_path))]
    assert reformat.call_args_list == expected


@pytest.mark.asyncio
async def test_model_generation_passes_schema(schema_files):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN model generation passes schema."""
    generate, model_path, reformat, reformatted, schema, source = schema_files
    await structured.GeneratePydanticModelStructuredOutput("question", str(source), str(reformatted), str(model_path))
    expected = [mock_call("question", "Result", input_type="pydantic", module_name="generated_models")]
    assert generate.await_args_list == expected


@pytest.fixture
def caller_frames(monkeypatch, tmp_path, frames):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(structured.inspect, "stack", lambda: [SimpleNamespace(filename=name) for name in frames])
    return tmp_path


@pytest.mark.parametrize("frames,expected", [(["/app/main.py", "/app/modules.py"], "/app"), (["/app/modules.py"], None), ([], None)])
def test_caller_directory_skips_modules_frames(caller_frames, frames, expected):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN the result is the caller directory or the current directory."""
    tmp_path = caller_frames
    result = structured.get_caller_script_dir()
    expected = expected or str(tmp_path)
    assert result == expected


@pytest.fixture
def collection_source(tmp_path):
    source = tmp_path / "model.py"
    source.write_text("from pydantic import BaseModel\nimport typing\nclass Child(BaseModel):\n    name: str\nclass Parent(BaseModel):\n    children: typing.list[Child]\n    indexed: typing.dict[str, Child]\n    optional: list[Child | None]\n    other: set[Child]\n    values: dict[str, int]\n")
    return source


def test_recursive_patterns_handles_nonmodel_collection_annotations(collection_source):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN the result identifies only supported model collections."""
    source = collection_source
    result = structured.find_recursive_patterns(str(source))
    expected = {"Parent": ["children", "indexed"]}
    assert result == expected


@pytest.fixture
def parse_failure():
    error = RuntimeError("parse failed")
    client = SimpleNamespace(responses=SimpleNamespace(parse=AsyncMock(side_effect=error)))
    schema = object()
    return (client, error, schema)


@pytest.mark.asyncio
async def test_responses_structured_output_propagates_provider_error(parse_failure):
    """GIVEN an operation that reports an error.
    WHEN the schema operation is called.
    THEN the provider error is returned to the caller."""
    client, error, schema = parse_failure
    with pytest.raises(RuntimeError) as exc:
        await structured.responses_structured_output("model", "s", "p", schema, client=client)
    expected = True
    assert (exc.value is error) == expected
