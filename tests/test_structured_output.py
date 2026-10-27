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
    return Mock()


@pytest.fixture
def function_module(client, monkeypatch):
    module = FunctionCall(client=client)
    arguments = '{"value": "answer"}'
    response = SimpleNamespace(output=[SimpleNamespace(function=SimpleNamespace(arguments=arguments))])
    method = AsyncMock(return_value=ClientReply(response=response))
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
async def test_legacy_output_configures_tool_and_returns_json(legacy_output_configures_tool_and_returns_json_case, system):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN the request uses the schema tool and returns valid JSON."""
    arguments, method, module, schema = legacy_output_configures_tool_and_returns_json_case
    result = await structured.legacy_structured_output("question", json.dumps(schema), system)
    instructions = method.await_args.args[0]
    actual = (result, module.tools, module.tool_choice, instructions, method.await_args.args[1])
    expected = (arguments, [{"type": "function", "function": schema}], {"type": "function", "function": {"name": "result"}}, "Supply the function variables for the given function according to the instruction." if system == "default" else system, "question")
    assert actual == expected


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
    method.return_value = ClientReply()


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
    method.return_value.response.output[0].function.arguments = "invalid"


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
async def test_schema_generation_requests_schema_tool(function_module, function, field, system):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN the request uses the schema tool and specified instructions."""
    module, method, arguments = function_module
    result = await function({"value": "string"}, system)
    tool = module.tools[0]
    instructions, prompt = method.await_args.args
    actual = (result, tool["type"], tool["function"]["name"], field in tool["function"]["parameters"]["properties"], field in tool["function"]["parameters"]["required"], module.tool_choice["function"]["name"], method.await_count, instructions, prompt)
    expected = (arguments, "function", "Tool_Schema", True, True, "Tool_Schema", 1, "Generate a JSON schema for the given instruction converting from an existing data model outline, using the function." if system == "default" else system, "Transform this JSON object: {'value': 'string'}")
    assert actual == expected


@pytest.fixture
def output_generator(monkeypatch):
    generate = SimpleNamespace(structured_output=AsyncMock(return_value=ClientReply(content='{"value":42}')))
    factory = Mock(return_value=generate)
    monkeypatch.setattr(structured, "make_generate", factory)
    return (generate, factory)


@pytest.fixture
def explicit_options(output_generator, model):
    generate, factory = output_generator
    client = Mock()
    params = {"max_tokens": 100, "client": client}
    expected_call = dict(schema="Result", input_type="pydantic", module_name="models")
    if model:
        expected_call["model"] = model
    return (client, factory, generate, params, expected_call)


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor,model", [("openai", ""), ("openai", "gpt-5.2"), ("openai", "gpt-4o"), ("google", "gemini-2.5-pro")])
async def test_output_generator_decodes_reply_and_forwards_explicit_options(explicit_options, vendor, model):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN the request uses the specified options and returns the decoded JSON."""
    client, factory, generate, params, expected_call = explicit_options
    result = await structured.structured_outputs_generator("prompt", "Result", system_msg="custom", input_type="pydantic", module_name="models", params=params, model=model, vendor=vendor, client=client)
    actual = (result, client.set_vendor.call_args_list, factory.call_args_list, generate.structured_output.await_args_list)
    expected = ({"value": 42}, [mock_call(vendor)], [mock_call(**params)], [mock_call("custom", "prompt", **expected_call)])
    assert actual == expected


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
    generate.structured_output.return_value = ClientReply(content="invalid")


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
async def test_output_generator_returns_raised_exception(request_error, output_client):
    """GIVEN an operation that reports an error.
    WHEN the schema operation is called.
    THEN the result retains the request error."""
    error = request_error
    result = await structured.structured_outputs_generator("p", "{}", client=output_client) is error
    expected = True
    assert result == expected


@pytest.fixture
def default_options(output_generator):
    _, factory = output_generator
    client = Mock()
    return (client, factory)


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor,model,limit", [("openai", "gpt-5.2", 128000), ("openai", "gpt-4o", 32768), ("google", "gemini-2.5-pro", 65536)])
async def test_output_generator_applies_vendor_and_token_defaults(default_options, vendor, model, limit):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN the generator uses the supplied client and model token limit."""
    client, factory = default_options
    await structured.structured_outputs_generator("p", "{}", model=model, vendor=vendor, client=client)
    result = factory.call_args.kwargs.get("max_tokens")
    actual = (result, factory.call_args.kwargs.get("client") is client)
    expected = (limit, True)
    assert actual == expected


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
async def test_generate_pydantic_model_reads_schema_and_orchestrates(schema_files):
    """GIVEN a schema request and provider response.
    WHEN the schema operation is called.
    THEN the schema is converted and the result contains the generated answer."""
    generate, model_path, reformat, reformatted, schema, source = schema_files
    result = await structured.GeneratePydanticModelStructuredOutput("question", str(source), str(reformatted), str(model_path))
    actual = (result, reformat.call_args_list, generate.await_args_list)
    expected = ({"answer": 42}, [mock_call(schema, str(reformatted), str(model_path))], [mock_call("question", "Result", input_type="pydantic", module_name="generated_models")])
    assert actual == expected


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
