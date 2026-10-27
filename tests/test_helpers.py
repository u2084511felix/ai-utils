import json
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call as mock_call

import pytest

from aiutils.modules import structured_output as structured
from aiutils.modules.function_call import FunctionCall
from aiutils.modules.prompt_engine import PromptEngine


@pytest.fixture
def tool_request(client, arguments):
    module = FunctionCall(client=client)
    function = Mock(return_value=5 if arguments != "{}" else "default")
    call = SimpleNamespace(id="call-1", function=SimpleNamespace(name="add", arguments=arguments))
    return (call, function, module)


@pytest.mark.parametrize("arguments,expected", [('{"a": 2, "b": 3}', "5"), ("{}", "default")])
def test_tool_dispatch_returns_output(tool_request, arguments, expected):
    """GIVEN a tool call and available functions.
    WHEN the tool call is dispatched.
    THEN tool dispatch returns output."""
    call, function, module = tool_request
    result = module.call_function([call], {"add": function})
    expected = {"add": {"tool_call_id": "call-1", "role": "tool", "name": "add", "content": expected}}
    assert result == expected


@pytest.mark.parametrize("arguments,expected", [('{"a": 2, "b": 3}', "5"), ("{}", "default")])
def test_tool_dispatch_passes_arguments(tool_request, arguments, expected):
    """GIVEN a tool call and available functions.
    WHEN the tool call is dispatched.
    THEN tool dispatch passes arguments."""
    call, function, module = tool_request
    module.call_function([call], {"add": function})
    expected = [mock_call(**json.loads(arguments))]
    assert function.call_args_list == expected


@pytest.fixture
def invalid_tool(client, name, arguments):
    module = FunctionCall(client=client)
    function = Mock()
    call = SimpleNamespace(id="id", function=SimpleNamespace(name=name, arguments=arguments))
    return (call, function, module)


@pytest.mark.parametrize("name,arguments,error", [("missing", "{}", KeyError), ("add", "broken", json.JSONDecodeError)])
def test_invalid_tool_calls(invalid_tool, name, arguments, error):
    """GIVEN an unknown tool name or invalid JSON arguments.
    WHEN the tool call is dispatched.
    THEN the invalid call is rejected before the tool runs."""
    call, function, module = invalid_tool
    with pytest.raises(error):
        module.call_function([call], {"add": function})
    expected = []
    assert function.call_args_list == expected


@pytest.fixture
def failed_tool(client):
    module = FunctionCall(client=client)
    error = RuntimeError("tool failed")
    call = SimpleNamespace(id="id", function=SimpleNamespace(name="fail", arguments="{}"))
    functions = {"fail": Mock(side_effect=error)}
    return (call, error, module, functions)


def test_function_exception_is_not_hidden(failed_tool):
    """GIVEN an operation that reports an error.
    WHEN the tool call is dispatched.
    THEN the tool error is returned to the caller."""
    call, error, module, functions = failed_tool
    with pytest.raises(RuntimeError) as exc:
        module.call_function([call], functions)
    expected = True
    assert (exc.value is error) == expected


@pytest.fixture
def prompt_engine(tmp_path):
    template = tmp_path / "prompt.txt"
    template.write_text("Hello {name}, {question}", encoding="utf-8")
    engine = PromptEngine()
    engine.filepath = str(template)
    engine.kwargs = {"name": "Zoë", "question": "why?"}
    return engine


def test_prompt_formatting(prompt_engine):
    """GIVEN a prompt template and its values.
    WHEN the prompt operation is called.
    THEN the prompt contains the supplied values."""
    prompt_engine.set_prompt()
    expected = "Hello Zoë, why?"
    assert prompt_engine.prompt == expected


@pytest.fixture
def rendered_prompt(prompt_engine, tmp_path):
    prompt_engine.set_prompt()
    return (prompt_engine, tmp_path / "rendered.txt")


def test_prompt_save(rendered_prompt):
    """GIVEN a prompt template and its values.
    WHEN the prompt operation is called.
    THEN the output file contains the prompt text."""
    engine, output = rendered_prompt
    engine.save_prompt(output)
    result = output.read_text()
    expected = engine.prompt
    assert result == expected


@pytest.fixture
def missing_prompt(tmp_path):
    template = tmp_path / "prompt.txt"
    template.write_text("{missing}")
    engine = PromptEngine()
    engine.filepath = str(template)
    return engine


def test_prompt_missing_placeholder_raises(missing_prompt):
    """GIVEN a template with a missing value.
    WHEN the prompt operation is called.
    THEN the missing value is reported."""
    engine = missing_prompt
    with pytest.raises(KeyError) as exc:
        engine.set_prompt()
    expected = KeyError
    assert exc.type == expected


@pytest.fixture
def custom_template(tmp_path):
    template = tmp_path / "prompt.j2"
    engine = PromptEngine()
    engine.filepath = str(template)
    engine.kwargs = {"values": ["one", "two"]}
    template.write_text("{% for value in kwargs['values'] %}{{ value }} {% endfor %}")
    return (engine, template)


def test_jinja_prompt_from_configured_file(custom_template):
    """GIVEN a prompt template and its values.
    WHEN the prompt operation is called.
    THEN the prompt contains the values from the configured template."""
    engine, template = custom_template
    engine.set_template_prompt(filepath=str(template))
    expected = "one two "
    assert engine.prompt == expected


@pytest.fixture
def default_template():
    engine = PromptEngine()
    engine.kwargs = {"task": "Update the function"}
    return engine


def test_default_template_includes_task(default_template):
    """GIVEN a prompt template and its values.
    WHEN the prompt operation is called.
    THEN default template includes task."""
    engine = default_template
    engine.set_template_prompt()
    expected = "Update the function"
    assert expected in engine.prompt


def test_default_template_includes_requirements(default_template):
    """GIVEN a prompt template and its values.
    WHEN the prompt operation is called.
    THEN default template includes requirements."""
    engine = default_template
    engine.set_template_prompt()
    expected = "OUTPUT REQUIREMENTS:"
    assert expected in engine.prompt


@pytest.fixture
def model_source(tmp_path):
    path = tmp_path / "models.py"
    path.write_text("from pydantic import BaseModel\n\nclass Child(BaseModel):\n    name: str\n\nclass Parent(BaseModel):\n    children: list[Child]\n    indexed: dict[str, Child]\n    direct: Child\n    labels: list[str]\n\nclass Other:\n    children: list[Child]\n")
    return path


def test_recursive_patterns_identify_only_model_collections(model_source):
    """GIVEN a source file with model classes.
    WHEN the model source is processed.
    THEN the result identifies only collections of model classes."""
    result = structured.find_recursive_patterns(str(model_source))
    expected = {"Parent": ["children", "indexed"]}
    assert result == expected


def test_rebuild_adds_parent_call(model_source):
    """GIVEN a source file with model classes.
    WHEN the model source is processed.
    THEN rebuild adds parent call."""
    structured.add_model_rebuild_calls(str(model_source))
    content = model_source.read_text()
    expected = 1
    assert content.count("Parent.model_rebuild()") == expected


def test_rebuild_omits_child_call(model_source):
    """GIVEN a source file with model classes.
    WHEN the model source is processed.
    THEN rebuild omits child call."""
    structured.add_model_rebuild_calls(str(model_source))
    content = model_source.read_text()
    expected = False
    assert ("Child.model_rebuild()" in content) == expected


def test_rebuild_preserves_valid_source(model_source):
    """GIVEN a source file with model classes.
    WHEN the model source is processed.
    THEN rebuild preserves valid source."""
    structured.add_model_rebuild_calls(str(model_source))
    content = model_source.read_text()
    expected = "<module>"
    assert compile(content, str(model_source), "exec").co_name == expected


@pytest.fixture
def rebuilt_source(model_source):
    structured.add_model_rebuild_calls(str(model_source))
    return (model_source, model_source.read_text())


def test_rebuild_calls_are_added_once(rebuilt_source):
    """GIVEN a model file that already has rebuild calls.
    WHEN the model source is processed.
    THEN the source does not change."""
    model_source, content = rebuilt_source
    structured.add_model_rebuild_calls(str(model_source))
    result = model_source.read_text()
    expected = content
    assert result == expected


@pytest.fixture
def caller_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(structured, "get_caller_script_dir", lambda: str(tmp_path))
    return tmp_path


def test_save_json_in_caller_directory(caller_directory):
    """GIVEN JSON data and a caller directory.
    WHEN the data is saved.
    THEN the JSON file is saved in the caller directory."""
    tmp_path = caller_directory
    structured.save_json_file({"ok": True}, "result.json")
    result = json.loads((tmp_path / "tmp" / "result.json").read_text())
    expected = {"ok": True}
    assert result == expected


@pytest.fixture
def script_process(monkeypatch):
    run = Mock()
    monkeypatch.setattr(structured.subprocess, "run", run)
    return run


def test_script_runner_forwards_command_without_executing(script_process):
    """GIVEN a command and a process runner.
    WHEN the command is submitted.
    THEN the command uses the specified process options."""
    run = script_process
    structured.run_script_action("example-command")
    expected = [mock_call("example-command", shell=True, check=True, text=True, capture_output=True)]
    assert run.call_args_list == expected


@pytest.fixture
def codegen_process(monkeypatch, failed):
    run = Mock(side_effect=subprocess.CalledProcessError(1, "codegen") if failed else None)
    rebuild = Mock()
    monkeypatch.setattr(structured, "run_script_action", run)
    monkeypatch.setattr(structured, "add_model_rebuild_calls", rebuild)
    return (run, rebuild)


@pytest.mark.parametrize("failed", [False, True])
def test_model_codegen_reports_status(codegen_process, failed):
    """GIVEN an operation that reports an error.
    WHEN the schema operation is called.
    THEN model codegen reports status."""
    expected = not failed
    assert structured.pydantic_model_generator("in.json", "out.py", "templates") is expected


@pytest.mark.parametrize("failed", [False, True])
def test_model_codegen_passes_input(codegen_process, failed):
    """GIVEN an operation that reports an error.
    WHEN the schema operation is called.
    THEN model codegen passes input."""
    run, rebuild = codegen_process
    structured.pydantic_model_generator("in.json", "out.py", "templates")
    expected = "--input in.json"
    assert expected in run.call_args.args[0]


@pytest.mark.parametrize("failed", [False, True])
def test_model_codegen_rebuilds_on_success(codegen_process, failed):
    """GIVEN an operation that reports an error.
    WHEN the schema operation is called.
    THEN model codegen rebuilds on success."""
    run, rebuild = codegen_process
    structured.pydantic_model_generator("in.json", "out.py", "templates")
    expected = [] if failed else [mock_call("out.py")]
    assert rebuild.call_args_list == expected


@pytest.fixture
def schema_codegen(tmp_path, monkeypatch):
    generator = Mock(return_value=True)
    monkeypatch.setattr(structured, "pydantic_model_generator", generator)
    schema = {"name": "Result", "description": "A result", "schema": {"properties": {"value": {"type": "string"}}, "required": ["value"]}}
    target = tmp_path / "schema.json"
    return (generator, schema, target)


def test_schema_conversion_reports_success(schema_codegen):
    """GIVEN a schema and a model generation service.
    WHEN the schema operation is called.
    THEN schema conversion reports success."""
    generator, schema, target = schema_codegen
    expected = True
    assert structured.reformat_generated_schema(schema, str(target), "model.py") is expected


def test_schema_conversion_writes_json(schema_codegen):
    """GIVEN a schema and a model generation service.
    WHEN the schema operation is called.
    THEN schema conversion writes json."""
    generator, schema, target = schema_codegen
    structured.reformat_generated_schema(schema, str(target), "model.py")
    expected = {"$schema": "http://json-schema.org/draft-07/schema#", "title": "Result", "description": "A result", "type": "object", "properties": schema["schema"]["properties"], "required": ["value"], "additionalProperties": False}
    assert json.loads(target.read_text()) == expected


def test_schema_conversion_passes_input_path(schema_codegen):
    """GIVEN a schema and a model generation service.
    WHEN the schema operation is called.
    THEN the model request has the schema input path."""
    generator, schema, target = schema_codegen
    structured.reformat_generated_schema(schema, str(target), "model.py")
    expected = str(target)
    assert generator.call_args.args[0] == expected


def test_schema_conversion_passes_output_path(schema_codegen):
    """GIVEN a schema and a model generation service.
    WHEN the schema operation is called.
    THEN the model request has the specified output path."""
    generator, schema, target = schema_codegen
    structured.reformat_generated_schema(schema, str(target), "model.py")
    expected = "model.py"
    assert generator.call_args.args[1] == expected


@pytest.fixture
def parsed_response():
    parsed = object()
    parse = AsyncMock(return_value=SimpleNamespace(output_parsed=parsed))
    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    schema = object()
    return (client, parse, parsed, schema)


@pytest.mark.asyncio
async def test_structured_response_returns_model(parsed_response):
    """GIVEN a schema and a model generation service.
    WHEN the schema operation is called.
    THEN structured response returns model."""
    client, parse, parsed, schema = parsed_response
    expected = parsed
    assert await structured.responses_structured_output("model", "instructions", "prompt", schema, client=client) is expected


@pytest.mark.asyncio
async def test_structured_response_passes_schema(parsed_response):
    """GIVEN a schema and a model generation service.
    WHEN the schema operation is called.
    THEN structured response passes schema."""
    client, parse, parsed, schema = parsed_response
    await structured.responses_structured_output("model", "instructions", "prompt", schema, client=client)
    expected = [mock_call(model="model", instructions="instructions", input="prompt", reasoning={"effort": "low"}, text_format=schema)]
    assert parse.await_args_list == expected
