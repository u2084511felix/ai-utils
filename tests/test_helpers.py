import json
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aiutils.modules import structured_output as structured
from aiutils.modules.function_call import _send_func_call_args_to_available_funcs
from aiutils.modules.prompt_engine import PromptEngine


@pytest.mark.parametrize("arguments,expected", [('{"a": 2, "b": 3}', "5"), ("{}", "default")])
def test_tool_function_dispatch(arguments, expected):
    function = Mock(return_value=5 if arguments != "{}" else "default")
    call = SimpleNamespace(id="call-1", function=SimpleNamespace(name="add", arguments=arguments))
    assert _send_func_call_args_to_available_funcs([call], {"add": function}) == {
        "add": {"tool_call_id": "call-1", "role": "tool", "name": "add", "content": expected},
    }
    function.assert_called_once_with(**json.loads(arguments))


@pytest.mark.parametrize("name,arguments,error", [("missing", "{}", KeyError), ("add", "broken", json.JSONDecodeError)])
def test_invalid_tool_calls(name, arguments, error):
    function = Mock()
    call = SimpleNamespace(id="id", function=SimpleNamespace(name=name, arguments=arguments))
    with pytest.raises(error):
        _send_func_call_args_to_available_funcs([call], {"add": function})
    function.assert_not_called()


def test_function_exception_is_not_hidden():
    error = RuntimeError("tool failed")
    call = SimpleNamespace(id="id", function=SimpleNamespace(name="fail", arguments="{}"))
    with pytest.raises(RuntimeError) as exc:
        _send_func_call_args_to_available_funcs([call], {"fail": Mock(side_effect=error)})
    assert exc.value is error


def test_prompt_formatting_and_save(tmp_path):
    template = tmp_path / "prompt.txt"
    template.write_text("Hello {name}, {question}", encoding="utf-8")
    engine = PromptEngine()
    engine.filepath = str(template)
    engine.kwargs = {"name": "Zoë", "question": "why?"}
    engine.set_prompt()
    assert engine.prompt == "Hello Zoë, why?"
    output = tmp_path / "rendered.txt"
    engine.save_prompt(output)
    assert output.read_text() == engine.prompt


def test_prompt_missing_placeholder_raises(tmp_path):
    template = tmp_path / "prompt.txt"
    template.write_text("{missing}")
    engine = PromptEngine()
    engine.filepath = str(template)
    with pytest.raises(KeyError):
        engine.set_prompt()


def test_jinja_prompt_from_configured_file(tmp_path):
    template = tmp_path / "prompt.j2"
    engine = PromptEngine()
    engine.filepath = str(template)
    engine.kwargs = {"values": ["one", "two"]}
    # Bracket access avoids colliding with the dict.values method.
    template.write_text("{% for value in kwargs['values'] %}{{ value }} {% endfor %}")
    engine.set_template_prompt(filepath=str(template))
    assert engine.prompt == "one two "


@pytest.mark.xfail(strict=True, raises=FileNotFoundError, reason="Default template path points inside modules instead of the package template directory")
def test_default_jinja_template():
    engine = PromptEngine()
    engine.kwargs = {"task": "Update the function"}
    engine.set_template_prompt()
    assert "Update the function" in engine.prompt
    assert "OUTPUT REQUIREMENTS:" in engine.prompt


@pytest.fixture
def model_source(tmp_path):
    path = tmp_path / "models.py"
    path.write_text('''from pydantic import BaseModel

class Child(BaseModel):
    name: str

class Parent(BaseModel):
    children: list[Child]
    indexed: dict[str, Child]
    direct: Child
    labels: list[str]

class Other:
    children: list[Child]
''')
    return path


def test_recursive_patterns_identify_only_model_collections(model_source):
    assert structured.find_recursive_patterns(str(model_source)) == {"Parent": ["children", "indexed"]}


def test_rebuild_calls_are_added_once_and_source_remains_valid(model_source):
    structured.add_model_rebuild_calls(str(model_source))
    content = model_source.read_text()
    assert content.count("Parent.model_rebuild()") == 1
    assert "Child.model_rebuild()" not in content
    compile(content, str(model_source), "exec")
    structured.add_model_rebuild_calls(str(model_source))
    assert model_source.read_text() == content


def test_save_json_in_caller_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(structured, "get_caller_script_dir", lambda: str(tmp_path))
    structured.save_json_file({"ok": True}, "result.json")
    assert json.loads((tmp_path / "tmp" / "result.json").read_text()) == {"ok": True}


def test_script_runner_forwards_command_without_executing(monkeypatch):
    run = Mock()
    monkeypatch.setattr(structured.subprocess, "run", run)
    structured.run_script_action("example-command")
    run.assert_called_once_with("example-command", shell=True, check=True, text=True, capture_output=True)


@pytest.mark.parametrize("failed", [False, True])
def test_pydantic_generator_success_and_failure(monkeypatch, failed):
    run = Mock(side_effect=subprocess.CalledProcessError(1, "codegen") if failed else None)
    rebuild = Mock()
    monkeypatch.setattr(structured, "run_script_action", run)
    monkeypatch.setattr(structured, "add_model_rebuild_calls", rebuild)
    assert structured.pydantic_model_generator("in.json", "out.py", "templates") is (not failed)
    assert "--input in.json" in run.call_args.args[0]
    if failed:
        rebuild.assert_not_called()
    else:
        rebuild.assert_called_once_with("out.py")


def test_reformat_schema_writes_json_and_invokes_codegen(tmp_path, monkeypatch):
    generator = Mock(return_value=True)
    monkeypatch.setattr(structured, "pydantic_model_generator", generator)
    schema = {"name": "Result", "description": "A result", "schema": {
        "properties": {"value": {"type": "string"}}, "required": ["value"],
    }}
    target = tmp_path / "schema.json"
    assert structured.reformat_generated_schema(schema, str(target), "model.py") is True
    assert json.loads(target.read_text()) == {
        "$schema": "http://json-schema.org/draft-07/schema#", "title": "Result",
        "description": "A result", "type": "object", "properties": schema["schema"]["properties"],
        "required": ["value"], "additionalProperties": False,
    }
    assert generator.call_args.args[:2] == (str(target), "model.py")


@pytest.mark.asyncio
async def test_responses_structured_output_returns_parsed_model():
    parsed = object()
    parse = AsyncMock(return_value=SimpleNamespace(output_parsed=parsed))
    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    schema = object()
    assert await structured.responses_structured_output("model", "instructions", "prompt", schema, client=client) is parsed
    parse.assert_awaited_once_with(model="model", instructions="instructions", input="prompt", reasoning={"effort": "low"}, text_format=schema)
