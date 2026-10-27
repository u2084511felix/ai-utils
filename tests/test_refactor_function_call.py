from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call as mock_call

import pytest

from aiutils.clients import ClientReply
from aiutils.exceptions import ClientError
from aiutils.models import TextModels
from aiutils.modules import function_call, prompt_engine
from aiutils.modules.function_call import FunctionCall
from aiutils.modules.prompt_engine import Refactor


@pytest.fixture
def legacy_request(client, monkeypatch, reasoning):
    if reasoning:
        monkeypatch.setattr(function_call, "REASONING_MODELS", [TextModels.previous])
    send = AsyncMock(return_value=ClientReply(content="answer"))
    monkeypatch.setattr(client, "async_chat_completions", send)
    module = FunctionCall(client=client)
    module.tools = [{"type": "function", "function": {"name": "test"}}]
    return (module, send)


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning", [False, True])
async def test_legacy_function_call_builds_request(legacy_request, reasoning):
    """GIVEN a module and specified input.
    WHEN the module operation is called.
    THEN the request contains the messages, model, tools, and reasoning setting."""
    module, send = legacy_request
    result = await module.function_call_legacy_structured_output("instructions", "question")
    result = result is send.return_value
    actual = (result, module.request_body["messages"], module.request_body["model"], module.request_body["tools"], module.request_body["reasoning_effort"] == "none" if reasoning else True, send.await_args_list)
    expected = (True, [{"role": "system", "content": "instructions"}, {"role": "user", "content": "question"}], TextModels.previous, module.tools, True, [mock_call(module.request_body)])
    assert actual == expected


@pytest.fixture
def legacy_failure(client, monkeypatch):
    error = RuntimeError("failed")
    monkeypatch.setattr(client, "async_chat_completions", AsyncMock(return_value=ClientReply(error=error)))
    module = FunctionCall(client=client)
    return (error, client, module)


@pytest.mark.asyncio
async def test_legacy_function_call_chains_error(legacy_failure):
    """GIVEN an operation that reports an error.
    WHEN the module operation is called.
    THEN the error retains the provider error as its cause."""
    error, client, module = legacy_failure
    with pytest.raises(ClientError) as exc:
        await module.function_call_legacy_structured_output("s", "p")
    expected = True
    assert (exc.value.__cause__ is error) == expected


@pytest.fixture
def tool_calls(client, empty):
    module = FunctionCall(client=client)
    call = SimpleNamespace(id="call-id", function=SimpleNamespace(name="double", arguments='{"value":3}'))
    calls = [] if empty else [call]
    expected = {} if empty else {"double": {"tool_call_id": "call-id", "role": "tool", "name": "double", "content": "6"}}
    functions = {"double": lambda value: value * 2}
    return (calls, expected, module, functions)


@pytest.mark.parametrize("empty", [False, True])
def test_call_function_public_wrapper(tool_calls, empty):
    """GIVEN a module and specified input.
    WHEN the module operation is called.
    THEN the result contains the tool output or an empty mapping."""
    calls, expected, module, functions = tool_calls
    result = module.call_function(calls, functions)
    assert result == expected


@pytest.fixture
def script_file(tmp_path):
    path = tmp_path / "script.py"
    path.write_text("print('hello')\n")
    first = Refactor()
    return (first, path)


def test_refactor_reads_script(script_file):
    """GIVEN a module and specified input.
    WHEN the module operation is called.
    THEN the result contains the script text."""
    first, path = script_file
    result = first.get_script(path)
    expected = "print('hello')\n"
    assert result == expected


@pytest.fixture
def changed_refactor():
    first = Refactor()
    first.target_file_paths.append("script.py")
    return first


def test_refactor_has_independent_state(changed_refactor):
    """GIVEN a module and specified input.
    WHEN the module operation is called.
    THEN the new module has independent target and prompt state."""
    first = changed_refactor
    second = Refactor()
    expected = ([], False)
    assert (second.target_file_paths, first.prompt_engine is second.prompt_engine) == expected


@pytest.fixture
def refactor_module(monkeypatch):
    diffs = SimpleNamespace(apply_diff=AsyncMock(), undo_last_diffs=Mock())
    monkeypatch.setattr(prompt_engine, "Diffs", lambda: diffs)
    refactor = Refactor()
    refactor.prompt_engine.prompt = "change the function"
    refactor.target_file_paths = ["target.py"]
    refactor.verbose = True
    return (refactor, diffs)


@pytest.mark.asyncio
async def test_refactor_delegates_prompt_targets(refactor_module):
    """GIVEN a module and specified input.
    WHEN the module operation is called.
    THEN the patch request uses the specified prompt and targets."""
    refactor, diffs = refactor_module
    await refactor.refactor_script()
    expected = [mock_call("change the function", ["target.py"], verbose=True)]
    assert diffs.apply_diff.await_args_list == expected


def test_refactor_delegates_undo(refactor_module):
    """GIVEN a module and specified input.
    WHEN the module operation is called.
    THEN the saved patches are undone."""
    refactor, diffs = refactor_module
    refactor.undo_diffs()
    expected = [mock_call()]
    assert diffs.undo_last_diffs.call_args_list == expected
