from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aiutils.clients import ClientReply
from aiutils.exceptions import ClientError
from aiutils.models import TextModels
from aiutils.modules import function_call, prompt_engine
from aiutils.modules.function_call import FunctionCall
from aiutils.modules.prompt_engine import Refactor


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning", [False, True])
async def test_legacy_function_call_builds_request(client, monkeypatch, reasoning):
    if reasoning:
        monkeypatch.setattr(function_call, "REASONING_MODELS", [TextModels.previous])
    send = AsyncMock(return_value=ClientReply(content="answer"))
    monkeypatch.setattr(client, "async_chat_completions", send)
    module = FunctionCall(client=client)
    module.tools = [{"type": "function", "function": {"name": "test"}}]
    result = await module.function_call_legacy_structured_output("instructions", "question")
    assert result is send.return_value
    assert module.request_body["messages"] == [
        {"role": "system", "content": "instructions"}, {"role": "user", "content": "question"},
    ]
    assert module.request_body["model"] == TextModels.previous
    assert module.request_body["tools"] == module.tools
    if reasoning:
        assert module.request_body["reasoning_effort"] == "none"
    send.assert_awaited_once_with(module.request_body)


@pytest.mark.asyncio
async def test_legacy_function_call_chains_error(client, monkeypatch):
    error = RuntimeError("failed")
    monkeypatch.setattr(client, "async_chat_completions", AsyncMock(return_value=ClientReply(error=error)))
    with pytest.raises(ClientError) as exc:
        await FunctionCall(client=client).function_call_legacy_structured_output("s", "p")
    assert exc.value.__cause__ is error


def test_call_function_public_wrapper(client):
    module = FunctionCall(client=client)
    call = SimpleNamespace(id="call-id", function=SimpleNamespace(name="double", arguments='{"value":3}'))
    assert module.call_function([call], {"double": lambda value: value * 2}) == {
        "double": {"tool_call_id": "call-id", "role": "tool", "name": "double", "content": "6"},
    }
    assert module.call_function([], {}) == {}


def test_refactor_reads_script_and_has_independent_state(tmp_path):
    path = tmp_path / "script.py"
    path.write_text("print('hello')\n")
    first, second = Refactor(), Refactor()
    assert first.get_script(path) == "print('hello')\n"
    first.target_file_paths.append(str(path))
    assert second.target_file_paths == []
    assert first.prompt_engine is not second.prompt_engine


@pytest.mark.asyncio
async def test_refactor_delegates_prompt_targets_and_undo(monkeypatch):
    diffs = SimpleNamespace(apply_diff=AsyncMock(), undo_last_diffs=Mock())
    monkeypatch.setattr(prompt_engine, "Diffs", lambda: diffs)
    refactor = Refactor()
    refactor.prompt_engine.prompt = "change the function"
    refactor.target_file_paths = ["target.py"]
    refactor.verbose = True
    await refactor.refactor_script()
    diffs.apply_diff.assert_awaited_once_with("change the function", ["target.py"], verbose=True)
    refactor.undo_diffs()
    diffs.undo_last_diffs.assert_called_once_with()
