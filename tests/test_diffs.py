from types import SimpleNamespace
from unittest.mock import AsyncMock, call as mock_call

import pytest
import pytest_asyncio

from aiutils.clients import ClientReply
from aiutils.exceptions import ClientError, V4APatchError
from aiutils.modules import diffs
from aiutils.modules.diffs import ApplyV4AMode, apply_v4a_diff_text, invert_v4a_diff


@pytest.fixture
def newline_patch(newline, trailing):
    original = newline.join(["first", "old", "last"]) + (newline if trailing else "")
    expected = newline.join(["first", "new", "last"]) + (newline if trailing else "")
    return (original, expected)


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("trailing", [True, False])
def test_update_preserves_newlines(newline_patch, newline, trailing):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN the updated text retains its line endings."""
    original, expected = newline_patch
    result = apply_v4a_diff_text(original, "@@\n first\n-old\n+new\n last")
    assert result == expected


@pytest.fixture
def multiple_hunks():
    original = "before\nold\nbetween\nremove\nafter\n"
    patch = "@@\n-old\n+new\n@@\n-remove"
    return (original, patch)


def test_multiple_hunks_leave_unmatched_lines_unchanged(multiple_hunks):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN the patch changes only the specified lines."""
    original, patch = multiple_hunks
    result = apply_v4a_diff_text(original, patch)
    expected = "before\nnew\nbetween\nafter\n"
    assert result == expected


@pytest.mark.parametrize("original", ["", "one\n", "one\r\ntwo", "one"])
def test_empty_diff_is_identity(original):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN the text does not change."""
    result = apply_v4a_diff_text(original, "")
    expected = original
    assert result == expected


@pytest.mark.parametrize("original,patch,expected", [("old", "+first\n+\n old", "first\n\nold"), ("", "+new", "new")])
def test_insertion_only_and_blank_lines(original, patch, expected):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN the result contains the inserted lines."""
    result = apply_v4a_diff_text(original, patch)
    assert result == expected


@pytest.mark.parametrize("patch", ["bad", "@@\n\n", "?unknown"])
def test_invalid_update_prefix(patch):
    """GIVEN a patch with an invalid line prefix.
    WHEN the patch operation is called.
    THEN the invalid line prefix is rejected."""
    with pytest.raises(V4APatchError, match="Invalid V4A") as exc:
        apply_v4a_diff_text("old", patch)
    expected = V4APatchError
    assert exc.type == expected


def test_missing_hunk_context():
    """GIVEN a patch with context absent from the source.
    WHEN the patch operation is called.
    THEN the missing context is reported."""
    with pytest.raises(V4APatchError, match="context not found") as exc:
        apply_v4a_diff_text("old", "-missing\n+new")
    expected = V4APatchError
    assert exc.type == expected


def test_hunks_cannot_reuse_consumed_context():
    """GIVEN patch hunks that use the same source context.
    WHEN the patch operation is called.
    THEN the repeated context is rejected."""
    with pytest.raises(V4APatchError, match="context not found") as exc:
        apply_v4a_diff_text("old", "-old\n+new\n@@\n-old\n+again")
    expected = V4APatchError
    assert exc.type == expected


def test_create_mode():
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN the result contains the added lines."""
    result = apply_v4a_diff_text("ignored", "@@\n+first\n+\n+last", ApplyV4AMode.CREATE)
    expected = "first\n\nlast"
    assert result == expected


@pytest.mark.parametrize("patch", [" context", "-deleted", "+first\n\n+last"])
def test_create_rejects_non_additions(patch):
    """GIVEN a creation patch with invalid lines.
    WHEN the patch operation is called.
    THEN the invalid creation patch is rejected."""
    with pytest.raises(V4APatchError, match="Invalid V4A create diff") as exc:
        apply_v4a_diff_text("", patch, ApplyV4AMode.CREATE)
    expected = V4APatchError
    assert exc.type == expected


@pytest.fixture
def inverted_patch():
    original = "first\nold\nlast\n"
    patch = "@@\n first\n-old\n+new\n+extra\n last"
    updated = apply_v4a_diff_text(original, patch)
    return (original, updated, invert_v4a_diff(patch), patch)


def test_inverted_patch_restores_original(inverted_patch):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN the original text is restored."""
    original, updated, inverse, patch = inverted_patch
    result = apply_v4a_diff_text(updated, inverse)
    expected = original
    assert result == expected


def test_inversion_is_reversible(inverted_patch):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN the original patch is restored."""
    original, updated, inverse, patch = inverted_patch
    result = invert_v4a_diff(inverse)
    expected = patch
    assert result == expected


@pytest.mark.parametrize("query,choices,expected", [("file.py", [], None), ("./src/../src/file.py", ["src/other.py", "src/file.py"], "src/file.py"), ("src/file.py", ["src/files.py", "other/file.py"], "other/file.py"), ("app/models/item.py", ["test/item.py", "app/models/item.py"], "app/models/item.py")])
def test_closest_path(query, choices, expected):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN the result is the closest allowed path or no path."""
    result = diffs.closest_path_stdlib(query, choices)
    assert result == expected


@pytest.fixture
def file_target(tmp_path):
    return tmp_path / "nested" / "test.py"


def test_file_create(file_target):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN the new file contains the patch text."""
    diffs.create_file(str(file_target), "+old\n+line")
    result = file_target.read_text()
    expected = "old\nline"
    assert result == expected


@pytest.fixture
def created_file(file_target):
    diffs.create_file(str(file_target), "+old\n+line")
    return file_target


def test_file_update(created_file):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN the file contains the updated text."""
    diffs.apply_patch(created_file, "-old\n+new")
    result = created_file.read_text()
    expected = "new\nline"
    assert result == expected


def test_file_delete(created_file):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN the file is removed."""
    diffs.delete_file(str(created_file))
    result = not created_file.exists()
    expected = True
    assert result == expected


@pytest.fixture
def unmatched_patch(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("original")
    return target


def test_failed_patch_leaves_file_unchanged(unmatched_patch):
    """GIVEN a patch that does not match the file.
    WHEN the patch operation is called.
    THEN the patch error is reported and the file does not change."""
    target = unmatched_patch
    with pytest.raises(V4APatchError):
        diffs.apply_patch(target, "-missing\n+replacement")
    result = target.read_text()
    expected = "original"
    assert result == expected


@pytest.fixture
def patch_case(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "file.txt"
    target.write_text("old\n")
    operation = {"type": "update_file", "path": str(target), "diff": "-old\n+new"}
    item = SimpleNamespace(model_dump=lambda: {"type": "apply_patch_call", "id": "call-1", "operation": operation})
    return (target, item, {})


@pytest.fixture
def apply_persists_history_case(patch_case, tmp_path):
    target, item, history = patch_case
    return (history, item, target, tmp_path)


def test_patch_updates_file(apply_persists_history_case):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN patch updates file."""
    history, item, target, tmp_path = apply_persists_history_case
    diffs.apply_diff(item, [str(target)], history)
    expected = "new\n"
    assert target.read_text() == expected


def test_patch_saves_history(apply_persists_history_case):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN patch saves history."""
    history, item, target, tmp_path = apply_persists_history_case
    diffs.apply_diff(item, [str(target)], history)
    expected = {"call-1": str(target)}
    assert diffs.load_last_diffs() == expected


def test_patch_saves_diff(apply_persists_history_case):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN patch saves diff."""
    history, item, target, tmp_path = apply_persists_history_case
    diffs.apply_diff(item, [str(target)], history)
    expected = "-old\n+new"
    assert (tmp_path / "diff_tmp" / "call-1.txt").read_text() == expected


@pytest.fixture
def applied_patch(patch_case):
    target, item, history = patch_case
    diffs.apply_diff(item, [str(target)], history)
    return target


def test_undo_restores_text(applied_patch, tmp_path):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN undo restores text."""
    diffs.undo_last_diff()
    expected = "old\n"
    assert applied_patch.read_text() == expected


def test_undo_removes_saved_patch(applied_patch, tmp_path):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN undo removes saved patch."""
    diffs.undo_last_diff()
    expected = False
    assert (tmp_path / "diff_tmp" / "call-1.txt").exists() == expected


@pytest.fixture
def forbidden_patch(tmp_path, capsys):
    target = tmp_path / "file.txt"
    item = SimpleNamespace(model_dump=lambda: {"type": "apply_patch_call", "id": "call-1", "operation": {"type": "create_file", "path": str(target), "diff": "+text"}})
    return (item, target, capsys)


def test_forbidden_patch_creates_no_file(forbidden_patch):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN forbidden patch creates no file."""
    item, target, capsys = forbidden_patch
    diffs.apply_diff(item, [], {})
    expected = False
    assert target.exists() is expected


def test_forbidden_patch_reports_path(forbidden_patch):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN forbidden patch reports path."""
    item, target, capsys = forbidden_patch
    diffs.apply_diff(item, [], {})
    expected = "Path not found"
    assert expected in capsys.readouterr().out


@pytest.fixture
def file_operation(tmp_path, monkeypatch, operation):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "file.txt"
    if operation == "delete_file":
        target.write_text("existing")
    item = SimpleNamespace(model_dump=lambda: {"type": "apply_patch_call", "id": "call-id", "operation": {"type": operation, "path": str(target), "diff": "+created"}})
    history = {}
    return (history, item, target, tmp_path)


@pytest.mark.parametrize("operation", ["create_file", "delete_file"])
def test_file_operation_changes_existence(file_operation, operation):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN file operation changes existence."""
    history, item, target, tmp_path = file_operation
    diffs.apply_diff(item, [str(target)], history)
    expected = operation == "create_file"
    assert target.exists() is expected


@pytest.mark.parametrize("operation", ["create_file"])
def test_file_creation_writes_content(file_operation, operation):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN file creation writes content."""
    history, item, target, tmp_path = file_operation
    diffs.apply_diff(item, [str(target)], history)
    expected = "created"
    assert target.read_text() == expected


@pytest.mark.parametrize("operation", ["create_file", "delete_file"])
def test_file_operation_retains_history(file_operation, operation):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN file operation retains history."""
    history, item, target, tmp_path = file_operation
    diffs.apply_diff(item, [str(target)], history)
    expected = {}
    assert history == expected


@pytest.mark.parametrize("operation", ["create_file", "delete_file"])
def test_file_operation_saves_no_history(file_operation, operation):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN file operation saves no history."""
    history, item, target, tmp_path = file_operation
    diffs.apply_diff(item, [str(target)], history)
    expected = True
    assert (not (tmp_path / "last_diffs.json").exists()) == expected


@pytest.fixture
def relative_patch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "file.txt"
    target.write_text("old")
    item = SimpleNamespace(model_dump=lambda: {"type": "apply_patch_call", "id": "id", "operation": {"type": "update_file", "path": "file.txt", "diff": "-old\n+new"}})
    history = {}
    return (history, item, target)


def test_relative_patch_updates_file(relative_patch):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN relative patch updates file."""
    history, item, target = relative_patch
    diffs.apply_diff(item, [str(target)], history)
    result = target.read_text()
    expected = "new"
    assert result == expected


def test_relative_patch_records_path(relative_patch):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN relative patch records path."""
    history, item, target = relative_patch
    diffs.apply_diff(item, [str(target)], history)
    expected = {"id": str(target)}
    assert history == expected


@pytest.fixture
def unknown_operation(tmp_path, capsys):
    target = tmp_path / "file.txt"
    target.write_text("original")
    item = SimpleNamespace(model_dump=lambda: {"type": "apply_patch_call", "id": "id", "operation": {"type": "unknown", "path": str(target)}})
    return (item, target, capsys)


def test_unknown_patch_preserves_file(unknown_operation):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN unknown patch preserves file."""
    item, target, capsys = unknown_operation
    diffs.apply_diff(item, [str(target)], {})
    result = target.read_text()
    expected = "original"
    assert result == expected


def test_unknown_patch_reports_operation(unknown_operation):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN unknown patch reports operation."""
    item, target, capsys = unknown_operation
    diffs.apply_diff(item, [str(target)], {})
    expected = "Unknown diff operation"
    assert expected in capsys.readouterr().out


@pytest.fixture
def failed_undo(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "file.txt"
    target.write_text("unrelated content")
    diffs.save_last_diffs({"id": str(target)})
    diffs.save_diff("id", "-old\n+new")
    return (target, tmp_path, capsys)


def test_failed_undo_preserves_file(failed_undo):
    """GIVEN a saved patch and file text that does not match it.
    WHEN the patch operation is called.
    THEN failed undo preserves file."""
    target, tmp_path, capsys = failed_undo
    diffs.undo_last_diff()
    result = target.read_text()
    expected = "unrelated content"
    assert result == expected


def test_failed_undo_retains_patch(failed_undo):
    """GIVEN a saved patch and file text that does not match it.
    WHEN the patch operation is called.
    THEN failed undo retains patch."""
    target, tmp_path, capsys = failed_undo
    diffs.undo_last_diff()
    expected = True
    assert (tmp_path / "diff_tmp" / "id.txt").exists() == expected


def test_failed_undo_reports_error(failed_undo):
    """GIVEN a saved patch and file text that does not match it.
    WHEN the patch operation is called.
    THEN failed undo reports error."""
    target, tmp_path, capsys = failed_undo
    diffs.undo_last_diff()
    expected = "Error while undoing diff"
    assert expected in capsys.readouterr().out


@pytest.fixture
def failed_operation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "file.txt"
    target.write_text("original")
    item = SimpleNamespace(model_dump=lambda: {"type": "apply_patch_call", "id": "id", "operation": {"type": "update_file", "path": str(target), "diff": "-missing\n+new"}})
    history = {}
    return (history, item, target, tmp_path)


def test_failed_patch_preserves_file(failed_operation):
    """GIVEN a patch operation that cannot be applied.
    WHEN the patch operation is called.
    THEN failed patch preserves file."""
    history, item, target, tmp_path = failed_operation
    diffs.apply_diff(item, [str(target)], history)
    result = target.read_text()
    expected = "original"
    assert result == expected


def test_failed_patch_retains_history(failed_operation):
    """GIVEN a patch operation that cannot be applied.
    WHEN the patch operation is called.
    THEN failed patch retains history."""
    history, item, target, tmp_path = failed_operation
    diffs.apply_diff(item, [str(target)], history)
    expected = {}
    assert history == expected


def test_failed_patch_saves_no_history(failed_operation):
    """GIVEN a patch operation that cannot be applied.
    WHEN the patch operation is called.
    THEN failed patch saves no history."""
    history, item, target, tmp_path = failed_operation
    diffs.apply_diff(item, [str(target)], history)
    expected = True
    assert (not (tmp_path / "last_diffs.json").exists()) == expected


@pytest.fixture
def provider_patch(client, monkeypatch, tmp_path):
    target = tmp_path / "created.txt"
    item = SimpleNamespace(model_dump=lambda: {"type": "apply_patch_call", "id": "id", "operation": {"type": "create_file", "path": str(target), "diff": "+content"}})
    send = AsyncMock(return_value=SimpleNamespace(output=[item]))
    monkeypatch.setattr(client, "async_responses_create", send)
    module = diffs.Diffs(client=client, model="gpt-4o")
    return (module, send, target)


@pytest.mark.asyncio
async def test_provider_patch_creates_file(provider_patch):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN provider patch creates file."""
    module, send, target = provider_patch
    await module.apply_diff("create file", [str(target)], verbose=True)
    result = target.read_text()
    expected = "content"
    assert result == expected


@pytest.mark.asyncio
async def test_provider_patch_passes_request(provider_patch):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN provider patch passes request."""
    module, send, target = provider_patch
    await module.apply_diff("create file", [str(target)], verbose=True)
    expected = [mock_call("create file", model="gpt-4o", verbose=True, tools=[{"type": "apply_patch"}])]
    assert send.await_args_list == expected


@pytest.fixture
def empty_reply(client, monkeypatch):
    monkeypatch.setattr(client, "async_responses_create", AsyncMock(return_value=ClientReply(response=SimpleNamespace(output=[]))))
    module = diffs.Diffs(client=client)
    return module


@pytest.mark.asyncio
async def test_diffs_module_accepts_client_reply(empty_reply):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN the empty response is accepted."""
    module = empty_reply
    result = await module.apply_diff("no edits", [])
    result = result is None
    expected = True
    assert result == expected


@pytest.fixture
def saved_history(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "file.txt"
    target.write_text("new")
    diffs.save_last_diffs({"id": str(target)})
    diffs.save_diff("id", "-old\n+new")
    module = diffs.Diffs(client=client)
    return (target, client, module)


def test_diffs_module_undo_uses_saved_history(saved_history):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN the saved patch restores the original text."""
    target, client, module = saved_history
    module.undo_last_diffs()
    result = target.read_text()
    expected = "old"
    assert result == expected


@pytest.fixture
def wrapped_patch(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "updated.txt"
    target.write_text("old")
    item = SimpleNamespace(model_dump=lambda: {"type": "apply_patch_call", "id": "patch-1", "operation": {"type": "update_file", "path": str(target), "diff": "-old\n+new"}})
    client.async_client.responses.create.return_value = SimpleNamespace(output_text="", output=[item], usage=SimpleNamespace(input_tokens=10, output_tokens=5))
    return (diffs.Diffs(client=client, model="test"), target)


@pytest.mark.asyncio
async def test_wrapped_patch_updates_file(wrapped_patch):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN wrapped patch updates file."""
    module, target = wrapped_patch
    await module.apply_diff("edit", [str(target)])
    expected = "new"
    assert target.read_text() == expected


@pytest.mark.asyncio
async def test_wrapped_patch_saves_history(wrapped_patch):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN wrapped patch saves history."""
    module, target = wrapped_patch
    await module.apply_diff("edit", [str(target)])
    expected = {"patch-1": str(target)}
    assert diffs.load_last_diffs() == expected


@pytest_asyncio.fixture
async def wrapped_history(wrapped_patch):
    module, target = wrapped_patch
    await module.apply_diff("edit", [str(target)])
    return target


@pytest.mark.asyncio
async def test_wrapped_response_can_be_undone(wrapped_history):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN the original text is restored."""
    diffs.undo_last_diff()
    result = wrapped_history.read_text()
    expected = "old"
    assert result == expected


@pytest.fixture
def patch_failure(client):
    error = RuntimeError("provider unavailable")
    client.async_client.responses.create.side_effect = error
    module = diffs.Diffs(client=client)
    return (error, client, module)


@pytest.mark.asyncio
async def test_diffs_module_chains_provider_failure(patch_failure):
    """GIVEN an operation that reports an error.
    WHEN the patch operation is called.
    THEN the error retains the provider error as its cause."""
    error, client, module = patch_failure
    with pytest.raises(ClientError) as exc:
        await module.apply_diff("edit", [])
    expected = True
    assert (exc.value.__cause__ is error) == expected


@pytest.fixture
def unrelated_patch():
    item = SimpleNamespace(model_dump=lambda: {"type": "message"})
    history = {"id": "file.txt"}
    return (item, history)


def test_apply_diff_ignores_unrelated_items(unrelated_patch):
    """GIVEN text and a file patch.
    WHEN the patch operation is called.
    THEN the saved history does not change."""
    item, history = unrelated_patch
    diffs.apply_diff(item, [], history)
    expected = {"id": "file.txt"}
    assert history == expected
