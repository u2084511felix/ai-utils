from types import SimpleNamespace

import pytest

from aiutils.exceptions import V4APatchError
from aiutils.modules import diffs
from aiutils.modules.diffs import ApplyV4AMode, apply_v4a_diff_text, invert_v4a_diff


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("trailing", [True, False])
def test_update_preserves_newlines(newline, trailing):
    original = newline.join(["first", "old", "last"]) + (newline if trailing else "")
    expected = newline.join(["first", "new", "last"]) + (newline if trailing else "")
    assert apply_v4a_diff_text(original, "@@\n first\n-old\n+new\n last") == expected


def test_multiple_hunks_leave_unmatched_lines_unchanged():
    original = "before\nold\nbetween\nremove\nafter\n"
    patch = "@@\n-old\n+new\n@@\n-remove"
    assert apply_v4a_diff_text(original, patch) == "before\nnew\nbetween\nafter\n"


@pytest.mark.parametrize("original", ["", "one\n", "one\r\ntwo", "one"])
def test_empty_diff_is_identity(original):
    assert apply_v4a_diff_text(original, "") == original


def test_insertion_only_and_blank_lines():
    assert apply_v4a_diff_text("old", "+first\n+\n old") == "first\n\nold"
    assert apply_v4a_diff_text("", "+new") == "new"


@pytest.mark.parametrize("patch", ["bad", "@@\n\n", "?unknown"])
def test_invalid_update_prefix(patch):
    with pytest.raises(V4APatchError, match="Invalid V4A"):
        apply_v4a_diff_text("old", patch)


def test_missing_hunk_context():
    with pytest.raises(V4APatchError, match="context not found"):
        apply_v4a_diff_text("old", "-missing\n+new")


def test_hunks_cannot_reuse_consumed_context():
    with pytest.raises(V4APatchError, match="context not found"):
        apply_v4a_diff_text("old", "-old\n+new\n@@\n-old\n+again")


def test_create_mode():
    assert apply_v4a_diff_text("ignored", "@@\n+first\n+\n+last", ApplyV4AMode.CREATE) == "first\n\nlast"


@pytest.mark.parametrize("patch", [" context", "-deleted", "+first\n\n+last"])
def test_create_rejects_non_additions(patch):
    with pytest.raises(V4APatchError, match="Invalid V4A create diff"):
        apply_v4a_diff_text("", patch, ApplyV4AMode.CREATE)


def test_inverted_patch_restores_original():
    original = "first\nold\nlast\n"
    patch = "@@\n first\n-old\n+new\n+extra\n last"
    updated = apply_v4a_diff_text(original, patch)
    assert apply_v4a_diff_text(updated, invert_v4a_diff(patch)) == original
    assert invert_v4a_diff(invert_v4a_diff(patch)) == patch


@pytest.mark.parametrize("query,choices,expected", [
    ("file.py", [], None),
    ("./src/../src/file.py", ["src/other.py", "src/file.py"], "src/file.py"),
    ("src/file.py", ["src/files.py", "other/file.py"], "other/file.py"),
    ("app/models/item.py", ["test/item.py", "app/models/item.py"], "app/models/item.py"),
])
def test_closest_path(query, choices, expected):
    assert diffs.closest_path_stdlib(query, choices) == expected


def test_file_create_update_and_delete(tmp_path):
    target = tmp_path / "nested" / "test.py"
    diffs.create_file(str(target), "+old\n+line")
    assert target.read_text() == "old\nline"
    diffs.apply_patch(target, "-old\n+new")
    assert target.read_text() == "new\nline"
    diffs.delete_file(str(target))
    assert not target.exists()


def test_failed_patch_leaves_file_unchanged(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("original")
    with pytest.raises(V4APatchError):
        diffs.apply_patch(target, "-missing\n+replacement")
    assert target.read_text() == "original"


def test_apply_and_undo_persists_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "file.txt"
    target.write_text("old\n")
    operation = {"type": "update_file", "path": str(target), "diff": "-old\n+new"}
    item = SimpleNamespace(model_dump=lambda: {"type": "apply_patch_call", "id": "call-1", "operation": operation})
    history = {}
    diffs.apply_diff(item, [str(target)], history)
    assert target.read_text() == "new\n"
    assert diffs.load_last_diffs() == {"call-1": str(target)}
    assert (tmp_path / "diff_tmp" / "call-1.txt").read_text() == "-old\n+new"
    diffs.undo_last_diff()
    assert target.read_text() == "old\n"
    assert not (tmp_path / "diff_tmp" / "call-1.txt").exists()


def test_apply_diff_without_allowed_paths_does_nothing(tmp_path, capsys):
    target = tmp_path / "file.txt"
    item = SimpleNamespace(model_dump=lambda: {
        "type": "apply_patch_call", "id": "call-1",
        "operation": {"type": "create_file", "path": str(target), "diff": "+text"},
    })
    diffs.apply_diff(item, [], {})
    assert not target.exists()
    assert "Path not found" in capsys.readouterr().out


@pytest.mark.parametrize("operation", ["create_file", "delete_file"])
def test_apply_diff_dispatches_file_operations(tmp_path, monkeypatch, operation):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "file.txt"
    if operation == "delete_file":
        target.write_text("existing")
    item = SimpleNamespace(model_dump=lambda: {
        "type": "apply_patch_call", "id": "call-id",
        "operation": {"type": operation, "path": str(target), "diff": "+created"},
    })
    history = {}
    diffs.apply_diff(item, [str(target)], history)
    assert target.exists() is (operation == "create_file")
    if target.exists():
        assert target.read_text() == "created"
    assert history == {}
    assert not (tmp_path / "last_diffs.json").exists()


def test_apply_diff_resolves_path_and_ignores_unrelated_items(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "file.txt"
    target.write_text("old")
    item = SimpleNamespace(model_dump=lambda: {
        "type": "apply_patch_call", "id": "id",
        "operation": {"type": "update_file", "path": "file.txt", "diff": "-old\n+new"},
    })
    history = {}
    diffs.apply_diff(item, [str(target)], history)
    assert target.read_text() == "new"
    assert history == {"id": str(target)}
    diffs.apply_diff(SimpleNamespace(model_dump=lambda: {"type": "message"}), [], history)
    assert history == {"id": str(target)}


def test_unknown_diff_operation_does_not_modify_files(tmp_path, capsys):
    target = tmp_path / "file.txt"
    target.write_text("original")
    item = SimpleNamespace(model_dump=lambda: {
        "type": "apply_patch_call", "id": "id", "operation": {"type": "unknown", "path": str(target)},
    })
    diffs.apply_diff(item, [str(target)], {})
    assert target.read_text() == "original"
    assert "Unknown diff operation" in capsys.readouterr().out


def test_undo_failure_preserves_file_and_saved_diff(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "file.txt"
    target.write_text("unrelated content")
    diffs.save_last_diffs({"id": str(target)})
    diffs.save_diff("id", "-old\n+new")
    diffs.undo_last_diff()
    assert target.read_text() == "unrelated content"
    assert (tmp_path / "diff_tmp" / "id.txt").exists()
    assert "Error while undoing diff" in capsys.readouterr().out


def test_failed_operation_does_not_record_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "file.txt"
    target.write_text("original")
    item = SimpleNamespace(model_dump=lambda: {
        "type": "apply_patch_call", "id": "id",
        "operation": {"type": "update_file", "path": str(target), "diff": "-missing\n+new"},
    })
    history = {}
    diffs.apply_diff(item, [str(target)], history)
    assert target.read_text() == "original"
    assert history == {}
    assert not (tmp_path / "last_diffs.json").exists()


@pytest.mark.asyncio
async def test_diffs_module_applies_provider_operations(client, monkeypatch, tmp_path):
    from unittest.mock import AsyncMock
    target = tmp_path / "created.txt"
    item = SimpleNamespace(model_dump=lambda: {
        "type": "apply_patch_call", "id": "id",
        "operation": {"type": "create_file", "path": str(target), "diff": "+content"},
    })
    # Isolate operation dispatch with the raw response shape this method expects;
    # the ClientReply boundary mismatch is tested separately below.
    send = AsyncMock(return_value=SimpleNamespace(output=[item]))
    monkeypatch.setattr(client, "async_responses_create", send)
    module = diffs.Diffs(client=client, model="gpt-4o")
    await module.apply_diff("create file", [str(target)], verbose=True)
    assert target.read_text() == "content"
    send.assert_awaited_once_with("create file", model="gpt-4o", verbose=True, tools=[{"type": "apply_patch"}])


@pytest.mark.asyncio
async def test_diffs_module_accepts_client_reply(client, monkeypatch):
    from unittest.mock import AsyncMock
    from aiutils.clients import ClientReply
    monkeypatch.setattr(client, "async_responses_create", AsyncMock(return_value=ClientReply(response=SimpleNamespace(output=[]))))
    await diffs.Diffs(client=client).apply_diff("no edits", [])


def test_diffs_module_undo_uses_saved_history(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "file.txt"
    target.write_text("new")
    diffs.save_last_diffs({"id": str(target)})
    diffs.save_diff("id", "-old\n+new")
    diffs.Diffs(client=client).undo_last_diffs()
    assert target.read_text() == "old"


@pytest.mark.asyncio
async def test_diffs_module_applies_wrapped_response(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "updated.txt"
    target.write_text("old")
    item = SimpleNamespace(model_dump=lambda: {
        "type": "apply_patch_call", "id": "patch-1",
        "operation": {"type": "update_file", "path": str(target), "diff": "-old\n+new"},
    })
    client.async_client.responses.create.return_value = SimpleNamespace(
        output_text="", output=[item],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    await diffs.Diffs(client=client, model="test").apply_diff("edit", [str(target)])
    assert target.read_text() == "new"
    assert diffs.load_last_diffs() == {"patch-1": str(target)}
    diffs.undo_last_diff()
    assert target.read_text() == "old"


@pytest.mark.asyncio
async def test_diffs_module_chains_provider_failure(client):
    from aiutils.exceptions import ClientError
    error = RuntimeError("provider unavailable")
    client.async_client.responses.create.side_effect = error
    with pytest.raises(ClientError) as exc:
        await diffs.Diffs(client=client).apply_diff("edit", [])
    assert exc.value.__cause__ is error
