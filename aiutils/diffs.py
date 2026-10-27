
from enum import StrEnum
import os
import json
from pathlib import Path, PurePosixPath
import posixpath
from difflib import SequenceMatcher


from .exceptions import V4APatchError


class ApplyV4AMode(StrEnum):
    DEFAULT = "default"
    CREATE = "create"

class DiffOp(StrEnum):
    CREATE_FILE = "create_file"
    DELETE_FILE = "delete_file"
    UPDATE_FILE = "update_file"

def _norm(p: str) -> str:
    return posixpath.normpath(p)

def closest_path_stdlib(query: str, choices: list[str]) -> str | None:
    if not choices:
        return None

    qn = _norm(query)
    qpp = PurePosixPath(qn)
    q_name = qpp.name
    q_parent = str(qpp.parent)

    norm_choices = [(_norm(c), c) for c in choices]
    same_name = [(nc, orig) for (nc, orig) in norm_choices if PurePosixPath(nc).name == q_name]
    candidates = same_name if same_name else norm_choices

    def ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()  # 0..1

    def score(nc: str) -> float:
        cpp = PurePosixPath(nc)
        c_parent = str(cpp.parent)

        dir_sim = ratio(q_parent, c_parent)  # 0..1

        # common suffix boost
        q_parts = qpp.parts[:-1]
        c_parts = cpp.parts[:-1]
        common_suffix = 0
        for a, b in zip(reversed(q_parts), reversed(c_parts)):
            if a != b:
                break
            common_suffix += 1

        return dir_sim + 0.08 * common_suffix

    best_norm, best_orig = max(candidates, key=lambda t: score(t[0]))
    return best_orig


def _detect_newline_style(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    if "\r" in text and "\n" not in text:
        return "\r"
    return "\n"


def _has_trailing_newline(text: str) -> bool:
    return text.endswith(("\n", "\r"))


def _find_subsequence(haystack: list[str], needle: list[str], start: int) -> int:
    if not needle:
        return start
    n = len(needle)
    for idx in range(start, len(haystack) - n + 1):
        if haystack[idx: idx + n] == needle:
            return idx
    return -1


def apply_v4a_diff_text(
    input_text: str, 
    diff: str, 
    mode: ApplyV4AMode = ApplyV4AMode.DEFAULT
) -> str:
    """
    Apply a headerless V4A diff to input_text.

    mode=ApplyV4AMode.DEFAULT:
      ' ' context, '-' delete, '+' insert, hunks separated by '@@'
    mode=ApplyV4AMode.CREATE:
      create-file syntax: every non-'@@' line must start with '+'
    """
    diff_lines = (diff or "").splitlines()

    if mode == ApplyV4AMode.CREATE:
        out: list[str] = []
        for line in diff_lines:
            if line.startswith("@@"):
                continue
            if line == "":
                raise V4APatchError("Invalid V4A create diff: empty line without '+' prefix.")
            if not line.startswith("+"):
                raise V4APatchError(
                    f"Invalid V4A create diff: expected '+' prefix, got {line[:80]!r}"
                )
            out.append(line[1:])
        # For create, newline style is your choice; default to '\n'
        return "\n".join(out)

    # --- default/update mode (your existing logic) ---
    newline = _detect_newline_style(input_text)
    had_trailing_nl = _has_trailing_newline(input_text)

    original = input_text.splitlines()
    hunks: list[list[str]] = []
    cur: list[str] = []

    for line in diff_lines:
        if line.startswith("@@"):
            if cur:
                hunks.append(cur)
                cur = []
            continue
        if line == "":
            raise V4APatchError("Invalid V4A diff: encountered empty line without a prefix.")
        if line[0] not in (" ", "+", "-"):
            raise V4APatchError(f"Invalid V4A diff prefix {line[0]!r} in line: {line[:80]!r}")
        cur.append(line)

    if cur:
        hunks.append(cur)

    out: list[str] = []
    pos = 0

    for hunk in hunks:
        needle = [ln[1:] for ln in hunk if ln[0] in (" ", "-")]
        start = _find_subsequence(original, needle, pos)
        if start < 0:
            preview = "\n".join(hunk)
            raise V4APatchError(
                "Could not apply V4A diff: hunk context not found.\n"
                f"Hunk preview:\n{preview}"
            )

        out.extend(original[pos:start])

        idx = start
        for ln in hunk:
            tag, content = ln[0], ln[1:]
            if tag == " ":
                if idx >= len(original) or original[idx] != content:
                    raise V4APatchError(
                        f"Context mismatch.\nExpected: {content!r}\nFound: "
                        f"{(original[idx] if idx < len(original) else None)!r}"
                    )
                out.append(content)
                idx += 1
            elif tag == "-":
                if idx >= len(original) or original[idx] != content:
                    raise V4APatchError(
                        f"Deletion mismatch.\nExpected: {content!r}\nFound: "
                        f"{(original[idx] if idx < len(original) else None)!r}"
                    )
                idx += 1
            else:  # '+'
                out.append(content)

        pos = idx

    out.extend(original[pos:])

    result = newline.join(out)
    if had_trailing_nl:
        result += newline
    return result


def invert_v4a_diff(diff: str) -> str:
    out = []
    for line in diff.splitlines():
        if line.startswith("@@"):
            out.append(line)
        elif line.startswith("+"):
            out.append("-" + line[1:])
        elif line.startswith("-"):
            out.append("+" + line[1:])
        elif line.startswith(""):
            out.append(line)
        elif line == "":
            # If your generator ever emits truly empty lines (rare), preserve them.
            out.append(line)
        else:
            raise ValueError(f"Invalid V4A diff line: {line!r}")
    return "\n".join(out)


def load_last_diffs():
    cwd = Path.cwd()
    last_diffs_path = cwd / "last_diffs.json"
    with open(last_diffs_path, "r") as f:
        last_diffs = json.load(f)
    return last_diffs

def save_last_diffs(last_diffs):
    cwd = Path.cwd()
    last_diffs_path = cwd / "last_diffs.json"
    with open(last_diffs_path, "w") as f:
        json.dump(last_diffs, f, indent=4)

def save_diff(id, diff):
    cwd = Path.cwd()
    diff_path = cwd / "diff_tmp" / f"{id}.txt"
    path = cwd / "diff_tmp"
    path.mkdir(parents=True, exist_ok=True)
    with open(diff_path, "w") as f:
        f.write(diff)

def apply_patch(path, diff):
    file_path = Path(path)
    old = file_path.read_text(encoding="utf-8")
    new = apply_v4a_diff_text(old, diff)
    file_path.write_text(new, encoding="utf-8")

def create_file(path: str, diff: str):
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    new = apply_v4a_diff_text("", diff, mode=ApplyV4AMode.CREATE)
    file_path.write_text(new, encoding="utf-8")
    print(f"Created file: {path}")

def delete_file(path: str):
    file_path = Path(path)
    file_path.unlink()
    print(f"Deleted file: {path}")


def undo_last_diff():
    last_diffs = load_last_diffs()
    diff_paths = []
    cwd = Path.cwd()

    try:
        for id, path in last_diffs.items():
            diff_path = cwd / "diff_tmp" / f"{id}.txt"
            diff_paths.append(diff_path)
            with open(diff_path, "r") as f:
                diff = f.read()
            undo_diff = invert_v4a_diff(diff)
            apply_patch(path, undo_diff)
        for diff_path in diff_paths:
            os.remove(diff_path)
    except Exception as e:
        print(f"Error while undoing diff: {e}")


def apply_diff(item, filepaths, last_diffs):
    item = item.model_dump()

    if item.get("type") == "apply_patch_call":
        operation = item.get("operation")
        id = item.get("id")
        # 'create_file', 'update_file', 'delete_file'
        diff_type = operation.get("type")
        path = operation.get("path")
        diff = ""

        if path not in filepaths:
            path = closest_path_stdlib(path, filepaths)
        if operation.get("diff") is not None:
            diff = operation.get("diff")

        if path is None:
            print(f"Path not found for diff id: {id}")
            return

        match diff_type:
            case "create_file":
                create_file(path, diff)
                print(f"Applied {diff_type} patch to {path}")

            case "update_file":
                try:
                    apply_patch(path, diff)
                except Exception as e:
                    print(f"Error while applying patch: {e}")

                last_diffs[id] = path
                save_diff(id, diff)
                save_last_diffs(last_diffs)
                print(f"Applied {diff_type} patch to {path}")

            case "delete_file":
                delete_file(path)
                print(f"Applied {diff_type} patch to {path}")

            case _:
                print(f"Unknown diff operation type: {diff_type}")
