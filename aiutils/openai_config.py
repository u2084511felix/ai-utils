from __future__ import annotations
import os
import json
from pydantic import BaseModel
from dataclasses import dataclass, field
import pprint
from typing import Literal, Optional, List, Dict, Any
from enum import Enum
import pdb
import subprocess
from tempfile import NamedTemporaryFile
#!/usr/bin/env python3

import sys
from pathlib import Path, PurePosixPath
import posixpath
from difflib import SequenceMatcher

from aiutils import (
    client,
    encoding,
    TextModels,
    EmbeddingModels,
    Models,
    reasoning_models,
    pedantic_resoning,
    brave_client,
    google_client,
    AudioModels
)

import importlib
from importlib import reload

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



class Client:
    def __init__(self):
        self.client = client
        self.vendor = "openai"

    def set_vendor(self, vendor):
        if vendor == "openai":
            self.client = client
            self.vendor = "openai"
        elif vendor == "google":
            self.client = google_client
            self.vendor = "google"
        elif vendor == "brave":
            self.client = brave_client
            self.vendor = "brave"


cclient = Client()


def clean_pydantic_model(model):
    model_dict = model.dict()
    clean_dict = {k: v for k, v in model_dict.items() if v is not None}
    return clean_dict


class UserLocation(BaseModel):
    type: str = field(default_factory="approximate")
    country: str
    city: str
    region: str
    timzeone: Optional[str] = None


"""
NOTE: 
    search_context_size: "low", "medium", "high"
    "new something"
"""


class SearchTools(BaseModel):
    type: str = "web_search_preview"
    search_context_size: Optional[str] = None
    user_location: Optional[UserLocation] = None


class WebSearchResponsesModel(BaseModel):
    model: str = f"{TextModels.gpt_4_1_nano}"
    tools: List = [SearchTools]
    input: str = None


class GPT_Module_Params(BaseModel):
    messages: list
    max_tokens: int
    temperature: float
    model: str
    reasoning_effort: str
    frequency_penalty: float
    presence_penalty: float
    stop: str
    top_p: float
    logit_bias: dict
    logprobs: bool
    top_logprobs: int
    n: int
    response_format: dict
    seed: str
    stream: bool
    tools: list
    tool_choice: dict
    user: str


@dataclass
class GPTModule:
    request_body: Optional[Dict[str, any]] = field(default_factory=dict)
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    messages: Optional[list] = field(default_factory=list)
    token_usage: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list] = field(default_factory=list)
    top_p: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = field(default_factory=dict)
    logprobs: Optional[int] = None
    top_logprobs: Optional[int] = None
    n: Optional[int] = None
    response_format: Optional[any] = field(default_factory=list)
    seed: Optional[int] = None
    stream: Optional[bool] = None
    tools: Optional[list] = field(default_factory=list)
    tool_choice: Optional[Dict[str, any]] = field(default_factory=dict)
    user: Optional[str] = None


def make_req_body(module: GPTModule):
    request_body = {}
    for attr in module.__dict__:
        value = getattr(module, attr)
        if attr == 'request_body' or value is None:
            continue
        if value == [] or value == {} or value == "":
            continue
        request_body[attr] = value
    return request_body






def _detect_newline_style(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    if "\r" in text and "\n" not in text:
        return "\r"
    return "\n"


def _has_trailing_newline(text: str) -> bool:
    return text.endswith("\n") or text.endswith("\r")


def _find_subsequence(haystack: list[str], needle: list[str], start: int) -> int:
    if not needle:
        return start
    n = len(needle)
    for i in range(start, len(haystack) - n + 1):
        if haystack[i: i + n] == needle:
            return i
    return -1


class V4APatchError(RuntimeError):
    pass

ApplyV4AMode = Literal["default", "create"]

def apply_v4a_diff_text(input_text: str, diff: str, mode: ApplyV4AMode = "default") -> str:
    """
    Apply a headerless V4A diff to input_text.

    mode="default":
      ' ' context, '-' delete, '+' insert, hunks separated by '@@'
    mode="create":
      create-file syntax: every non-'@@' line must start with '+'
    """
    diff_lines = (diff or "").splitlines()

    if mode == "create":
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
    cwd = os.getcwd()
    last_diffs_path = cwd + "/last_diffs.json"
    with open(last_diffs_path, "r") as f:
        last_diffs = json.load(f)
    return last_diffs

def save_last_diffs(last_diffs):
    cwd = os.getcwd()
    last_diffs_path = cwd + "/last_diffs.json"
    with open(last_diffs_path, "w") as f:
        json.dump(last_diffs, f, indent=4)

def save_diff(id, diff):
    cwd = os.getcwd()
    diff_path = cwd + "/diff_tmp/" + id + ".txt"
    path = cwd + "/diff_tmp"
    os.makedirs(path, exist_ok=True)
    with open(diff_path, "w") as f:
        f.write(diff)



def apply_patch(path, diff):
    file_path = Path(path)
    old = file_path.read_text(encoding="utf-8")
    new = apply_v4a_diff_text(old, diff)
    file_path.write_text(new, encoding="utf-8")

def create_file(path: str, diff: str):
    file_path = Path(path)
    new = apply_v4a_diff_text("", diff, mode="create")
    file_path.write_text(new, encoding="utf-8")
    print(f"Created file: {path}")


def delete_file(path: str):
    file_path = Path(path)
    file_path.unlink()
    print(f"Deleted file: {path}")


@dataclass
class Generate(GPTModule):
    def __init__(self):
        self.model = TextModels.gpt_4_1
        self.messages = []
        self.temperature = 0

    async def apply_diff(self, prompt, filepaths, verbose=False):

        model = TextModels.gpt_5_2
        tools = [{"type": "apply_patch"}]

        response = cclient.client.responses.create(
            model=model,
            input=prompt,
            tools=[{"type": "apply_patch"}],
        )
        if verbose == True:
            pr = response.model_dump()
            pprint.pprint(pr)

        # - update lib/fib.py
        # - update run.py
        last_diffs = {}
        for item in response.output:
            item = item.model_dump()


            if item.get("type") == "apply_patch_call":
                operation = item.get("operation")
                id = item.get("id")
                # 'create_file', 'update_file', 'delete_file'
                diff_type = operation.get("type")
                path = operation.get("path")

                if path not in filepaths:
                    path = closest_path_stdlib(path, filepaths)
                if operation.get("diff") is not None:
                    diff = operation.get("diff")
                    last_diffs[id] = path 
                    save_diff(id, diff)

                if diff_type == "create_file":
                    create_file(path, diff)
                elif diff_type == "update_file":
                    apply_patch(path, diff)
                    print(f"Applied {diff_type} patch to {path}")
                elif diff_type == "delete_file":
                    delete_file(path)
                else:
                    print(f"Unknown diff operation type: {diff_type}")

                save_last_diffs(last_diffs)
                print(f"Applied {diff_type} patch to {path}")


    def undo_last_diffs(self):
        last_diffs = load_last_diffs()
        diff_paths = []

        try:
            for id, path in last_diffs.items():
                cwd = os.getcwd()
                diff_path = cwd + "/diff_tmp/" + id + ".txt"
                diff_paths.append(diff_path)
                with open(diff_path, "r") as f:
                    diff = f.read()
                undo_diff = invert_v4a_diff(diff)
                apply_patch(path, undo_diff)
            for diff_path in diff_paths:
                os.remove(diff_path)
        except Exception as e:
            print(f"Error while undoing diff: {e}")



    async def web_search(self, tool_dict=None, **kwargs) -> str:
        """
        kwargs:
            model: str = f"{TextModels.hipster_mini}"
            tools: List = 
                type: "web_search_preview"
                search_context_size: Optional[str] = None
                user_location: Optional[UserLocation] = None

            input: str = None

        NOTE: search_context_size: "low", "medium", "high"

        """

        web_search_model = WebSearchResponsesModel(**kwargs)
        clean_model = clean_pydantic_model(web_search_model)

        if tool_dict is not None:
            new_tools = SearchTools()
            tool_dump = new_tools.model_dump()
            new_dict = {}
            for k, v in tool_dump.items():
                if v is None:
                    continue
                else:
                    if k in tool_dict.keys():
                        new_dict[k] = tool_dict[k]

            clean_model["tools"] = []
            clean_model["tools"].append(tool_dump)

        try:
            response = await ResponsesCall(**clean_model)
            return response

        except Exception as e:
            return e

    async def generate(self, system_message, prompt, model=TextModels.gpt_4_1, temperature=0, chat=True, reasoning_effort="default"):
        if cclient.vendor == "google":
            if model == TextModels.gemini_25_pro or model == TextModels.gemini_25_flash or model == TextModels.gemini_25_flash_lite or model == TextModels.gemini_3_pro:
                self.model = model
            else:
                self.model = TextModels.gemini_25_flash
        else:
            self.model = model
        if self.model in reasoning_models:
            self.temperature = 1
            if reasoning_effort == "default":
                self.reasoning_effort = "none"
            else:
                self.reasoning_effort = reasoning_effort
        elif self.model in pedantic_resoning:
            self.temperature = 1
            if reasoning_effort == "default":
                self.reasoning_effort = "minimal"
            else:
                self.reasoning_effort = reasoning_effort
        else:
            self.temperature = temperature
        self.messages.append({"role": "system", "content": system_message})
        self.messages.append({"role": "user", "content": prompt})

        self.request_body = make_req_body(self)

        if (chat):
            response = await Chat(self.request_body)
            return response

        else:
            response_body = await ChatBody(self.request_body)
            return response_body

    async def continued_response(self, assistant_response, prompt):
        self.messages.append({"role": "assistant", "content": prompt})
        self.messages.append({"role": "user", "content": prompt})
        self.request_body = make_req_body(self)

        response = await Chat(self.request_body)
        return response

    async def function_call_legacy_structured_output(self, system_message, prompt):
        self.model = TextModels.previous
        self.messages.append({"role": "system", "content": system_message})
        self.messages.append({"role": "user", "content": prompt})
        if self.model in reasoning_models:
            self.reasoning_effort = "none"
        self.request_body = make_req_body(self)

        response = await Chat(self.request_body)
        return response

    def call_function(self, function_response, available_functions):
        return send_functioncall_args_to_available_functions(function_response, available_functions)

    async def structured_output(self, system_message, prompt, schema={}, input_type="json", module_name='', model=TextModels.gpt_4_1, reasoning_effort="default"):
        if cclient.vendor == "google":
            if model == TextModels.gemini_25_pro or model == TextModels.gemini_25_flash or model == TextModels.gemini_25_flash_lite or model == TextModels.gemini_3_pro:
                self.model = model
            else:
                self.model = TextModels.gemini_25_flash
        else:
            self.model = model
        if self.model in reasoning_models:
            self.temperature = 1
            if reasoning_effort == "default":
                self.reasoning_effort = "none"
            else:
                self.reasoning_effort = reasoning_effort
        elif self.model in pedantic_resoning:
            self.temperature = 1
            if reasoning_effort == "default":
                self.reasoning_effort = "minimal"
            else:
                self.reasoning_effort = reasoning_effort
        self.messages.append({"role": "system", "content": system_message})
        self.messages.append({"role": "user", "content": prompt})

        if (input_type == 'json'):
            schema = json.loads(schema)
            self.response_format = {
                "type": "json_schema",
                "json_schema": schema
            }
            self.request_body = make_req_body(self)
            pprint.pprint(self.request_body)
            response = await ChatBody(self.request_body, input_type)

        elif (input_type == 'pydantic'):
            if (module_name != ''):
                module = importlib.import_module(module_name)
                module = reload(module)
                schema_title = schema
                model_class = getattr(module, schema_title)
                self.response_format = model_class

            else:
                self.response_format = schema

            self.request_body = make_req_body(self)
            response = await ChatBody(self.request_body, input_type)

        pprint.pprint(response)
        return response.choices[0].message.content


"""
Audio Models

"""


class SpeechMode(str, Enum):
    """High level presets for your audio pipeline."""
    high_quality = "high_quality"   # gpt-4o-transcribe + gpt-4o-mini-tts
    fast = "fast"                   # gpt-4o-mini-transcribe + gpt-4o-mini-tts
    legacy = "legacy"               # whisper-1 + tts-1


@dataclass
class SpeechIO:
    mode: SpeechMode = SpeechMode.high_quality
    tts_voice: str = "nova"
    tts_format: str = "mp3"

    stt_model: str = field(init=False)
    tts_model: str = field(init=False)

    def __post_init__(self):
        if self.mode == SpeechMode.fast:
            self.stt_model = AudioModels.gpt_4o_mini_transcribe
            self.tts_model = AudioModels.gpt_4o_mini_tts
        elif self.mode == SpeechMode.legacy:
            self.stt_model = AudioModels.whisper
            self.tts_model = AudioModels.tts
        else:
            self.stt_model = AudioModels.gpt_4o_transcribe
            self.tts_model = AudioModels.gpt_4o_mini_tts

    async def transcribe(
        self,
        file,
        *,
        response_format: str = "text",
        language: str | None = None,
        **kwargs: Any,
    ) -> str:
        params: dict[str, Any] = {
            "model": self.stt_model,
            "file": file,
            "response_format": response_format,
        }
        if language:
            params["language"] = language
        params.update(kwargs)

        # Make sure the file pointer is at the beginning
        try:
            file.seek(0)
        except Exception:
            pass

        result = cclient.client.audio.transcriptions.create(**params)

        if isinstance(result, str):
            return result
        if hasattr(result, "text"):
            return result.text
        return str(result)

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str | None = None,
        **kwargs: Any,
    ) -> bytes:
        """
        Text → audio using the configured TTS model.
        Returns raw bytes suitable for base64 encoding.
        """
        v = voice or self.tts_voice
        fmt = response_format or self.tts_format

        params: dict[str, Any] = {
            "model": self.tts_model,
            "voice": v,
            "input": text,
            "response_format": fmt,
        }
        params.update(kwargs)

        audio_resp = cclient.client.audio.speech.create(**params)

        # This is the key: read bytes from the response object
        if hasattr(audio_resp, "read"):
            return audio_resp.read()

        # Extra fallback just in case (you probably won't need this)
        try:
            return bytes(audio_resp)  # type: ignore[arg-type]
        except Exception:
            return str(audio_resp).encode("utf-8")


def create_generator_module(**kwargs):
    """
    args: (**kwargs)
    eg:

    some_generator = create_generator_module(
        temperature=0,
        model=Models.text.hipster_latest
    )
    """
    module = Generate()
    for key, value in kwargs.items():
        setattr(module, key, value)

    return module


async def ChatBody(params, input_type='json'):
    try:
        if (input_type == 'json'):
            response = cclient.client.chat.completions.create(**params)

        elif (input_type == 'pydantic'):
            response = cclient.client.beta.chat.completions.parse(**params)

        return response
    except Exception as e:
        print(e)


async def Chat(params: GPT_Module_Params):

    try:
        response = cclient.client.chat.completions.create(**params)
        response_message = response.choices[0].message
        if response_message.tool_calls:
            print("\n\nfunction call detected.\n\n")
            return (response_message.tool_calls)

        else:
            response_text = response.choices[0].message.content
            return (response_text)

    except Exception as e:
        print(e)


async def ResponsesCall(**kwargs):
    try:

        response = cclient.client.responses.create(**kwargs)
        return response.output_text
    except Exception as e:
        return e


def send_functioncall_args_to_available_functions(response, available_functions):

    return_values = {}

    try:
        for tool_call in response:
            print("tool call detected\n", tool_call, "\n")
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)

            print(str(function_args))

            if function_args == {}:
                return_value = function_to_call()
                return_value = str(return_value)
                print(return_value)
            else:
                return_value = function_to_call(**function_args)
                return_value = str(return_value)
                print(return_value)

            return_values[function_name] = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": return_value,
            }

        return return_values

    except Exception as e:
        print(f"\nException: {e}\n")
