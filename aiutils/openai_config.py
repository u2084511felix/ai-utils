from __future__ import annotations
import json

from dataclasses import dataclass, field
import pprint
from typing import Any
from enum import StrEnum
import importlib
from importlib import reload
from tempfile import NamedTemporaryFile
from dataclasses import dataclass
from threading import Lock
from typing import Any, Self


import logging


import sys
from pathlib import Path, PurePosixPath
import posixpath
from difflib import SequenceMatcher


from openai import APIResponse, AsyncAPIResponse
from pydantic import BaseModel


from .models import (
    TextModels,
    OpenAIModels,
    GoogleModels,
    ClassificationModels,
    REASONING_MODELS,
    PEDANTIC_REASONING,
    AudioModels,
    Vendor
)
from .budget import TokenUsage
from .diffs import undo_last_diff, apply_diff
from .clients import make_client, Client



def clean_pydantic_model(model):
    model_dict = model.dict()
    clean_dict = {k: v for k, v in model_dict.items() if v is not None}
    return clean_dict



@dataclass(frozen=True)
class ProviderReply:
    """Provider output and its reported token usage.

    Args:
        content: Text or tool calls returned by the provider.
        usage: Token counts reported for the call.
    """
    content: str | Exception
    usage: TokenUsage



class InputType(StrEnum):
    JSON = "json"
    PYDANTIC = "pydantic"


class UserLocation(BaseModel):
    type: str = "approximate"
    country: str
    city: str
    region: str
    timezone: str | None = None


"""
NOTE: 
    search_context_size: "low", "medium", "high"
    "new something"
"""


class SearchTools(BaseModel):
    type: str = "web_search_preview"
    search_context_size: str | None = None
    user_location: UserLocation | None = None


class WebSearchResponsesModel(BaseModel):
    model: str = TextModels.gpt_4_1_nano
    tools: list[SearchTools] = field(default_factory=list)
    input: str | None = None


class GPTModuleParams(BaseModel):
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
    request_body: dict[str, Any] | None = field(default_factory=dict)
    model: str | None = None
    reasoning_effort: str | None = None
    messages: list = field(default_factory=list)
    token_usage: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list | None = field(default_factory=list)
    top_p: float | None = None
    logit_bias: dict[str, float] | None = field(default_factory=dict)
    logprobs: int | None = None
    top_logprobs: int | None = None
    n: int | None = None
    response_format: Any | None = field(default_factory=list)
    seed: int | None = None
    stream: bool | None = None
    tools: list | None = field(default_factory=list)
    tool_choice: dict[str, Any] | None = field(default_factory=dict)
    user: str | None = None





def _openai_usage(
    response: APIResponse | AsyncAPIResponse
) -> TokenUsage:
    """Read token usage from a provider response.

    Args:
        response: Chat completion returned by the provider SDK.

    Returns:
        Validated token counts for the call.

    Raises:
        RuntimeError: The provider omitted usage data.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        raise RuntimeError("The model provider did not report token usage.")
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", 0) if details is not None else 0
    return TokenUsage(
        input_tokens=int(usage.prompt_tokens),
        output_tokens=int(usage.completion_tokens),
        reasoning_tokens=int(reasoning),
    )


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


def _parse_chat_completion(response):
    if response is None:
        return RuntimeError("No response received from the model.")

    if isinstance(response, (Exception,)):
        return response

    choices = response.choices[0]
    if not choices:
        return RuntimeError("No choices returned from the model.")

    response_msg = choices.message
    if response_msg.tool_calls:
        print("\n\nfunction call detected.\n\n")
        return response_msg.tool_calls
    else:
        if response_msg.content is None:
            return RuntimeError("The model provider returned no content.")
        return response_msg.content

    return ProviderReply(
        content=response, 
        usage=_openai_usage(response)
    )




@dataclass
class Generate(GPTModule):
    model: str = TextModels.gpt_5_2
    client: Client = make_client()
    logger: logging.Logger = logging.getLogger(__name__)

    def __post_init__(self):
        self.messages = []
        self.temperature = 0
        self._tools = {
            "apply_patch": [{"type": "apply_patch"}]
        }
        self._client = self.client.client
        self._async_client = self.client.async_client
        self._model = self.model

    def _response_create(self, prompt, tools=None, verbose=False, **kwargs):
        response = self._client.responses.create(
            input=prompt,
            model=self.model,
            tools=tools,
            **kwargs
        )
        if verbose:
            self.logger.debug(response.model_dump())
        return response

    async def _async_response_create(self, prompt, tools=None, verbose=False, **kwargs):
        response = await self._async_client.responses.create(
            input=prompt,
            model=self.model,
            tools=tools,
            **kwargs
        )
        if verbose:
            self.logger.debug(response.model_dump())
        return response


    def _chat_completions_body(self, params, verbose=False, input_type='json'):
        response = None
        comp = self._client.chat.completions
        try:
            match input_type:
                case 'json':     response = comp.create(**params)
                case 'pydantic': response = comp.parse(**params)
                case _:          raise ValueError(f"Unsupported input_type: {input_type}")
            return response
        except Exception as e:
            response = e
        if verbose:
            self.logger.debug(response)
        return response


    async def _async_chat_completions_body(self, params, verbose=False, input_type='json'):
        response = None
        comp = self._async_client.chat.completions
        try:
            match input_type:
                case 'json':     response = await comp.create(**params)
                case 'pydantic': response = await comp.parse(**params)
                case _:         raise ValueError(f"Unsupported input_type: {input_type}")
        except Exception as e:
            response = e
        if verbose:
            self.logger.debug(response)
        return response


    def _chat_completions(self, params, verbose=False, parse_response=True):
        response = None
        try:
            response = self._client.chat.completions.create(**params)
        except Exception as e:
            response = e
        if verbose:
            self.logger.debug(response)
        if not parse_response:
            return response
        return _parse_chat_completion(response)


    async def _async_chat_completions(self, params, verbose=False, parse_response=True):
        response = None
        try:
            response = await self._async_client.chat.completions.create(**params)
        except Exception as e:
            response = e
        if verbose:
            self.logger.debug(response)
        if not parse_response:
            return response
        return _parse_chat_completion(response)





    def _response_output_items(self, response):
        for item in response.output:
            yield item


    async def apply_diff(self, prompt, filepaths, verbose=False):

        response = await self._async_response_create(
            prompt, 
            verbose=verbose,
            tools=self._tools["apply_patch"],
        )

        # - update lib/fib.py
        # - update run.py
        last_diffs = {}

        for item in self._response_output_items(response):
            apply_diff(item, filepaths, last_diffs)


    def undo_last_diffs(self):
        return undo_last_diff()


    async def web_search(self, tool_dict=None, **kwargs) -> str:
        """
        kwargs:
            model: str = f"{TextModels.hipster_mini}"
            tools: List = 
                type: "web_search_preview"
                search_context_size: str | None = None
                user_location: UserLocation | None = None

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
            response = await self._async_response_create(**clean_model)
            return response
        except Exception as e:
            self.logger.error(f"Error in web_search: {e}")
            raise e

    def _set_reasoning_effort_and_temperature(self, reasoning_effort, temperature=None): 
        if self.model in REASONING_MODELS:
            self.temperature = 1
            if reasoning_effort == "default":
                self.reasoning_effort = "none"
            else:
                self.reasoning_effort = reasoning_effort
        elif self.model in PEDANTIC_REASONING:
            self.temperature = 1
            if reasoning_effort == "default":
                self.reasoning_effort = "minimal"
            else:
                self.reasoning_effort = reasoning_effort
        else:
            self.temperature = temperature or 0


    def _check_model(self, model):
        vendor = self._client.vendor
        try:
            model = TextModels(model)
        except ValueError:
            raise ValueError(f"Invalid model: {model}")
        if model in GoogleModels and vendor != Vendor.GOOGLE:
            raise ValueError(f"Invalid model for vendor Google: {model}")
        if model in OpenAIModels and vendor != Vendor.OPENAI:
            raise ValueError(f"Invalid model for vendor OpenAI: {model}")
        return model


    async def generate(
        self, 
        system_message: str, 
        prompt: str,
        model: str = TextModels.gpt_5_2, 
        temperature: float = 0, 
        chat: bool = True, 
        reasoning_effort: str = "default"
    ) -> str | Exception:
        """Returns a semantic LLM response string"""
        if not system_message:
            raise ValueError("system_message cannot be empty")
        if not prompt:
            raise ValueError("prompt cannot be empty")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")

        model = self._check_model(model)

        self.model = model
        self._set_reasoning_effort_and_temperature(reasoning_effort, temperature)

        self.messages.append({"role": "system", "content": system_message})
        self.messages.append({"role": "user", "content": prompt})

        self.request_body = make_req_body(self)

        if chat:
            response = await self._async_chat_completions(self.request_body)
        else:
            response = await self._async_chat_completions_body(self.request_body)
        return response


    async def continued_response(
        self, 
        assistant_response: str, 
        prompt:str
    ) -> str | Exception:
        self.messages.append({"role": "assistant", "content": assistant_response})
        self.messages.append({"role": "user", "content": prompt})
        self.request_body = make_req_body(self)
        return await self._async_chat_completions(self.request_body)


    async def function_call_legacy_structured_output(
        self, 
        system_message: str, 
        prompt: str
    ):
        self.model = TextModels.previous
        self.messages.append({"role": "system", "content": system_message})
        self.messages.append({"role": "user", "content": prompt})
        if self.model in REASONING_MODELS:
            self.reasoning_effort = "none"
        self.request_body = make_req_body(self)
        return await self._async_chat_completions(self.request_body)


    def call_function(self, function_response, available_funcs):
        return send_func_call_args_to_available_funcs(
            function_response, 
            available_funcs
        )


    async def structured_output(
        self, 
        system_message: str, 
        prompt: str, 
        schema: dict | type[BaseModel] = {}, 
        input_type: str = "json", 
        module_name: str = '', 
        model: str = TextModels.gpt_5_2, 
        reasoning_effort: str = "default", 
        assistance_response: str | None=None, 
        debug: bool = False
    ) -> str | Exception:
        """Returns a JSON string."""
        model = self._check_model(model)
        input_type = InputType(input_type)

        self.model = model

        self._set_reasoning_effort_and_temperature(reasoning_effort)

        if assistance_response is not None:
            msg = {"role": "assistant", "content": assistance_response}
        else:
            msg = {"role": "system", "content": system_message}

        self.messages.append(msg)
        self.messages.append({"role": "user", "content": prompt})

        match input_type:
            case InputType.JSON:
                schema = json.loads(schema)
                self.response_format = {
                    "type": "json_schema",
                    "json_schema": schema
                }
                self.request_body = make_req_body(self)
                response = await self._async_chat_completions_body(
                    self.request_body,
                    input_type=input_type,
                )
            case InputType.PYDANTIC:
                if module_name != '':
                    module = importlib.import_module(module_name)
                    module = reload(module)
                    schema_title = schema
                    model_class = getattr(module, schema_title)
                    self.response_format = model_class
                else:
                    self.response_format = schema

                self.request_body = make_req_body(self)
                response = await self._async_chat_completions_body(
                    self.request_body,
                    input_type=input_type,
                )
            case _:
                raise ValueError(f"Unsupported input_type: {input_type}")

        if debug == True:
            if assistance_response is not None:
                self.logger.debug("Running stateful structured output with previous assistant response in context.")
            self.logger.debug(response)
        
        if hasattr(response, "choices"):
            return response.choices[0].message.content
        
        else:
            return response




def make_generate(**kwargs):
    """
    args: (**kwargs)
    eg:

    some_generator = make_generate(
        temperature=0,
        model=Models.text.hipster_latest
    )
    """
    module = Generate()
    _ = [setattr(module, k, v) for k, v in kwargs.items()]
    return module





async def response_call(client, use_async=True, **kwargs):
    try:
        if use_async:
            response = await client.async_client.responses.create(**kwargs)
        else:
            response = client.responses.create(**kwargs)
        return response.output_text
    except Exception as e:
        raise e


def send_func_call_args_to_available_funcs(
    response, 
    available_functions
):
    func_dict = {}

    try:
        for tool_call in response:
            print("tool call detected\n", tool_call, "\n")
            name = tool_call.function.name
            call = available_functions[name]
            args = json.loads(tool_call.function.arguments)

            print(str(args))

            func_return = call(**args) if args else call()
            func_return = str(func_return)
            print(func_return)

            func_dict[name] = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": name,
                "content": func_return,
            }

        return func_dict

    except Exception as e:
        raise e
