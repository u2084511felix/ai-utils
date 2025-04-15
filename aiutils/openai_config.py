import os
import json
from pydantic import BaseModel
from dataclasses import dataclass, field
import pprint
from typing import Optional, List, Dict, Any
from enum import Enum
from aiutils import client, encoding, TextModels, EmbeddingModels, Models
import importlib
from importlib import reload


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
"""


class SearchTools(BaseModel):
    type: str = "web_search_preview"
    search_context_size: Optional[str] = None
    user_location: Optional[UserLocation] = None



class WebSearchResponsesModel(BaseModel):
    model: str = f"{TextModels.hipster_mini}"
    tools: List = [SearchTools]
    input: str = None

class GPT_Module_Params(BaseModel):
    messages: list
    max_tokens: int
    temperature: float
    model: str
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


@dataclass
class Generate(GPTModule):
    def __init__(self):
        self.model = TextModels.latest
        self.messages = []
        self.temperature = 0

    async def web_search(self, **kwargs) -> str:
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
        tools_list = web_search_model["tools"][0]
        clean_tools = {k: v for k, v in tools_list.items() if v is not None}
        web_search_model["tools"] = []
        web_search_model["tools"].append(clean_tools)

        clean_model = clean_pydantic_model(web_search_model)

        try:
            response = await ResponsesCall(**clean_model)
            return response

        except Exception as e:
            return e

    async def generate(self, system_message, prompt, model=TextModels.latest, temperature=0, chat=True):
        self.model = model
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
        self.request_body = make_req_body(self)

        response = await Chat(self.request_body)
        return response

    def call_function(self, function_response, available_functions):
        return send_functioncall_args_to_available_functions(function_response, available_functions)

    async def structured_output(self, system_message, prompt, schema={}, input_type="json", module_name=''):
        self.model = TextModels.hipster_latest
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
            response = client.chat.completions.create(**params)

        elif (input_type == 'pydantic'):
            response = client.beta.chat.completions.parse(**params)

        return response
    except Exception as e:
        print(e)


async def Chat(params: GPT_Module_Params):

    try:
        response = client.chat.completions.create(**params)
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

        response = client.responses.create(**kwargs)
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
