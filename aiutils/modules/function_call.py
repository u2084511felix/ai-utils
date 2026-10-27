

from dataclasses import dataclass, field
import json


from pydantic import BaseModel


from ..clients import make_client, Client, ClientReply
from ..exceptions import ClientError
from .gpt_module import GPTModule, make_req_body
from ..models import TextModels, REASONING_MODELS



def _send_func_call_args_to_available_funcs(
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



class FunctionCall(GPTModule):
    client: Client = make_client()

    async def function_call_legacy_structured_output(
        self, 
        system_message: str, 
        prompt: str
    ) -> ClientReply:
        self.model = TextModels.previous
        self.messages.append({"role": "system", "content": system_message})
        self.messages.append({"role": "user", "content": prompt})
        if self.model in REASONING_MODELS:
            self.reasoning_effort = "none"
        self.request_body = make_req_body(self)
        response = await self.client.async_chat_completions(self.request_body)
        if response.error:
            raise ClientError from response.error
        return response


    def call_function(self, function_response, available_funcs):
        return _send_func_call_args_to_available_funcs(
            function_response, 
            available_funcs
        )

    