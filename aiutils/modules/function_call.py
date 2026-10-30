import json


from pydantic import Field


from ..clients import make_client, Client, ClientReply
from ..exceptions import ClientError
from ..control import check_cancel, check_non_negative_int
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
    client: Client = Field(default_factory=make_client)

    async def function_call_legacy_structured_output(
        self,
        system_msg: str,
        prompt: str,
        *,
        input_tokens: int | None = None,
        cancel=None,
        model: str = TextModels.previous,
    ) -> ClientReply:
        """Generate a tool response with reported usage.

        Args:
            system_msg: System instruction.
            prompt: User input.
            input_tokens: Input estimate required when a budget is configured.
            cancel: Optional cancellation signal.
            model: Selected text model.

        Returns:
            A reply with tool calls and reported usage.
        """
        check_cancel(cancel)
        if self.client.budget is not None or input_tokens is not None:
            check_non_negative_int(input_tokens)
        if not system_msg or not prompt:
            raise ValueError("system_msg and prompt cannot be empty.")
        self.model = TextModels(model)
        self.messages.append({"role": "system", "content": system_msg})
        self.messages.append({"role": "user", "content": prompt})
        self.reasoning_effort = "none" if self.model in REASONING_MODELS else None
        self.request_body = make_req_body(self)
        options = {}
        if input_tokens is not None:
            options["input_tokens"] = input_tokens
        if cancel is not None:
            options["cancel"] = cancel
        response = await self.client.async_chat_completions(self.request_body, **options)
        if response.error:
            raise ClientError(e=response.error) from response.error
        return response


    def call_function(self, function_response, available_funcs):
        return _send_func_call_args_to_available_funcs(
            function_response,
            available_funcs
        )

