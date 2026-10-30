import logging


from pydantic import BaseModel, Field


from .exceptions import ClientError, InitializationError
from .models import (
    TextModels,
    OpenAIModels,
    GoogleModels,
    REASONING_MODELS,
    PEDANTIC_REASONING,
    Vendor
)
from .modules.gpt_module import GPTModule, make_req_body
from .clients import make_client, Client, ClientReply, InputType
from .control import Cancellation, check_cancel, check_non_negative_int
from .schemas import validate_schema, validate_model


def _check_model(model, vendor):
    try:
        model = TextModels(model)
    except ValueError:
        raise ValueError(f"Invalid model: {model}")
    if model in GoogleModels and vendor != Vendor.GOOGLE:
        raise ValueError(f"Invalid model for vendor Google: {model}")
    if model in OpenAIModels and vendor != Vendor.OPENAI:
        raise ValueError(f"Invalid model for vendor OpenAI: {model}")
    return model


class Generate(GPTModule):
    """ChatGPT-based text generation module."""
    client: Client = Field(default_factory=make_client)
    logger: logging.Logger = logging.getLogger(__name__)


    def _set_reasoning(
        self,
        reasoning_effort,
        temperature=None
    ):
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
            self.reasoning_effort = None
            self.temperature = temperature or 0


    async def generate(
        self,
        *,
        system_msg: str,
        prompt: str,
        model: str = TextModels.gpt_5_2,
        temperature: float = 0,
        chat: bool = True,
        reasoning_effort: str = "default",
        input_tokens: int | None = None,
        cancel: Cancellation | None = None
    ) -> ClientReply:
        """
        Generate a text response from an LLM.

        Args:
            system_msg: The system message to set the context.
            prompt: The user prompt to generate a response for.
            model: The model to use for generation.
            temperature: The sampling temperature.
            chat: Whether to use chat mode.
            reasoning_effort: The reasoning effort to apply.
            input_tokens: Input estimate required when a budget is configured.
            cancel: Optional cancellation signal.

        Returns:
            ClientReply: An LLM response string, wrapped in a ClientReply object.

        Raises:
            ValueError: If any of the input arguments are invalid.
            ClientError: If the client returns an error.
        """
        if not system_msg:
            raise ValueError("system_msg cannot be empty")
        if not prompt:
            raise ValueError("prompt cannot be empty")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")

        options = _call_options(self.client, input_tokens, cancel)
        model = _check_model(model, self.client.vendor)

        self.model = model
        self._set_reasoning(reasoning_effort, temperature)

        self.messages.append({"role": "system", "content": system_msg})
        self.messages.append({"role": "user", "content": prompt})

        self.request_body = make_req_body(self)

        if chat:
            response = await self.client.async_chat_completions(self.request_body, **options)
        else:
            response = await self.client.async_chat_completions_body(self.request_body, **options)
        if response.error:
            raise ClientError(e=response.error) from response.error
        return response


    async def continued_response(
        self,
        *,
        assistant_response: str,
        prompt: str,
        input_tokens: int | None = None,
        cancel: Cancellation | None = None
    ) -> ClientReply:
        """Continue an existing conversation.

        Args:
            assistant_response: Previous assistant response.
            prompt: New user prompt.
            input_tokens: Input estimate required when a budget is configured.
            cancel: Optional cancellation signal.

        Returns:
            ClientReply: An LLM response string, wrapped in a ClientReply object.

        Raises:
            ValueError: If any of the input arguments are invalid.
            ClientError: If the client returns an error.
        """
        if not assistant_response:
            raise ValueError("assistant_response cannot be empty")
        if not prompt:
            raise ValueError("prompt cannot be empty")

        options = _call_options(self.client, input_tokens, cancel)
        self.messages.append({"role": "assistant", "content": assistant_response})
        self.messages.append({"role": "user", "content": prompt})
        self.request_body = make_req_body(self)
        response = await self.client.async_chat_completions(self.request_body, **options)
        if response.error:
            raise ClientError(e=response.error) from response.error
        return response


    async def structured_output(
        self,
        *,
        system_msg: str,
        prompt: str,
        schema: dict | str | type[BaseModel] | None = None,
        input_type: str | InputType = InputType.JSON,
        module_name: str = '',
        model: str = TextModels.gpt_5_2,
        reasoning_effort: str = "default",
        assistant_response: str | None = None,
        debug: bool = False,
        input_tokens: int | None = None,
        cancel: Cancellation | None = None
    ) -> ClientReply:
        """
        Generates structured output based on the provided system message, user prompt, and schema.

        Args:
            system_msg (str): The system message to set the context for the model.
            prompt (str): The user prompt to generate a response for.
            schema: Named JSON Schema wrapper as a dictionary or JSON string, or a Pydantic model class.
            input_type (str | InputType, optional): The input type for the request body. Defaults to InputType.JSON.
            module_name (str, optional): The module name to import the schema from if using Pydantic. Defaults to ''.
            model (str, optional): The model to use for generation. Defaults to TextModels.gpt_5_2.
            reasoning_effort (str, optional): The reasoning effort level. Defaults to "default".
            assistant_response (str | None, optional): The previous assistant response to include in the context. Defaults to None.
            debug (bool, optional): Whether to enable debug logging. Defaults to False.
            input_tokens: Input estimate required when a budget is configured.
            cancel: Optional cancellation signal.

        Returns:
            ClientReply: The JSON string, wrapped in a ClientReply object.

        Raises:
            ValueError: If system_msg or prompt is empty, or if the model or input type are invalid
            TypeError: If the schema type does not match the input type.
            ClientError: If the client encounters an error during the request.

        """
        if not system_msg:
            raise ValueError("system_msg cannot be empty")
        if not prompt:
            raise ValueError("prompt cannot be empty")

        options = _call_options(self.client, input_tokens, cancel)
        model = _check_model(model, self.client.vendor)
        input_type = InputType(input_type)
        response_format = ({"type": "json_schema", "json_schema": validate_schema(schema)}
                           if input_type == InputType.JSON else validate_model(schema, module_name))
        self.model = model

        self._set_reasoning(reasoning_effort)

        if assistant_response is not None:
            msg = {"role": "assistant", "content": assistant_response}
        else:
            msg = {"role": "system", "content": system_msg}

        self.messages.append(msg)
        self.messages.append({"role": "user", "content": prompt})

        self.response_format = response_format
        self.request_body = make_req_body(self)
        response = await self.client.async_chat_completions_body(
            self.request_body, input_type=input_type, **options,
        )

        if debug == True:
            if assistant_response is not None:
                self.logger.debug("Running stateful structured output with previous assistant response in context.")
            self.logger.debug(response)
        if response.error:
            raise ClientError(e=response.error) from response.error
        return response


def make_generate(
    client: Client = make_client(),
    logger: logging.Logger = logging.getLogger(__name__),
    **kwargs
):
    """
    Factory function to create a Generate instance.

    Args:
        client (Client): The client instance to use.
        logger (logging.Logger): The logger instance to use.
        **kwargs: Additional keyword arguments to set on the Generate instance.

    Returns:
        Generate: An instance of the Generate class with the specified configuration.

    Raises:
        InitializationError: If the Generate instance could not be initialized successfully.
    """
    try:
        module = Generate(client=client, logger=logger)
        _ = [setattr(module, k, v) for k, v in kwargs.items()]
        return module
    except Exception as e:
        raise InitializationError(f"Failed to initialize Generate instance: {e}") from e


def _call_options(client, input_tokens, cancel):
    check_cancel(cancel)
    if client.budget is not None or input_tokens is not None:
        check_non_negative_int(input_tokens)
    options = {}
    if input_tokens is not None:
        options["input_tokens"] = input_tokens
    if cancel is not None:
        options["cancel"] = cancel
    return options
