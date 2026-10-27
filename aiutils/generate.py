import importlib
import json
import logging
from dataclasses import dataclass


from pydantic import BaseModel


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


@dataclass
class Generate(GPTModule):
    """ChatGPT-based text generation module."""
    client: Client = make_client()
    logger: logging.Logger = logging.getLogger(__name__)


    def __post_init__(self):
        self.messages = []
        self.temperature = 0


    def _set_reasoning_effort_and_temperature(
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
            self.temperature = temperature or 0


    async def generate(
        self, 
        system_message: str, 
        prompt: str,
        model: str = TextModels.gpt_5_2, 
        temperature: float = 0, 
        chat: bool = True, 
        reasoning_effort: str = "default"
    ) -> ClientReply:
        """
        Generate a text response from an LLM.

        Args:
            system_message: The system message to set the context.
            prompt: The user prompt to generate a response for.
            model: The model to use for generation.
            temperature: The sampling temperature.
            chat: Whether to use chat mode.
            reasoning_effort: The reasoning effort to apply.

        Returns:
            ClientReply: An LLM response string, wrapped in a ClientReply object.

        Raises:
            ValueError: If any of the input arguments are invalid.
            ClientError: If the client returns an error.
        """
        if not system_message:
            raise ValueError("system_message cannot be empty")
        if not prompt:
            raise ValueError("prompt cannot be empty")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")

        model = _check_model(model, self.client.vendor)

        self.model = model
        self._set_reasoning_effort_and_temperature(reasoning_effort, temperature)

        self.messages.append({"role": "system", "content": system_message})
        self.messages.append({"role": "user", "content": prompt})

        self.request_body = make_req_body(self)

        if chat:
            response = await self.client.async_chat_completions(self.request_body)
        else:
            response = await self.client.async_chat_completions_body(self.request_body)
        if response.error:
            raise ClientError from response.error
        return response


    async def continued_response(
        self, 
        assistant_response: str, 
        prompt: str
    ) -> ClientReply:
        """Continue an existing conversation.

        Args:
            assistant_response: Previous assistant response.
            prompt: New user prompt.

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

        self.messages.append({"role": "assistant", "content": assistant_response})
        self.messages.append({"role": "user", "content": prompt})
        self.request_body = make_req_body(self)
        response = await self.client.async_chat_completions(self.request_body)
        if response.error:
            raise ClientError from response.error
        return response


    async def structured_output(
        self, 
        system_message: str, 
        prompt: str, 
        schema: dict | type[BaseModel] = {}, 
        input_type: str | InputType = InputType.JSON, 
        module_name: str = '', 
        model: str = TextModels.gpt_5_2, 
        reasoning_effort: str = "default", 
        assistant_response: str | None = None, 
        debug: bool = False
    ) -> ClientReply:
        """
        Generates structured output based on the provided system message, user prompt, and schema.

        Args:
            system_message (str): The system message to set the context for the model.
            prompt (str): The user prompt to generate a response for.
            schema (dict | type[BaseModel], optional): The schema to validate the structured output against. Defaults to {}.
            input_type (str | InputType, optional): The input type for the request body. Defaults to InputType.JSON.
            module_name (str, optional): The module name to import the schema from if using Pydantic. Defaults to ''.
            model (str, optional): The model to use for generation. Defaults to TextModels.gpt_5_2.
            reasoning_effort (str, optional): The reasoning effort level. Defaults to "default".
            assistant_response (str | None, optional): The previous assistant response to include in the context. Defaults to None.
            debug (bool, optional): Whether to enable debug logging. Defaults to False.

        Returns:
            ClientReply: The JSON string, wrapped in a ClientReply object.

        Raises:
            ValueError: If system_message or prompt is empty, or if the model or input type are invalid
            TypeError: If the schema type does not match the input type.
            ClientError: If the client encounters an error during the request.
            
        """
        if not system_message:
            raise ValueError("system_message cannot be empty")
        if not prompt:
            raise ValueError("prompt cannot be empty")

        model = _check_model(model, self.client.vendor)
        input_type = InputType(input_type)
        self.model = model

        self._set_reasoning_effort_and_temperature(reasoning_effort)

        if assistant_response is not None:
            msg = {"role": "assistant", "content": assistant_response}
        else:
            msg = {"role": "system", "content": system_message}

        self.messages.append(msg)
        self.messages.append({"role": "user", "content": prompt})

        match input_type:
            case InputType.JSON:
                if not isinstance(schema, str):
                    raise TypeError(f"For JSON input type, schema must be a JSON string, got {type(schema).__name__}")
                self.response_format = {
                    "type": "json_schema",
                    "json_schema": json.loads(schema)
                }
                self.request_body = make_req_body(self)
                response = await self.client.async_chat_completions_body(
                    self.request_body,
                    input_type=input_type,
                )
            case InputType.PYDANTIC:
                if module_name != '':
                    module = importlib.import_module(module_name)
                    module = importlib.reload(module)
                    schema_title = schema
                    model_class = getattr(module, schema_title)
                    self.response_format = model_class
                else:
                    self.response_format = schema

                self.request_body = make_req_body(self)
                response = await self.client.async_chat_completions_body(
                    self.request_body,
                    input_type=input_type,
                )
            case _:
                raise ValueError(f"Unsupported input_type: {input_type}")

        if debug == True:
            if assistant_response is not None:
                self.logger.debug("Running stateful structured output with previous assistant response in context.")
            self.logger.debug(response)
        if response.error:
            raise ClientError from response.error
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
