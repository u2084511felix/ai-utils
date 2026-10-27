"""Client instantiation utilities for OpenAI and other LLM clients."""
import logging
import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias


from openai import OpenAI, AsyncOpenAI, APIResponse, AsyncAPIResponse


from .budget import TokenUsage
from .exceptions import InitializationError
from .models import Vendor



class InputType(StrEnum):
    JSON = "json"
    PYDANTIC = "pydantic"


ClientType: TypeAlias = type[OpenAI] | type[AsyncOpenAI]
ClientInstance: TypeAlias = OpenAI | AsyncOpenAI


class EnvNames(StrEnum):
    GEMINI_API_KEY = "GEMINI_API_KEY"
    BRAVE_API_KEY = "BRAVE_API_KEY"
    OPENAI_API_KEY = "OPENAI_API_KEY"


def _make_client_type(
    client_type: ClientType, 
    *, 
    api_key: str,
    base_url: str | None = None
) -> ClientInstance:
    if not api_key:
        raise InitializationError(f"api_key must be provided.")

    try:
        client = client_type(api_key=api_key, base_url=base_url)
        return client
    except Exception as e:
        raise InitializationError(f"Failed to instantiate client type: {e}") from e


def _client_type_kwargs(vendor: Vendor) -> dict:
    match vendor:
        case Vendor.OPENAI:
            api_key = EnvNames.OPENAI_API_KEY
            base_url = None
        case Vendor.GOOGLE:
            api_key = EnvNames.GEMINI_API_KEY
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        case Vendor.BRAVE:
            api_key = EnvNames.BRAVE_API_KEY
            base_url = "https://api.search.brave.com/res/v1"
        case _:
            raise ValueError(f"Unsupported vendor: {vendor}")
    api_key = os.getenv(api_key)
    if not api_key:
        raise EnvironmentError(f"API key for {vendor} not found in environment.")
    return {
        "api_key": api_key,
        "base_url": base_url
    }


def make_google_client() -> ClientInstance:
    return _make_client_type(OpenAI, **_client_type_kwargs(Vendor.GOOGLE))


def make_brave_client() -> ClientInstance:
    return _make_client_type(OpenAI, **_client_type_kwargs(Vendor.BRAVE))


def make_openai_client() -> ClientInstance:
    return _make_client_type(OpenAI, **_client_type_kwargs(Vendor.OPENAI))


def make_async_openai_client() -> ClientInstance:
    return _make_client_type(AsyncOpenAI, **_client_type_kwargs(Vendor.OPENAI))


@dataclass(frozen=True)
class ClientReply:
    """Provider output and its reported token usage.

    Args:
        content: Text or tool calls returned by the provider.
        usage: Token counts reported for the call.
    """
    content: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    error: Exception | None = None
    response: APIResponse | AsyncAPIResponse | None = None


def _get_usage(response: APIResponse | AsyncAPIResponse) -> TokenUsage:
    """Read token usage from a provider response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        raise RuntimeError("The model provider did not report token usage.")
    is_responses = hasattr(usage, "input_tokens")
    details = getattr(usage, "output_tokens_details" if is_responses else "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", 0) if details is not None else 0
    return TokenUsage(
        input_tokens=int(usage.input_tokens if is_responses else usage.prompt_tokens),
        output_tokens=int(usage.output_tokens if is_responses else usage.completion_tokens),
        reasoning_tokens=int(reasoning or 0),
    )


def _parse_response(response):
    content = ""
    error = None
    match response:
        case None:
            error = RuntimeError("No response received from the model.")
            return ClientReply(error=error)
        case Exception():
            return ClientReply(error=response)
        # case APIResponse() | AsyncAPIResponse():
        #     pass
        # case _:
        #     error = RuntimeError(f"Unexpected response type, {type(response).__name__}.")
        #     return ClientReply(error=error)

    if hasattr(response, "output"):
        return ClientReply(
            content=getattr(response, "output_text", ""),
            usage=_get_usage(response),
            response=response,
        )

    choices = getattr(response, "choices", None)
    if not choices:
        error = RuntimeError("No choices returned from the model.")
        return ClientReply(error=error, response=response)

    response_msg = choices[0].message
    if response_msg.tool_calls:
        print("\n\nfunction call detected.\n\n")
        content = response_msg.tool_calls
    else:
        if response_msg.content is None:
            error = RuntimeError("The model provider returned no content.")
        else:
            content = response_msg.content

    return ClientReply(
        content=content, 
        usage=_get_usage(response),
        error=error,
        response=response,
    )




@dataclass
class Client:
    async_client: ClientInstance
    client: ClientInstance
    vendor: Vendor = Vendor.OPENAI
    logger: logging.Logger = logging.getLogger(__name__)


    def set_vendor(self, vendor: str):
        self.vendor = Vendor(vendor.lower())
        match self.vendor:
            case Vendor.OPENAI:
                self.client = make_openai_client()
                self.async_client = make_async_openai_client()
            case Vendor.GOOGLE:
                self.client = make_google_client()
                # TODO
                self.async_client = make_async_openai_client()
            case Vendor.BRAVE:
                self.client = make_brave_client()
                # TODO
                self.async_client = make_async_openai_client()
            case _:
                raise ValueError(f"Unsupported vendor: {vendor}")
        if self.client is None:
            raise RuntimeError(f"Sync client not available for {vendor}.")
        if self.async_client is None:
            raise RuntimeError(f"Async client not available for {vendor}.")


    def responses_create(self, prompt=None, *, model, tools=None, verbose=False, **kwargs):
        if "input" in kwargs:
            if prompt is not None:
                raise TypeError("Supply either prompt or input, not both.")
            prompt = kwargs.pop("input")
        if tools is not None:
            kwargs["tools"] = tools
        response = None
        try:
            response = self.client.responses.create(
                input=prompt,
                model=model,
                **kwargs
            )
            if verbose:
                self.logger.debug(response.model_dump())
        except Exception as e:
            response = e
        return _parse_response(response)


    async def async_responses_create(self, prompt=None, *, model, tools=None, verbose=False, **kwargs):
        if "input" in kwargs:
            if prompt is not None:
                raise TypeError("Supply either prompt or input, not both.")
            prompt = kwargs.pop("input")
        if tools is not None:
            kwargs["tools"] = tools
        response = None
        try:
            response = await self.async_client.responses.create(
                input=prompt,
                model=model,
                **kwargs
            )
            if verbose:
                self.logger.debug(response.model_dump())
        except Exception as e:
            response = e
        return _parse_response(response)


    def chat_completions_body(
        self, 
        params: dict, 
        verbose: bool = False, 
        input_type: str | InputType = InputType.JSON
    ) -> ClientReply:
        response = None
        comp = self.client.chat.completions
        try:
            match input_type:
                case InputType.JSON:     response = comp.create(**params)
                case InputType.PYDANTIC: response = comp.parse(**params)
                case _:                  
                    raise ValueError(f"Unsupported input_type: {input_type}")
        except Exception as e:
            response = e
        if verbose:
            self.logger.debug(response)
        return _parse_response(response)


    async def async_chat_completions_body(
        self, 
        params: dict, 
        verbose: bool = False, 
        input_type: str | InputType = InputType.JSON
    ) -> ClientReply:
        response = None
        comp = self.async_client.chat.completions
        try:
            input_type = InputType(input_type)
            match input_type:
                case InputType.JSON:     response = await comp.create(**params)
                case InputType.PYDANTIC: response = await comp.parse(**params)
                case _:         
                    raise ValueError(f"Unsupported input_type: {input_type}")
        except Exception as e:
            response = e
        if verbose:
            self.logger.debug(response)
        return _parse_response(response)


    def chat_completions(
        self, 
        params: dict, 
        verbose: bool = False,
    ) -> ClientReply:
        response = None
        try:
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            response = e
        if verbose:
            self.logger.debug(response)
        return _parse_response(response)


    async def async_chat_completions(
        self, 
        params: dict, 
        verbose: bool = False, 
    ) -> ClientReply:
        response = None
        try:
            response = await self.async_client.chat.completions.create(**params)
        except Exception as e:
            response = e
        if verbose:
            self.logger.debug(response)
        return _parse_response(response)


    async def make_embeddings(
        self,
        input: str,
        model: str,
        encoding_format: str,
        dimensions: int
    ) -> list[float] | Exception:
        try:
            response = await self.async_client.embeddings.create(
                input=input,
                model=model,
                encoding_format=encoding_format,
                dimensions=dimensions
            )
        except Exception as e:
            return e
        else:
            return response.data[0].embedding


def make_client(
    vendor: str | Vendor = Vendor.OPENAI,
    client: ClientInstance = make_openai_client(),
    async_client: ClientInstance = make_async_openai_client(),
    logger: logging.Logger = logging.getLogger(__name__)
):
    vendor = Vendor(vendor.lower())

    try:
        return Client(
            client=client,
            async_client=async_client,
            vendor=vendor,
            logger=logger
        )
    except RuntimeError as e:
        raise InitializationError(f"Failed to make client for vendor {vendor}: {e}") from e
