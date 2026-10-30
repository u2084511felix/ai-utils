"""Provider clients, replies, retries, and usage accounting."""

import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from inspect import signature
from typing import Any, Callable, TypeAlias
from types import SimpleNamespace

from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, SkipValidation

from .budget import SpendBudget, TokenUsage
from .exceptions import BudgetExceededError, CallCancelledError, InitializationError
from .models import Vendor
from .retries import make_request_runner, RequestRunner, RetryPolicy, make_retry_policy
from .schemas import validate_model, validate_schema


class InputType(StrEnum):
    JSON = "json"
    PYDANTIC = "pydantic"


ClientType: TypeAlias = type[OpenAI] | type[AsyncOpenAI]
ClientInstance: TypeAlias = OpenAI | AsyncOpenAI


class EnvNames(StrEnum):
    GEMINI_API_KEY = "GEMINI_API_KEY"
    BRAVE_API_KEY = "BRAVE_API_KEY"
    OPENAI_API_KEY = "OPENAI_API_KEY"


def _make_sdk(client_type, *, api_key, base_url=None):
    if not api_key:
        raise InitializationError("api_key must be provided.")
    try:
        return client_type(api_key=api_key, base_url=base_url, max_retries=0)
    except Exception as error:
        raise InitializationError(f"Failed to instantiate client type: {error}") from error


def _vendor_kwargs(vendor):
    match vendor:
        case Vendor.OPENAI:
            key, url = EnvNames.OPENAI_API_KEY, None
        case Vendor.GOOGLE:
            key, url = EnvNames.GEMINI_API_KEY, "https://generativelanguage.googleapis.com/v1beta/openai/"
        case Vendor.BRAVE:
            key, url = EnvNames.BRAVE_API_KEY, "https://api.search.brave.com/res/v1"
        case _:
            raise ValueError(f"Unsupported vendor: {vendor}")
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(f"API key for {vendor} not found in environment.")
    return {"api_key": value, "base_url": url}


def make_google_client() -> ClientInstance:
    """Create a synchronous Google transport with SDK retries disabled."""
    return _make_sdk(OpenAI, **_vendor_kwargs(Vendor.GOOGLE))


def make_brave_client() -> ClientInstance:
    """Create a synchronous Brave transport with SDK retries disabled."""
    return _make_sdk(OpenAI, **_vendor_kwargs(Vendor.BRAVE))


def make_openai_client() -> ClientInstance:
    """Create a synchronous OpenAI transport with SDK retries disabled."""
    return _make_sdk(OpenAI, **_vendor_kwargs(Vendor.OPENAI))


def make_async_openai_client() -> ClientInstance:
    """Create an asynchronous OpenAI transport with SDK retries disabled."""
    return _make_sdk(AsyncOpenAI, **_vendor_kwargs(Vendor.OPENAI))


@dataclass(frozen=True)
class ClientReply:
    """Provider output and explicitly reported usage.

    Attributes:
        content: Text or tool calls returned by the provider.
        usage: Reported counts. Required for successful replies.
        error: Original failure for an unsuccessful reply.
        response: Complete provider response.
        parsed: Pydantic output when the parsing API is used.
    """

    content: Any = ""
    usage: TokenUsage | None = None
    error: Exception | None = None
    response: Any = None
    parsed: BaseModel | None = None

    def __post_init__(self):
        if self.error is None and not isinstance(self.usage, TokenUsage):
            raise ValueError("Successful replies require explicit TokenUsage.")

_USAGE_ATTRS = (
    "input_tokens_details", 
    "prompt_tokens_details", 
    "output_tokens_details", 
    "completion_tokens_details"
)

def _get_usage(response) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if isinstance(response, Exception):
        body = getattr(response, "body", None)
        if usage is None and isinstance(body, dict):
            usage = body.get("usage")
        http_response = getattr(response, "response", None)
        if usage is None and http_response is not None:
            try:
                body = http_response.json()
            except ValueError:
                body = None
            if isinstance(body, dict):
                usage = body.get("usage")
        if usage is None:
            return None
    if isinstance(usage, TokenUsage):
        return usage
    if isinstance(usage, dict):
        usage = SimpleNamespace(**usage)
        for name in _USAGE_ATTRS:
            details = getattr(usage, name, None)
            if isinstance(details, dict):
                setattr(usage, name, SimpleNamespace(**details))
    if usage is None:
        raise RuntimeError("The model provider did not report token usage.")
    responses = hasattr(usage, "input_tokens")
    embedding = hasattr(response, "data")
    details = getattr(usage, "output_tokens_details" if responses else "completion_tokens_details", None)
    input_details = getattr(usage, "input_tokens_details" if responses else "prompt_tokens_details", None)
    return TokenUsage(
        input_tokens=usage.input_tokens if responses else usage.prompt_tokens,
        output_tokens=0 if embedding else usage.output_tokens if responses else usage.completion_tokens,
        cached_input_tokens=(getattr(input_details, "cached_tokens", 0) or 0) if input_details is not None else 0,
        reasoning_tokens=(getattr(details, "reasoning_tokens", 0) or 0) if details is not None else 0,
    )


def _parse_response(response):
    if response is None:
        return ClientReply(error=RuntimeError("No response received from the model."))
    if isinstance(response, Exception):
        return ClientReply(error=response)
    if hasattr(response, "output"):
        content = getattr(response, "output_text", "")
        if not content:
            content = [item for item in response.output if getattr(item, "type", None) == "function_call"] or ""
        return ClientReply(content=content, usage=_get_usage(response), response=response,
                           parsed=getattr(response, "output_parsed", None))
    choices = getattr(response, "choices", None)
    if not choices:
        return ClientReply(error=RuntimeError("No choices returned from the model."), response=response)
    message = choices[0].message
    content = message.tool_calls or message.content
    error = RuntimeError("The model provider returned no content.") if content is None else None
    return ClientReply(content=content or "", usage=_get_usage(response), error=error,
                       response=response, parsed=getattr(message, "parsed", None))


@dataclass
class Client:
    """Apply retry and budget contracts to injected SDK transports.

    Attributes:
        async_client: Asynchronous SDK client with max_retries set to zero.
        client: Synchronous SDK client with max_retries set to zero.
        vendor: Provider for the injected pair.
        logger: Request logger.
        retry_policy: Library retry limits.
        budget: Optional shared budget.
        runner_func: Factory for request runners.
    """
    async_client: ClientInstance
    client: ClientInstance
    vendor: Vendor = Vendor.OPENAI
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    retry_policy: SkipValidation[RetryPolicy] = field(default_factory=make_retry_policy)
    budget: SkipValidation[SpendBudget] | None = None
    runner_func: Callable[..., RequestRunner] = make_request_runner


    def set_vendor(self, vendor: str):
        """Select a provider and replace this client's transports.

        Args:
            vendor: Provider name.

        Raises:
            ValueError: The provider name is unsupported.
        """
        selected = Vendor(vendor.lower())
        match selected:
            case Vendor.OPENAI:
                sync = make_openai_client()
            case Vendor.GOOGLE:
                sync = make_google_client()
            case Vendor.BRAVE:
                sync = make_brave_client()
        asynchronous = make_async_openai_client()
        if sync is None:
            raise RuntimeError(f"Sync client not available for {selected}.")
        if asynchronous is None:
            raise RuntimeError(f"Async client not available for {selected}.")
        self.client, self.async_client, self.vendor = sync, asynchronous, selected

    def _runner(self):
        runner = self.runner_func(
            policy=self.retry_policy,
            budget=self.budget,
        )
        return runner


    def _request(self, operation, params, input_tokens, cancel, verbose):
        params = deepcopy(params)
        _validate_request(operation, params)
        runner = self._runner()
        runner.validate(input_tokens, cancel)
        try:
            response = runner.run_sync(
                lambda: operation(**params), 
                model=params.get("model"),
                input_tokens=input_tokens, 
                cancel=cancel, 
                usage=_get_usage
            )
        except (BudgetExceededError, CallCancelledError):
            raise
        except Exception as error:
            return ClientReply(error=error)
        if verbose:
            self.logger.debug(response.model_dump() if "input" in params else response)
        return _parse_response(response)

    async def _arequest(self, operation, params, input_tokens, cancel, verbose):
        params = deepcopy(params)
        _validate_request(operation, params)
        runner = self._runner()
        runner.validate(input_tokens, cancel)
        try:
            response = await runner.run(
                lambda: operation(**params), 
                model=params.get("model"),
                input_tokens=input_tokens, 
                cancel=cancel, 
                usage=_get_usage
            )
        except (BudgetExceededError, CallCancelledError):
            raise
        except Exception as error:
            return ClientReply(error=error)
        if verbose:
            self.logger.debug(response.model_dump() if "input" in params else response)
        return _parse_response(response)


    def responses_create(self, prompt=None, *, model, tools=None, verbose=False,
                         input_tokens=None, cancel=None, **kwargs):
        """Request a Responses reply with optional accounting and cancellation."""
        params = _response_params(prompt, model, tools, kwargs)
        return self._request(self.client.responses.create, params, input_tokens, cancel, verbose)

    async def async_responses_create(self, prompt=None, *, model, tools=None, verbose=False,
                                     input_tokens=None, cancel=None, **kwargs):
        """Request an asynchronous Responses reply."""
        params = _response_params(prompt, model, tools, kwargs)
        return await self._arequest(self.async_client.responses.create, params, input_tokens, cancel, verbose)

    def chat_completions_body(self, params, verbose=False, input_type=InputType.JSON, *,
                              input_tokens=None, cancel=None):
        """Request chat output through the create or parsing API."""
        kind = InputType(input_type)
        if kind == InputType.PYDANTIC:
            _validate_parse(params)
        operation = self.client.chat.completions.parse if kind == InputType.PYDANTIC else self.client.chat.completions.create
        return self._request(operation, params, input_tokens, cancel, verbose)

    async def async_chat_completions_body(self, params, verbose=False, input_type=InputType.JSON, *,
                                          input_tokens=None, cancel=None):
        """Request asynchronous chat output through the create or parsing API."""
        kind = InputType(input_type)
        if kind == InputType.PYDANTIC:
            _validate_parse(params)
            op = self.async_client.chat.completions.parse
        else:
            op = self.async_client.chat.completions.create
        return await self._arequest(op, params, input_tokens, cancel, verbose)

    def chat_completions(self, params, verbose=False, *, input_tokens=None, cancel=None):
        """Request chat output with per-attempt accounting."""
        return self._request(self.client.chat.completions.create, params, input_tokens, cancel, verbose)

    async def async_chat_completions(self, params, verbose=False, *, input_tokens=None, cancel=None):
        """Request asynchronous chat output with per-attempt accounting."""
        return await self._arequest(
            self.async_client.chat.completions.create, 
            params, input_tokens, cancel, verbose
        )

    async def make_embeddings(self, input, model, encoding_format, dimensions, *, input_tokens=None, cancel=None):
        """Request an embedding with the client retry policy."""
        params = dict(input=input, model=model, encoding_format=encoding_format, dimensions=dimensions)
        runner = self._runner()
        runner.validate(input_tokens, cancel)
        try:
            response = await runner.run(lambda: self.async_client.embeddings.create(**params), model=model,
                                        input_tokens=input_tokens, cancel=cancel, usage=_get_usage)
        except (BudgetExceededError, CallCancelledError):
            raise
        except Exception as error:
            return error
        _get_usage(response)
        return response.data[0].embedding


def _response_params(prompt, model, tools, kwargs):
    if "input" in kwargs:
        if prompt is not None:
            raise TypeError("Supply either prompt or input, not both.")
        prompt = kwargs.pop("input")
    if tools is not None:
        kwargs["tools"] = tools
    return dict(input=prompt, model=model, **kwargs)


def _validate_request(operation, params):
    signature(operation).bind(**params)
    response_format = params.get("response_format")
    if isinstance(response_format, type):
        validate_model(response_format)
    elif isinstance(response_format, dict) and response_format.get("type") == "json_schema":
        validate_schema(response_format.get("json_schema"))


def _validate_parse(params):
    validate_model(params.get("response_format"))
    for tool in params.get("tools", []):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError("Pydantic parsing requires strict function tools.")
        function = tool.get("function")
        if not isinstance(function, dict) or function.get("strict") is not True:
            raise ValueError("Pydantic parsing requires strict function tools.")


def make_client(
    vendor: str | Vendor = Vendor.OPENAI, 
    client: ClientInstance = make_openai_client(),
    async_client: ClientInstance = make_async_openai_client(),
    logger: logging.Logger = logging.getLogger(__name__), 
    runner_func = make_request_runner,
    retry_policy: dict | RetryPolicy = make_retry_policy(),
    budget: SpendBudget | None = None, 
) -> Client:
    """Create a client with one retry owner and an optional shared budget.

    Args:
        vendor: Provider for the SDK transport pair.
        client: Explicit synchronous SDK client.
        async_client: Explicit asynchronous SDK client.
        logger: Request logger.
        retry_policy: Retry limits after the initial attempt.
        budget: Shared capacity and spend budget.
        runner_func: Factory for request runners.

    Returns:
        The configured client.

    Raises:
        ValueError: Injected transports permit SDK retries.
    """
    for c in (client, async_client):
        if c is None:
            raise ValueError(f"Both SDK transports are required, but {c} is None.")
        if type(getattr(c, "max_retries", None)) is not int or c.max_retries != 0:
            raise ValueError("Injected SDK clients must have max_retries=0.")

    if isinstance(retry_policy, dict):
        try:
            retry_policy = make_retry_policy(**retry_policy)
        except Exception as e:
            raise ValueError(f"Failed to create retry policy from dictionary: {e}") from e

    vendor = Vendor(vendor.lower())
    try:
        return Client(
            client=client,
            async_client=async_client,
            vendor=vendor, 
            logger=logger,
            retry_policy=retry_policy,
            budget=budget, 
            runner_func=runner_func,
        )
    except RuntimeError as error:
        raise InitializationError(f"Failed to make client for vendor {vendor}: {error}") from error
