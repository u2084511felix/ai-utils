"""Provider calls that preserve token usage."""
from dataclasses import dataclass
import importlib
import json

from aiutils import (
    pedantic_resoning as pedantic_reasoning, # FIXME Felix, this is why you need spellcheckers in vscode! 
    reasoning_models
)
from aiutils.openai_config import Generate, cclient, make_req_body
from openai import AsyncOpenAI

from utils.configs import configs
from utils.budget import TokenUsage
from utils.logger import make_logger


logger = make_logger(__name__)


@dataclass(frozen=True)
class ProviderReply:
    """Provider output and its reported token usage.

    Args:
        content: Text or tool calls returned by the provider.
        usage: Token counts reported for the call.
    """
    content: object
    usage: TokenUsage


def _openai_usage(response: object) -> TokenUsage:
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


def _openai_content(response: object) -> object:
    """Read content from a provider response.

    Args:
        response: Chat completion returned by the provider SDK.

    Returns:
        Text content or tool calls.

    Raises:
        RuntimeError: The response contains neither form of output.
    """
    message = response.choices[0].message
    if message.tool_calls:
        return message.tool_calls
    if message.content is None:
        raise RuntimeError("The model provider returned no content.")
    return message.content




async def AsyncChatBody(input_type='json', **params):
    response = None
    async_client = params.pop("async_client")
    try:
        if (input_type == 'json'):
            response = cclient.client.chat.completions.create(**params)

        elif (input_type == 'pydantic'):
            response = cclient.client.chat.completions.parse(**params)

        if response is None:
            raise RuntimeError("No response was generated.")
        return response

    except Exception as e:
        return e



async def AsyncChat(**params):
    response = None
    try:
        response = cclient.client.chat.completions.create(**params)
        if response is None:
            raise RuntimeError("No response was generated.")
        response_message = response.choices[0].message
        if response_message.tool_calls:
            print("\n\nfunction call detected.\n\n")
            return (response_message.tool_calls)
        else:
            response_text = response.choices[0].message.content
            return (response_text)
    except Exception as e:
        return e


class UsageGenerate(Generate):
    """Generate model output with provider-reported usage."""

    def _settings(self, model: str, temperature: float, effort: str) -> None:
        """Set request parameters for one model call.

        Args:
            model: Model to call.
            temperature: Requested sampling temperature.
            effort: Requested reasoning effort.
        """
        self.model = model
        if model in reasoning_models:
            self.temperature = 1
            self.reasoning_effort = "none" if effort == "default" else effort
        elif model in pedantic_reasoning:
            self.temperature = 1
            self.reasoning_effort = "minimal" if effort == "default" else effort
        else:
            self.temperature = temperature

    async def generate(
        self,
        system_message,
        prompt,
        model,
        temperature=0,
        chat=True,
        reasoning_effort="default",
    ) -> ProviderReply:
        """Generate an unstructured response.

        Args:
            system_message: System instruction.
            prompt: User prompt.
            model: Model to call.
            temperature: Sampling temperature.
            chat: Must be true for chat completion calls.
            reasoning_effort: Requested reasoning effort.

        Returns:
            Provider output and reported token usage.

        Raises:
            ValueError: ``chat`` is false.
        """

        if not chat:
            raise ValueError("UsageGenerate supports chat completion calls only.")
        self._settings(model, temperature, reasoning_effort)
        if self.messages is None:
            raise RuntimeError("No messages have been initialized.")
        self.messages.extend((
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ))
        response = self._openai_completion()
        return ProviderReply(_openai_content(response), _openai_usage(response))

    def _openai_completion(self):
        return cclient.client.chat.completions.create(**make_req_body(self))

    async def _openai_async_completion(self):
        return await AsyncChatBody(make_req_body(self))

    async def continued_response(self, assistant_response, prompt) -> ProviderReply:
        """Continue an existing conversation.

        Args:
            assistant_response: Previous assistant response.
            prompt: New user prompt.

        Returns:
            Provider output and reported token usage.
        """

        self.messages.extend((
            {"role": "assistant", "content": assistant_response},
            {"role": "user", "content": prompt},
        ))
        response = self._openai_completion()
        return ProviderReply(_openai_content(response), _openai_usage(response))

    async def structured_output(
        self,
        system_message,
        prompt,
        schema=None,
        input_type="json",
        module_name="",
        model=None,
        reasoning_effort="default",
        assistance_response=None,
        debug=False,
    ) -> ProviderReply:
        """Generate a structured response.

        Args:
            system_message: System instruction.
            prompt: User prompt.
            schema: JSON or Pydantic response schema.
            input_type: Schema type, ``json`` or ``pydantic``.
            module_name: Module holding a named Pydantic schema.
            model: Model to call.
            reasoning_effort: Requested reasoning effort.
            assistance_response: Previous assistant response.
            debug: Reserved compatibility flag.

        Returns:
            Provider output and reported token usage.

        Raises:
            ValueError: The schema type is not supported.
        """
        if self.messages is None:
            raise RuntimeError("No messages have been initialized.")

        if debug:
            logger.debug("Structured output called with system_message=%s, prompt=%s, schema=%s, input_type=%s, module_name=%s, model=%s, reasoning_effort=%s, assistance_response=%s",
                         system_message, prompt, schema, input_type, module_name, model, reasoning_effort, assistance_response)

        if model is None:
            raise ValueError("A model is required for structured output.")
        self._settings(model, 0, reasoning_effort)
        role = "assistant" if assistance_response is not None else "system"
        content = assistance_response if assistance_response is not None else system_message
        self.messages.extend((
            {"role": role, "content": content},
            {"role": "user", "content": prompt},
        ))
        if input_type == "json":
            parsed_schema = json.loads(schema) if isinstance(schema, str) else schema
            if not isinstance(parsed_schema, dict):
                raise TypeError("A JSON structured-output schema must be an object.")
            if (
                not isinstance(parsed_schema.get("name"), str)
                or not parsed_schema["name"].strip()
            ):
                raise ValueError("A JSON structured-output schema requires a name.")
            if not isinstance(parsed_schema.get("schema"), dict):
                raise ValueError("A JSON structured-output schema requires an object schema.")
            self.response_format = {
                "type": "json_schema",
                "json_schema": parsed_schema,
            }
            response = self._openai_completion()
        elif input_type == "pydantic":
            if module_name:
                module = importlib.reload(importlib.import_module(module_name))
                self.response_format = getattr(module, schema)
            else:
                self.response_format = schema
            response = self._openai_completion()
        else:
            raise ValueError(f"Unsupported structured input type '{input_type}'.")
        return ProviderReply(_openai_content(response), _openai_usage(response))
