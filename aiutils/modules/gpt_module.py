from dataclasses import dataclass, field
from typing import Any


from pydantic import BaseModel, Field


class GPTModule(BaseModel):
    request_body: dict[str, Any] | None = Field(default_factory=dict)
    model: str | None = None
    reasoning_effort: str | None = None
    messages: list = Field(default_factory=list)
    token_usage: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list | None = Field(default_factory=list)
    top_p: float | None = None
    logit_bias: dict[str, float] | None = Field(default_factory=dict)
    logprobs: int | None = None
    top_logprobs: int | None = None
    n: int | None = None
    response_format: Any | None = Field(default_factory=list)
    seed: int | None = None
    stream: bool | None = None
    tools: list | None = Field(default_factory=list)
    tool_choice: dict[str, Any] | None = Field(default_factory=dict)
    user: str | None = None


def make_req_body(module: GPTModule) -> dict[str, Any]:
    ignored_attrs = [
        'request_body',
        'client',
        'logger',
    ]
    request_body = {}
    for attr in module.__dict__:
        value = getattr(module, attr)
        if attr.startswith('_'):
            continue
        if attr in ignored_attrs or value is None:
            continue
        if value == [] or value == {} or value == "":
            continue
        request_body[attr] = value
    return request_body

