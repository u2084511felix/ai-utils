"""Run paid OpenAI contract checks only when explicitly enabled."""

import pytest
from pydantic import BaseModel

from aiutils import TokenUsage


class Result(BaseModel):
    answer: str


@pytest.fixture
def structured_options(live_model):
    return dict(system_msg="Return a short answer.", prompt="Say OK.", model=live_model,
                input_type="pydantic", schema=Result, input_tokens=128)


@pytest.fixture
def tool_params(chat_params):
    return dict(**chat_params, tools=[{"type": "function", "function": {
        "name": "reply", "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    }}], tool_choice={"type": "function", "function": {"name": "reply"}})


def test_sync_chat_reports_usage(live_client, chat_params):
    """GIVEN live credentials. WHEN chat is requested. THEN the reply has reported usage."""
    reply = live_client.chat_completions(chat_params, input_tokens=128)
    expected = TokenUsage
    assert isinstance(reply.usage, expected)


@pytest.mark.asyncio
async def test_async_chat_reports_usage(async_live, chat_params):
    """GIVEN live credentials. WHEN async chat is requested. THEN usage is reported."""
    reply = await async_live.async_chat_completions(chat_params, input_tokens=128)
    expected = TokenUsage
    assert isinstance(reply.usage, expected)


@pytest.mark.asyncio
async def test_responses_reports_usage(async_live, live_model):
    """GIVEN live credentials. WHEN Responses is requested. THEN usage is reported."""
    reply = await async_live.async_responses_create("Reply with OK.", model=live_model, max_output_tokens=256, input_tokens=128)
    expected = TokenUsage
    assert isinstance(reply.usage, expected)


@pytest.mark.asyncio
async def test_pydantic_output_is_parsed(structured_gen, structured_options):
    """GIVEN a supported model. WHEN structured output is requested. THEN it is parsed."""
    reply = await structured_gen.structured_output(**structured_options)
    expected = Result
    assert isinstance(reply.parsed, expected)


@pytest.mark.asyncio
async def test_tool_call_reports_usage(async_live, tool_params):
    """GIVEN a forced tool. WHEN chat is requested. THEN usage is reported."""
    reply = await async_live.async_chat_completions(tool_params, input_tokens=128)
    expected = TokenUsage
    assert isinstance(reply.usage, expected)
