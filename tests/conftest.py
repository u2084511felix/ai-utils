"""Supply test clients and prevent network requests."""

import socket
from types import SimpleNamespace as SN
from unittest.mock import AsyncMock, Mock

import pytest
from openai import OpenAI, AsyncOpenAI


from aiutils.clients import Client


_environment = pytest.MonkeyPatch()


def pytest_configure(config):
    # Set test keys before the package modules are imported.
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "BRAVE_API_KEY"):
        _environment.setenv(name, "pytest-not-a-real-key")


def pytest_unconfigure(config):
    _environment.undo()


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Tests must not make network requests")
    monkeypatch.setattr(socket.socket, "connect", fail)
    monkeypatch.setattr(socket.socket, "connect_ex", fail)


@pytest.fixture
def default_content():
    return "hello"

@pytest.fixture
def default_reasoning():
    return 2

@pytest.fixture
def default_tool_calls():
    return None

@pytest.fixture
def default_prompt_tokens():
    return 10

@pytest.fixture
def default_completion_tokens():
    return 5

@pytest.fixture
def default_embedding():
    return [0.1, 0.2]


@pytest.fixture
def response_factory(
    default_content,
    default_tool_calls,
    default_reasoning,
    default_prompt_tokens,
    default_completion_tokens,
):
    def make(
        content=default_content, 
        *, 
        tool_calls=default_tool_calls, 
        reasoning=default_reasoning,
    ):
        msg = SN(content=content, tool_calls=tool_calls)
        return SN(
            choices=[SN(message=msg)],
            usage=SN(
                prompt_tokens=default_prompt_tokens,
                completion_tokens=default_completion_tokens,
                completion_tokens_details=SN(reasoning_tokens=reasoning)
            )
        )
    return make


@pytest.fixture
def mock_sync_client(response_factory):
    reply = response_factory()
    client = Mock(spec=OpenAI)
    client.max_retries = 0
    client.chat = Mock()
    client.chat.completions = Mock()
    client.responses = Mock()
    client.chat.completions.create.return_value = reply
    client.chat.completions.parse.return_value = reply
    return client


@pytest.fixture
def mock_async_client(response_factory, default_embedding):
    reply = AsyncMock(return_value=response_factory())
    async_client = Mock(spec=AsyncOpenAI)
    async_client.max_retries = 0
    async_client.chat = Mock()
    async_client.chat.completions = Mock()
    async_client.chat.completions.create = reply
    async_client.chat.completions.parse = reply
    async_client.responses = Mock()
    async_client.responses.create = reply
    async_client.embeddings = Mock()
    async_client.embeddings.create = AsyncMock(
        return_value=SN(
            data=[SN(embedding=default_embedding)], 
            usage=SN(prompt_tokens=10, total_tokens=10)
        )
    )
    return async_client


@pytest.fixture
def client(mock_sync_client, mock_async_client):
    return Client(
        client=mock_sync_client, 
        async_client=mock_async_client
    )
