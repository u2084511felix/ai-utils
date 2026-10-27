"""Supply test clients and prevent network requests."""

import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

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
def response_factory():
    def make(content="hello", *, tool_calls=None, reasoning=2):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=content,
                        tool_calls=tool_calls,
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
            ),
        )

    return make


@pytest.fixture
def client(response_factory):
    from openai import AsyncOpenAI, OpenAI
    from aiutils.clients import Client

    sync = Mock(spec=OpenAI)
    sync.chat = SimpleNamespace(completions=Mock())
    sync.responses = Mock()
    sync.chat.completions.create.return_value = response_factory()
    sync.chat.completions.parse.return_value = response_factory()
    async_sdk = Mock(spec=AsyncOpenAI)
    async_sdk.chat = SimpleNamespace(
        completions=SimpleNamespace(
            create=AsyncMock(return_value=response_factory()),
            parse=AsyncMock(return_value=response_factory()),
        )
    )
    async_sdk.responses = SimpleNamespace(create=AsyncMock(return_value=response_factory()))
    async_sdk.embeddings = SimpleNamespace(
        create=AsyncMock(
            return_value=SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.1, 0.2])],
            )
        )
    )
    return Client(client=sync, async_client=async_sdk)
