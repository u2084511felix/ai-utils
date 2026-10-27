from unittest.mock import AsyncMock, Mock

import pytest

from aiutils.exceptions import ClientError
from aiutils.models import EmbeddingModels
from aiutils.modules import embeddings


@pytest.fixture
def encoding(monkeypatch):
    # Character-level tokenizer keeps these tests deterministic and avoids
    # tiktoken downloading vocabulary files on a fresh machine.
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text: list(text)
    tokenizer.decode.side_effect = lambda tokens: "".join(tokens)
    lookup = Mock(return_value=tokenizer)
    monkeypatch.setattr(embeddings.tiktoken, "encoding_for_model", lookup)
    return tokenizer


@pytest.mark.parametrize("text,limit,expected", [("abc", 3, "abc"), ("abc", 10, "abc"), ("abc", 2, "ab"), ("abc", 0, ""), ("", 3, "")])
def test_truncate_at_token_boundary(encoding, text, limit, expected):
    assert embeddings.truncate_to_token_limit(text, "test", limit) == expected
    encoding.encode.assert_called_once_with(text)


@pytest.mark.asyncio
@pytest.mark.parametrize("model,requested,expected", [
    (EmbeddingModels.LARGE, 3072, 3072), (EmbeddingModels.SMALL, 3072, 1536),
    (EmbeddingModels.LEGACY, 3072, 1536), (EmbeddingModels.SMALL, 256, 256),
])
async def test_embedding_dimensions_and_parameters(encoding, model, requested, expected):
    client = Mock(make_embeddings=AsyncMock(return_value=[0.1, -0.2]))
    result = await embeddings.get_embedding("text", client=client, model=model.upper(), dimensions=requested)
    assert result == [0.1, -0.2]
    client.make_embeddings.assert_awaited_once_with(input="text", model=model, encoding_format="float", dimensions=expected)


@pytest.mark.asyncio
async def test_provider_failure_is_chained(encoding):
    error = RuntimeError("embedding failed")
    client = Mock(make_embeddings=AsyncMock(return_value=error))
    with pytest.raises(ClientError) as exc:
        await embeddings.get_embedding("text", client=client)
    assert exc.value.__cause__ is error


@pytest.mark.asyncio
async def test_unknown_embedding_model_rejected():
    client = Mock(make_embeddings=AsyncMock())
    with pytest.raises(ValueError):
        await embeddings.get_embedding("text", client=client, model="unknown")
    client.make_embeddings.assert_not_awaited()


@pytest.mark.xfail(strict=True, raises=KeyError, reason="get_token_count passes an encoding name to encoding_for_model")
def test_token_count_uses_named_encoding(monkeypatch):
    def encoding_for_model(model):
        raise KeyError(model)
    monkeypatch.setattr(embeddings.tiktoken, "encoding_for_model", encoding_for_model)
    monkeypatch.setattr(embeddings.tiktoken, "get_encoding", lambda name: Mock(encode=lambda text: list(text)))
    assert embeddings.get_token_count("abc") == 3
