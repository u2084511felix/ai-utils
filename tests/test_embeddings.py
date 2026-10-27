from unittest.mock import AsyncMock, Mock, call as mock_call

import pytest

from aiutils.exceptions import ClientError
from aiutils.models import EmbeddingModels
from aiutils.modules import embeddings


@pytest.fixture
def encoding(monkeypatch):
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text: list(text)
    tokenizer.decode.side_effect = lambda tokens: "".join(tokens)
    lookup = Mock(return_value=tokenizer)
    monkeypatch.setattr(embeddings.tiktoken, "encoding_for_model", lookup)
    monkeypatch.setattr(embeddings.tiktoken, "get_encoding", lookup)
    return tokenizer


@pytest.mark.parametrize("text,limit,expected", [("abc", 3, "abc"), ("abc", 10, "abc"), ("abc", 2, "ab"), ("abc", 0, ""), ("", 3, "")])
def test_truncate_at_token_boundary(encoding, text, limit, expected):
    """GIVEN text and a model.
    WHEN the text operation is called.
    THEN the text contains no more than the specified number of tokens."""
    result = embeddings.truncate_to_token_limit(text, "test", limit)
    assert result == expected


@pytest.fixture
def embedding_request(encoding):
    client = Mock(make_embeddings=AsyncMock(return_value=[0.1, -0.2]))
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("model,requested,expected", [(EmbeddingModels.LARGE, 3072, 3072), (EmbeddingModels.SMALL, 3072, 1536), (EmbeddingModels.LEGACY, 3072, 1536), (EmbeddingModels.SMALL, 256, 256)])
async def test_embedding_returns_vector(embedding_request, model, requested, expected):
    """GIVEN text and a model.
    WHEN the text operation is called.
    THEN embedding returns vector."""
    client = embedding_request
    result = await embeddings.get_embedding("text", client=client, model=model.upper(), dimensions=requested)
    expected = [0.1, -0.2]
    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("model,requested,expected", [(EmbeddingModels.LARGE, 3072, 3072), (EmbeddingModels.SMALL, 3072, 1536), (EmbeddingModels.LEGACY, 3072, 1536), (EmbeddingModels.SMALL, 256, 256)])
async def test_embedding_passes_dimensions(embedding_request, model, requested, expected):
    """GIVEN text and a model.
    WHEN the text operation is called.
    THEN embedding passes dimensions."""
    client = embedding_request
    await embeddings.get_embedding("text", client=client, model=model.upper(), dimensions=requested)
    expected = [mock_call(input="text", model=model, encoding_format="float", dimensions=expected)]
    assert client.make_embeddings.await_args_list == expected


@pytest.fixture
def embedding_failure(encoding):
    error = RuntimeError("embedding failed")
    client = Mock(make_embeddings=AsyncMock(return_value=error))
    return (client, error)


@pytest.mark.asyncio
async def test_provider_failure_is_chained(embedding_failure):
    """GIVEN an operation that reports an error.
    WHEN the text operation is called.
    THEN the error retains the provider error as its cause."""
    client, error = embedding_failure
    with pytest.raises(ClientError) as exc:
        await embeddings.get_embedding("text", client=client)
    expected = True
    assert (exc.value.__cause__ is error) == expected


@pytest.fixture
def embedding_client():
    client = Mock(make_embeddings=AsyncMock())
    return client


@pytest.mark.asyncio
async def test_unknown_embedding_model_rejected(embedding_client):
    """GIVEN an unknown embedding model.
    WHEN the text operation is called.
    THEN the model is rejected before a provider request."""
    client = embedding_client
    with pytest.raises(ValueError):
        await embeddings.get_embedding("text", client=client, model="unknown")
    expected = []
    assert client.make_embeddings.await_args_list == expected


@pytest.fixture
def named_encoding(monkeypatch):
    monkeypatch.setattr(embeddings.tiktoken, "get_encoding", lambda name: Mock(encode=lambda text: list(text)))


def test_token_count_uses_named_encoding(named_encoding):
    """GIVEN text and a model.
    WHEN the text operation is called.
    THEN the named encoding gives the expected token count."""
    result = embeddings.get_token_count("abc")
    expected = 3
    assert result == expected


@pytest.mark.parametrize("text,expected", [("", 0), ("abc", 3), ("café", 4)])
def test_token_count_counts_encoder_output(encoding, text, expected):
    """GIVEN text and a model.
    WHEN the text operation is called.
    THEN the token count matches the encoded text."""
    result = embeddings.get_token_count(text)
    assert result == expected
