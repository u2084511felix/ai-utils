import tiktoken


from ..clients import make_client, Vendor
from ..models import EmbeddingModels
from ..exceptions import ClientError


ENCODING_NAME = 'cl100k_base'


def truncate_to_token_limit(
    text: str, 
    model: str,
    max_tokens: int = 8191
):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    length = len(tokens)

    if length <= max_tokens:
        return text

    print(f'Chunk too large ({length}), truncating to {max_tokens} tokens')
    truncated_text = encoding.decode(tokens[:max_tokens])
    return truncated_text


def get_token_count(text):
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    return len(encoding.encode(text))


async def get_embedding(
    text,
    client=make_client(Vendor.OPENAI), 
    model=EmbeddingModels.LARGE, 
    dimensions=3072, 
    encoding_format="float"
) -> list[float]:
    """
    Response object:
    {
      "object": "list",
      "data": [
        {
          "object": "embedding",
          "index": 0,
          "embedding": [
            -0.006929283495992422,
            -0.005336422007530928,
            -4.547132266452536e-05,
            -0.024047505110502243
          ],
        }
      ],
      "model": "text-embedding-3-small",
      "usage": {
        "prompt_tokens": 5,
        "total_tokens": 5
      }
    }

    """
    min_dimensions = 1536
    model = EmbeddingModels(model.lower())
    match model:
        case EmbeddingModels.SMALL | EmbeddingModels.LEGACY:
            if dimensions > min_dimensions:
                dimensions = min_dimensions

    truncated_text = truncate_to_token_limit(text, model)

    response = await client.make_embeddings(
        input=truncated_text,
        model=model,
        encoding_format=encoding_format,
        dimensions=dimensions
    )
    if isinstance(response, Exception):
        raise ClientError from response
    return response
