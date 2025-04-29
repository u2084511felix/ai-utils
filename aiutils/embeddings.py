from openai import OpenAI
from aiutils import client, EmbeddingModels
import tiktoken

encoding_name = 'cl100k_base'


def truncate_to_token_limit(text, model):

    max_tokens = 8191
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    length = len(tokens)

    if length <= max_tokens:
        return text

    print(f'Chunk too large ({length}), truncating to 8191 tokens')
    truncated_tokens = tokens[:max_tokens]
    truncated_text = encoding.decode(truncated_tokens)
    return truncated_text


async def get_embedding(text, model='small', dimensions=3072, encoding_format="float"):

    truncated_text = truncate_to_token_limit(text, model)

    if model == 'small':
        model = EmbeddingModels.small
        if dimensions > 1536:
            dimensions = 1536

    if model == 'large':
        model = EmbeddingModels.large

    if model == 'legacy':
        model = EmbeddingModels.legacy
        if dimensions > 1536:
            dimensions = 1536

    response = client.embeddings.create(
        input=truncated_text,
        model=model,
        encoding_format=encoding_format,
        dimensions=dimensions
    )
    return response.data[0].embedding
