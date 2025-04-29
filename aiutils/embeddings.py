from openai import OpenAI

from aiutils import client, EmbeddingModels


async def get_embedding(text, model=EmbeddingModels.large, dimensions=3072, encoding_format="float"):
    response = client.embeddings.create(
        input=text,
        model=model,
        encoding_format=encoding_format,
        dimensions=dimensions
    )
    return response.data[0].embedding
