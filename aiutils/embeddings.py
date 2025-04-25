from openai import OpenAI

from aiutils import client, EmbeddingModels


def get_embedding(text, model=EmbeddingModels.large):
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding
