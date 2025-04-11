from openai import OpenAI
import os
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
# tokenizer = tiktoken.get_encoding(encoding)


class TextModels:

    latest = "gpt-4-turbo"
    previous = "gpt-4-turbo-preview"
    previous1 = "gpt-4-1106-preview"

    legacy = "gpt-4"
    og = "gpt-4-0314"

    alpha = "gpt-4o-64k-output-alpha"

    hipster = "gpt-4o"
    hipster_latest = "gpt-4o-2024-08-06"
    hipster_mini = "gpt-4o-mini"


class EmbeddingModels:
    large = "text-embedding-3-large"
    small = "text-embedding-3-small"
    legacy = "text-embedding-ada-002"


class Models:
    text = TextModels
    moderation = "text-moderation-latest"
    embedding = EmbeddingModels
    vision = "gpt-4-vision-preview"
    images = {
        "latest": "dalle-3",
        "legacy": "dalle-2"
    }
    audio = {
        "tts": "tts-1",
        "ttshd": "tts-1-hd",
        "whisper": "whisper-1"
    }
