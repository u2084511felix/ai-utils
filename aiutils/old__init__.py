from openai import OpenAI
import os
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
# tokenizer = tiktoken.get_encoding(encoding)


class TextModels:

    # Turn into lowercase = (highlight code section) then: Vu
    gpt_3_5_turbo = "gpt-3.5-turbo"
    gpt_4 = "gpt-4"
    gpt_4_turbo = "gpt-4-turbo"
    gpt_4o = "gpt-4o"
    gpt_4o_mini = "gpt-4o-mini"
    gpt_4_5_preview = "gpt-4.5-preview"
    o1 = "o1"
    o1_preview = "o1-preview"
    o1_mini = "o1-mini"
    o3_mini = "o3-mini"
    gpt_4_1 = "gpt-4.1-2025-04-14"
    gpt_4_1_mini = "gpt-4.1-mini-2025-04-14"
    gpt_4_1_nano = "gpt-4.1-nano-2025-04-14"
    o4_mini = "o4-mini"
    o3 = "o3"
    o3_pro = "o3-pro"
    gpt_5 = "gpt-5"
    gpt_5_mini = "gpt-5-mini"
    gpt_5_nano = "gpt-5-nano"


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
