from openai import OpenAI
import os
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
# tokenizer = tiktoken.get_encoding(encoding)


reasoning_models = ["o3", "o3-pro", "gpt-5", "gpt-5-mini", "gpt-5-nano", "o1", "o1-preview", "o1-mini", "o3-mini", "o4-mini"]
instruct_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4.5-preview", "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14"]

class TextModels:

    # New model names

    #Reasoning token models:
    o3 = "o3"
    o3_pro = "o3-pro"
    gpt_5 = "gpt-5"
    gpt_5_mini = "gpt-5-mini"
    gpt_5_nano = "gpt-5-nano"
    o1 = "o1"
    o1_preview = "o1-preview"
    o1_mini = "o1-mini"
    o3_mini = "o3-mini"
    o4_mini = "o4-mini"

    #Instruct models:
    gpt_3_5_turbo = "gpt-3.5-turbo"
    gpt_4 = "gpt-4"
    gpt_4_turbo = "gpt-4-turbo"
    gpt_4o = "gpt-4o"
    gpt_4o_mini = "gpt-4o-mini"
    gpt_4_5_preview = "gpt-4.5-preview"
    gpt_4_1 = "gpt-4.1-2025-04-14"
    gpt_4_1_mini = "gpt-4.1-mini-2025-04-14"
    gpt_4_1_nano = "gpt-4.1-nano-2025-04-14"




    # Legacy model names (phaseing out of all projects)
    smarter = "gpt-4.1"
    o4mini = "o4-mini"
    latest = "gpt-4o"
    latest_mini = "gpt-4o-mini"
    previous = "gpt-4-turbo-preview"
    previous1 = "gpt-4-1106-preview"
    legacy = "gpt-4"
    og = "gpt-4-0314"
    alpha = "gpt-4o-64k-output-alpha"
    turbo4 = "gpt-4-turbo"
    hipster = "gpt-4o"
    hipster_latest = "gpt-4o-2024-08-06"
    hipster_mini = "gpt-4o-mini"
    turbo35 = "gpt-3.5-turbo"
    preview45 = "gpt-4.5-preview"
    o1prevew = "o1-preview"
    o1mini = "o1-mini"
    o3mini = "o3-mini"
    o3 = "o3"
    nano41 = "gpt-4.1-nano"


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
