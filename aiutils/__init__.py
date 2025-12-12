from openai import OpenAI
import os
#from google import genai

google_client  = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# response = client.models.generate_content(
#     model="gemini-2.5-flash", contents="Explain how AI works in a few words"
# )
# print(response.text)


brave_client = OpenAI(
    api_key=os.getenv("BRAVE_API_KEY"),
    base_url="https://api.search.brave.com/res/v1",
)


# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
# tokenizer = tiktoken.get_encoding(encoding)

pedantic_resoning = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gemini_25_pro",  "gemini-3-pro-preview"]
reasoning_models = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "o3", "o3-pro", "o1", "o1-preview", "o1-mini", "o3-mini", "o4-mini"]
instruct_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4.5-preview", "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14"]

class TextModels:

    #Reasoning token models:
    o3 = "o3"
    o3_pro = "o3-pro"
    #TODO: Verify account to unlock gpt-5 models
    gpt_5 = "gpt-5"
    gpt_5_mini = "gpt-5-mini"
    gpt_5_1 = "gpt-5.1"
    gpt_5_2 = "gpt-5.2"
    gpt_5_1_codex = "gpt-5.1-codex"
    gpt_5_2_codex = "gpt-5.2-codex"
    gpt_5_1_codex_mini = "gpt-5.1-codex-mini"
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


    #Google models:
    gemini_3_pro = "gemini-3-pro-preview"
    gemini_25_flash = "gemini-2.5-flash"
    gemini_25_pro = "gemini-2.5-pro"
    gemini_25_flash_lite = "gemini-2.5-flash-lite"

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


class AudioModels:
    # Legacy / older audio models
    tts = "tts-1"
    tts_hd = "tts-1-hd"
    whisper = "whisper-1"

    gpt_audio_mini = "gpt-audio-mini"
    gpt_audio = "gpt-audio"

    # New GPT-4o-based speech-to-text models
    gpt_4o_transcribe = "gpt-4o-transcribe"
    gpt_4o_mini_transcribe = "gpt-4o-mini-transcribe"

    # New GPT-4o-based text-to-speech model
    gpt_4o_mini_tts = "gpt-4o-mini-tts"

    # New GPT-4o-based realtime models
    gpt_realtime_2025_08_28 = "gpt-realtime-2025-08-28"
    gpt_4o_realtime_preview = "gpt-4o-realtime-preview"
    gpt_4o_realtime_preview_latest = "gpt-4o-realtime-preview-latest"
    gpt_realtime_mini = "gpt-realtime-mini"




class Models:
    text = TextModels
    moderation = "text-moderation-latest"
    embedding = EmbeddingModels
    vision = "gpt-4-vision-preview"
    images = {
        "latest": "dalle-3",
        "legacy": "dalle-2",
    }
    audio = AudioModels
