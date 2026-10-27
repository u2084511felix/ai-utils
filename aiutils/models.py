"""Model availability for various AI vendors and their models."""
from enum import StrEnum


class Vendor(StrEnum):
    OPENAI = "openai"
    GOOGLE = "google"
    BRAVE = "brave"


class _GoogleModels:
    gemini_3_pro = "gemini-3-pro-preview"
    gemini_25_flash = "gemini-2.5-flash"
    gemini_25_pro = "gemini-2.5-pro"
    gemini_25_flash_lite = "gemini-2.5-flash-lite"


class _OpenAIModels:
    #Reasoning token models:
    o3 = "o3"
    o3_pro = "o3-pro"
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

    gpt_4_1 = "gpt-4.1"
    gpt_4_1_mini = "gpt-4.1-mini"
    gpt_4_1_nano = "gpt-4.1-nano"
    gpt_4_1_2025_04_14 = "gpt-4.1-2025-04-14"
    gpt_4_1_mini_2025_04_14 = "gpt-4.1-mini-2025-04-14"
    gpt_4_1_nano_2025_04_14 = "gpt-4.1-nano-2025-04-14"
    gpt_5_4 = "gpt-5.4"
    gpt_5_4_mini = "gpt-5.4-mini"
    gpt_5_4_nano = "gpt-5.4-nano"
    gpt_5_4_pro = "gpt-5.4-pro"
    gpt_5_2_pro = "gpt-5.2-pro"
    gpt_5_pro = "gpt-5-pro"
    o1_pro = "o1-pro"
    gpt_4o_2024_05_13 = "gpt-4o-2024-05-13"
    gpt_4_turbo_2024_04_09 = "gpt-4-turbo-2024-04-09"
    gpt_4_0125_preview = "gpt-4-0125-preview"
    gpt_4_1106_preview = "gpt-4-1106-preview"
    gpt_4_1106_vision_preview = "gpt-4-1106-vision-preview"
    gpt_4_0613 = "gpt-4-0613"
    gpt_4_0314 = "gpt-4-0314"
    gpt_4_32k = "gpt-4-32k"
    gpt_3_5_turbo_0125 = "gpt-3.5-turbo-0125"
    gpt_3_5_turbo_1106 = "gpt-3.5-turbo-1106"
    gpt_3_5_turbo_0613 = "gpt-3.5-turbo-0613"
    gpt_3_5_0301 = "gpt-3.5-0301"
    gpt_3_5_turbo_instruct = "gpt-3.5-turbo-instruct"
    gpt_3_5_turbo_16k_0613 = "gpt-3.5-turbo-16k-0613"

    davinci_002 = "davinci-002"
    babbage_002 = "babbage-002"

class LegacyModels:
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
    o1preview = "o1-preview"
    o1mini = "o1-mini"
    o3mini = "o3-mini"
    o3 = "o3"
    nano41 = "gpt-4.1-nano"


class TypeSafeModels:
    jev_1_14_0 = "jev-1.14.0"
    jev_1_13_0 = "jev-1.13.0"
    jev = "jev-latest"


def _model_members(*sources):
    return {
        name: value
        for source in sources
        for name, value in vars(source).items()
        if not name.startswith('_') and isinstance(value, str)
    }


GoogleModels = StrEnum('GoogleModels', _model_members(_GoogleModels))
OpenAIModels = StrEnum('OpenAIModels', _model_members(_OpenAIModels))
TextModels = StrEnum('TextModels', _model_members(_GoogleModels, _OpenAIModels, LegacyModels))
ClassificationModels = StrEnum('ClassificationModels', _model_members(TypeSafeModels))


class EmbeddingModels(StrEnum):
    """
    cost = tokens per dollar

    models                     cost            score   max_tokens
    text-embedding-3-small	   50,000,000	   62.3%   8192
    text-embedding-3-large	   7,692,000	   64.6%   8192
    text-embedding-ada-002	   10,000,000	   61.0%   8192

    """
    LARGE = "text-embedding-3-large"
    SMALL = "text-embedding-3-small"
    LEGACY = "text-embedding-ada-002"


class ModerationModels(StrEnum):
    """Supported moderation models."""
    text_moderation_latest = "text-moderation-latest"


class VisionModels(StrEnum):
    """Supported vision models."""
    gpt_4_vision_preview = "gpt-4-vision-preview"


class ImageModels(StrEnum):
    """Supported image models."""
    latest = "dalle-3"
    legacy = "dalle-2"


class AudioModels(StrEnum):
    """Supported audio models."""
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




PEDANTIC_REASONING = [
    TextModels.gpt_5,
    TextModels.gpt_5_mini,
    TextModels.gpt_5_nano,
    TextModels.gemini_25_pro,
    TextModels.gemini_3_pro
]
REASONING_MODELS = [
    TextModels.gemini_25_flash_lite,
    TextModels.gemini_25_flash,
    TextModels.o3,
    TextModels.o3_pro,
    TextModels.o1,
    TextModels.o1preview,
    TextModels.o1mini,
    TextModels.o3mini,
    TextModels.o4mini,
    TextModels.gpt_5_1,
    TextModels.gpt_5_2
]
INSTRUCT_MODELS = [
    TextModels.gpt_3_5_turbo,
    TextModels.gpt_4,
    TextModels.gpt_4_turbo,
    TextModels.gpt_4o,
    TextModels.gpt_4o_mini,
    TextModels.gpt_4_5_preview,
    TextModels.gpt_4_1_2025_04_14,
    TextModels.gpt_4_1_mini_2025_04_14,
    TextModels.gpt_4_1_nano_2025_04_14
]


class Models:
    text = TextModels
    moderation = ModerationModels
    embedding = EmbeddingModels
    vision = VisionModels
    images = ImageModels
    audio = AudioModels
