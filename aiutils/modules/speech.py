from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


from ..clients import make_client, Client
from .gpt_module import GPTModule
from ..models import AudioModels


class SpeechMode(StrEnum):
    """High level presets for your audio pipeline."""
    HIGH_QUALITY = "high_quality"   # gpt-4o-transcribe + gpt-4o-mini-tts
    FAST = "fast"                   # gpt-4o-mini-transcribe + gpt-4o-mini-tts
    LEGACY = "legacy"               # whisper-1 + tts-1


class SpeechIO(GPTModule):
    mode: SpeechMode = SpeechMode.HIGH_QUALITY
    tts_voice: str = "nova"
    tts_format: str = "mp3"

    stt_model: str = field(init=False)
    tts_model: str = field(init=False)
    client: Client = field(default_factory=make_client)

    def __post_init__(self):
        match self.mode:
            case SpeechMode.FAST:
                self.stt_model = AudioModels.gpt_4o_mini_transcribe
                self.tts_model = AudioModels.gpt_4o_mini_tts
            case SpeechMode.LEGACY:
                self.stt_model = AudioModels.whisper
                self.tts_model = AudioModels.tts
            case _:
                self.stt_model = AudioModels.gpt_4o_transcribe
                self.tts_model = AudioModels.gpt_4o_mini_tts

    async def transcribe(
        self,
        file,
        *,
        response_format: str = "text",
        language: str | None = None,
        **kwargs: Any,
    ) -> str:
        params: dict[str, Any] = {
            "model": self.stt_model,
            "file": file,
            "response_format": response_format,
        }
        if language:
            params["language"] = language
        params.update(kwargs)

        # Make sure the file pointer is at the beginning
        try:
            file.seek(0)
        except Exception:
            pass

        result = self.client.client.audio.transcriptions.create(**params)

        if isinstance(result, str):
            return result
        if hasattr(result, "text"):
            return result.text
        return str(result)

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str | None = None,
        **kwargs: Any,
    ) -> bytes:
        """Text → audio using the configured TTS model.
        
        Args:
            text: The text to be converted to speech.
            voice: The voice to use for TTS. Defaults to the configured TTS voice.
            response_format: The audio format for the output. Defaults to the configured TTS format.
            **kwargs: Additional parameters to pass to the TTS model.
        
        Returns:
            bytes: Raw audio bytes suitable for base64 encoding.
        """


        params = {
            "model": self.tts_model,
            "voice": voice or self.tts_voice,
            "input": text,
            "response_format": response_format or self.tts_format,
        }
        params.update(kwargs)

        audio_resp = self.client.client.audio.speech.create(**params)

        # This is the key: read bytes from the response object
        if hasattr(audio_resp, "read"):
            return audio_resp.read()

        # Extra fallback just in case (you probably won't need this)
        try:
            return bytes(audio_resp)  # type: ignore[arg-type]
        except Exception:
            return str(audio_resp).encode("utf-8")