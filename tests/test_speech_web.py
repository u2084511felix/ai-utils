import io
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aiutils.modules.speech import SpeechIO, SpeechMode
from aiutils.modules.web import SearchTools, UserLocation, WebSearch, WebSearchResponsesModel


@pytest.fixture
def speech(client):
    client.client.audio = SimpleNamespace(
        transcriptions=SimpleNamespace(create=Mock(return_value="transcribed")),
        speech=SimpleNamespace(create=Mock(return_value=b"audio")),
    )
    # Explicit model overrides isolate audio request handling from preset setup.
    return SpeechIO(client=client, stt_model="stt-test", tts_model="tts-test")


@pytest.mark.asyncio
@pytest.mark.parametrize("result,expected", [("words", "words"), (SimpleNamespace(text="words"), "words"), (42, "42")])
async def test_transcribe_rewinds_file_and_extracts_text(speech, result, expected):
    file = io.BytesIO(b"recording")
    file.seek(3)
    create = speech.client.client.audio.transcriptions.create
    create.return_value = result
    assert await speech.transcribe(file, language="en", temperature=0.2) == expected
    assert file.tell() == 0
    create.assert_called_once_with(model="stt-test", file=file, response_format="text", language="en", temperature=0.2)


@pytest.mark.asyncio
async def test_transcribe_accepts_nonseekable_input(speech):
    file = object()
    assert await speech.transcribe(file) == "transcribed"
    assert "language" not in speech.client.client.audio.transcriptions.create.call_args.kwargs


@pytest.mark.asyncio
async def test_synthesize_uses_defaults(speech):
    assert await speech.synthesize("hello") == b"audio"
    speech.client.client.audio.speech.create.assert_called_once_with(model="tts-test", voice="nova", input="hello", response_format="mp3")


@pytest.mark.asyncio
async def test_synthesize_reads_response_and_accepts_overrides(speech):
    response = Mock()
    response.read.return_value = b"wav-audio"
    speech.client.client.audio.speech.create.return_value = response
    assert await speech.synthesize("hello", voice="alloy", response_format="wav", speed=1.2) == b"wav-audio"
    speech.client.client.audio.speech.create.assert_called_once_with(model="tts-test", voice="alloy", input="hello", response_format="wav", speed=1.2)
    response.read.assert_called_once_with()


@pytest.mark.asyncio
async def test_synthesize_falls_back_to_encoded_text(speech):
    speech.client.client.audio.speech.create.return_value = "audio"
    assert await speech.synthesize("hello") == b"audio"


@pytest.mark.asyncio
async def test_audio_provider_errors_propagate(speech):
    error = RuntimeError("audio failed")
    speech.client.client.audio.transcriptions.create.side_effect = error
    with pytest.raises(RuntimeError) as exc:
        await speech.transcribe(io.BytesIO())
    assert exc.value is error
    speech.client.client.audio.speech.create.side_effect = error
    with pytest.raises(RuntimeError) as exc:
        await speech.synthesize("hello")
    assert exc.value is error


@pytest.mark.xfail(strict=True, reason="Pydantic does not invoke the dataclass __post_init__ that selects speech presets")
@pytest.mark.parametrize("mode,stt,tts", [
    (SpeechMode.FAST, "gpt-4o-mini-transcribe", "gpt-4o-mini-tts"),
    (SpeechMode.LEGACY, "whisper-1", "tts-1"),
    (SpeechMode.HIGH_QUALITY, "gpt-4o-transcribe", "gpt-4o-mini-tts"),
])
def test_speech_presets(client, mode, stt, tts):
    instance = SpeechIO(client=client, mode=mode)
    assert instance.stt_model == stt
    assert instance.tts_model == tts


def test_search_tool_location_and_independent_defaults():
    location = UserLocation(country="US", city="Seattle", region="WA")
    tool = SearchTools(search_context_size="low", user_location=location)
    assert tool.model_dump()["user_location"]["type"] == "approximate"
    first, second = WebSearchResponsesModel(), WebSearchResponsesModel()
    first.tools.append(tool)
    assert second.tools == []


@pytest.mark.xfail(strict=True, raises=AttributeError, reason="WebSearch dataclass constructor bypasses Pydantic initialization")
def test_web_search_can_be_constructed(client):
    assert WebSearch(client=client).client is client
