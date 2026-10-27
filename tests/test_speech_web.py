import io
from types import SimpleNamespace
from unittest.mock import Mock, call as mock_call

import pytest

from aiutils.clients import ClientReply
from aiutils.exceptions import ClientError
from aiutils.modules.speech import SpeechIO, SpeechMode
from aiutils.modules.web import SearchTools, UserLocation, WebSearch, WebSearchResponsesModel


@pytest.fixture
def speech(client):
    client.client.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=Mock(return_value="transcribed")), speech=SimpleNamespace(create=Mock(return_value=b"audio")))
    return SpeechIO(client=client, stt_model="stt-test", tts_model="tts-test")


@pytest.fixture
def audio_file(speech, result):
    file = io.BytesIO(b"recording")
    file.seek(3)
    create = speech.client.client.audio.transcriptions.create
    create.return_value = result
    return (create, file, speech)


@pytest.mark.asyncio
@pytest.mark.parametrize("result,expected", [("words", "words"), (SimpleNamespace(text="words"), "words"), (42, "42")])
async def test_transcription_returns_text(audio_file, result, expected):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN transcription returns text."""
    create, file, speech = audio_file
    result = await speech.transcribe(file, language="en", temperature=0.2)
    expected = expected
    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("result,expected", [("words", "words"), (SimpleNamespace(text="words"), "words"), (42, "42")])
async def test_transcription_rewinds_file(audio_file, result, expected):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN transcription rewinds file."""
    create, file, speech = audio_file
    await speech.transcribe(file, language="en", temperature=0.2)
    expected = 0
    assert file.tell() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("result,expected", [("words", "words"), (SimpleNamespace(text="words"), "words"), (42, "42")])
async def test_transcription_passes_options(audio_file, result, expected):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN transcription passes options."""
    create, file, speech = audio_file
    await speech.transcribe(file, language="en", temperature=0.2)
    expected = [mock_call(model="stt-test", file=file, response_format="text", language="en", temperature=0.2)]
    assert create.call_args_list == expected


@pytest.fixture
def audio_stream(speech):
    file = object()
    return (file, speech)


@pytest.mark.asyncio
async def test_stream_transcription_returns_text(audio_stream):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN stream transcription returns text."""
    file, speech = audio_stream
    result = await speech.transcribe(file)
    expected = "transcribed"
    assert result == expected


@pytest.mark.asyncio
async def test_stream_transcription_omits_language(audio_stream):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN stream transcription omits language."""
    file, speech = audio_stream
    await speech.transcribe(file)
    expected = "language"
    assert expected not in speech.client.client.audio.transcriptions.create.call_args.kwargs


@pytest.mark.asyncio
async def test_default_speech_returns_audio(speech):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN default speech returns audio."""
    result = await speech.synthesize("hello")
    expected = b"audio"
    assert result == expected


@pytest.mark.asyncio
async def test_default_speech_passes_options(speech):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN default speech passes options."""
    await speech.synthesize("hello")
    expected = [mock_call(model="tts-test", voice="nova", input="hello", response_format="mp3")]
    assert speech.client.client.audio.speech.create.call_args_list == expected


@pytest.fixture
def audio_response(speech):
    response = Mock()
    response.read.return_value = b"wav-audio"
    speech.client.client.audio.speech.create.return_value = response
    return (response, speech)


@pytest.mark.asyncio
async def test_custom_speech_returns_audio(audio_response):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN custom speech returns audio."""
    response, speech = audio_response
    result = await speech.synthesize("hello", voice="alloy", response_format="wav", speed=1.2)
    expected = b"wav-audio"
    assert result == expected


@pytest.mark.asyncio
async def test_custom_speech_passes_options(audio_response):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN custom speech passes options."""
    response, speech = audio_response
    await speech.synthesize("hello", voice="alloy", response_format="wav", speed=1.2)
    expected = [mock_call(model="tts-test", voice="alloy", input="hello", response_format="wav", speed=1.2)]
    assert speech.client.client.audio.speech.create.call_args_list == expected


@pytest.mark.asyncio
async def test_custom_speech_reads_response(audio_response):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN custom speech reads response."""
    response, speech = audio_response
    await speech.synthesize("hello", voice="alloy", response_format="wav", speed=1.2)
    expected = [mock_call()]
    assert response.read.call_args_list == expected


@pytest.fixture
def audio_text(speech):
    speech.client.client.audio.speech.create.return_value = "audio"
    return speech


@pytest.mark.asyncio
async def test_synthesize_falls_back_to_encoded_text(audio_text):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN the response text is returned as bytes."""
    speech = audio_text
    result = await speech.synthesize("hello")
    expected = b"audio"
    assert result == expected


@pytest.fixture
def audio_failure(speech):
    error = RuntimeError("audio failed")
    speech.client.client.audio.transcriptions.create.side_effect = error
    speech.client.client.audio.speech.create.side_effect = error
    return (error, speech)


@pytest.mark.asyncio
@pytest.mark.parametrize("method,prompt", [("transcribe", io.BytesIO()), ("synthesize", "hello")])
async def test_audio_provider_errors_propagate(audio_failure, method, prompt):
    """GIVEN an operation that reports an error.
    WHEN the request operation is called.
    THEN the provider error is returned to the caller."""
    error, speech = audio_failure
    with pytest.raises(RuntimeError) as exc:
        await getattr(speech, method)(prompt)
    expected = True
    assert (exc.value is error) == expected


@pytest.mark.parametrize("mode,stt,tts", [(SpeechMode.FAST, "gpt-4o-mini-transcribe", "gpt-4o-mini-tts"), (SpeechMode.LEGACY, "whisper-1", "tts-1"), (SpeechMode.HIGH_QUALITY, "gpt-4o-transcribe", "gpt-4o-mini-tts")])
def test_speech_preset_selects_transcription_model(client, mode, stt, tts):
    """GIVEN a speech mode.
    WHEN the speech module is created.
    THEN speech preset selects transcription model."""
    instance = SpeechIO(client=client, mode=mode)
    expected = stt
    assert instance.stt_model == expected


@pytest.mark.parametrize("mode,stt,tts", [(SpeechMode.FAST, "gpt-4o-mini-transcribe", "gpt-4o-mini-tts"), (SpeechMode.LEGACY, "whisper-1", "tts-1"), (SpeechMode.HIGH_QUALITY, "gpt-4o-transcribe", "gpt-4o-mini-tts")])
def test_speech_preset_selects_synthesis_model(client, mode, stt, tts):
    """GIVEN a speech mode.
    WHEN the speech module is created.
    THEN speech preset selects synthesis model."""
    instance = SpeechIO(client=client, mode=mode)
    expected = tts
    assert instance.tts_model == expected


@pytest.fixture
def location_tool():
    location = UserLocation(country="US", city="Seattle", region="WA")
    tool = SearchTools(search_context_size="low", user_location=location)
    return tool


def test_search_tool_location(location_tool):
    """GIVEN a search tool with a user location.
    WHEN the search tool is serialized.
    THEN the location type is approximate."""
    tool = location_tool
    result = tool.model_dump()["user_location"]["type"]
    expected = "approximate"
    assert result == expected


@pytest.fixture
def changed_search():
    first = WebSearchResponsesModel()
    first.tools.append(SearchTools())


def test_search_tool_independent_defaults(changed_search):
    """GIVEN a search request with an added tool.
    WHEN another search request is created.
    THEN the new request has no tools."""
    second = WebSearchResponsesModel()
    expected = []
    assert second.tools == expected


def test_web_search_can_be_constructed(client):
    """GIVEN a client and specified request options.
    WHEN the search module is created.
    THEN the search module uses the supplied client."""
    result = WebSearch(client=client).client is client
    expected = True
    assert result == expected


@pytest.fixture
def web_search(client, monkeypatch):
    from unittest.mock import AsyncMock
    from aiutils.clients import ClientReply

    monkeypatch.setattr(client, "async_responses_create", AsyncMock(return_value=ClientReply(content="search result")))
    return WebSearch(client=client)


@pytest.fixture
def search_request(web_search):
    tool = SearchTools(search_context_size="low")
    return (tool, web_search)


@pytest.mark.asyncio
async def test_web_search_returns_text(search_request):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN web search returns text."""
    tool, web_search = search_request
    result = await web_search.web_search(input="question", model="gpt-4o", tools=[tool])
    expected = "search result"
    assert result == expected


@pytest.mark.asyncio
async def test_web_search_passes_tool(search_request):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN web search passes tool."""
    tool, web_search = search_request
    await web_search.web_search(input="question", model="gpt-4o", tools=[tool])
    expected = [mock_call(input="question", model="gpt-4o", tools=[tool.model_dump()])]
    assert web_search.client.async_responses_create.await_args_list == expected


@pytest.fixture
def search_failure(web_search):
    error = RuntimeError("search failed")
    web_search.client.async_responses_create.return_value = ClientReply(error=error)
    return (error, web_search)


@pytest.mark.asyncio
async def test_web_search_chains_provider_error(search_failure):
    """GIVEN an operation that reports an error.
    WHEN the request operation is called.
    THEN the error retains the provider error as its cause."""
    error, web_search = search_failure
    with pytest.raises(ClientError) as exc:
        await web_search.web_search(input="question")
    expected = True
    assert (exc.value.__cause__ is error) == expected


@pytest.mark.asyncio
async def test_web_search_overrides_tool_type(web_search):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN web search overrides tool type."""
    await web_search.web_search(input="question", tool_dict={"type": "custom_search", "search_context_size": "high"})
    tool = web_search.client.async_responses_create.await_args.kwargs["tools"][0]
    expected = "custom_search"
    assert tool["type"] == expected


@pytest.mark.asyncio
async def test_web_search_overrides_context_size(web_search):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN web search overrides context size."""
    await web_search.web_search(input="question", tool_dict={"type": "custom_search", "search_context_size": "high"})
    tool = web_search.client.async_responses_create.await_args.kwargs["tools"][0]
    expected = "high"
    assert tool["search_context_size"] == expected


@pytest.fixture
def search_module(client):
    module = WebSearch(client=client)
    return module


@pytest.mark.asyncio
async def test_web_search_uses_client_method_signature(search_module):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN the client accepts the request and returns its text."""
    module = search_module
    result = await module.web_search(input="question")
    expected = "hello"
    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [None, ""])
async def test_web_search_retains_present_input(web_search, text):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN web search retains present input."""
    await web_search.web_search(input=text)
    kwargs = web_search.client.async_responses_create.await_args.kwargs
    expected = text is not None
    assert ("input" in kwargs) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [None, ""])
async def test_web_search_retains_input_value(web_search, text):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN web search retains input value."""
    await web_search.web_search(input=text)
    kwargs = web_search.client.async_responses_create.await_args.kwargs
    expected = text
    assert kwargs.get("input") == expected


@pytest.mark.asyncio
async def test_default_search_returns_text(web_search):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN default search returns text."""
    result = await web_search.web_search(input="question", tool_dict={})
    expected = "search result"
    assert result == expected


@pytest.mark.asyncio
async def test_default_search_uses_one_tool(web_search):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN default search uses one tool."""
    await web_search.web_search(input="question", tool_dict={})
    tools = web_search.client.async_responses_create.await_args.kwargs["tools"]
    expected = 1
    assert len(tools) == expected


@pytest.mark.asyncio
async def test_default_search_selects_tool_type(web_search):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN default search selects tool type."""
    await web_search.web_search(input="question", tool_dict={})
    tools = web_search.client.async_responses_create.await_args.kwargs["tools"]
    expected = "web_search_preview"
    assert tools[0]["type"] == expected
