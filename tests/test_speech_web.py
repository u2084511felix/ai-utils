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
async def test_transcribe_rewinds_file_and_extracts_text(audio_file, result, expected):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN the request starts at the start of the file and returns text."""
    create, file, speech = audio_file
    result = await speech.transcribe(file, language="en", temperature=0.2)
    actual = (result, file.tell(), create.call_args_list)
    expected = (expected, 0, [mock_call(model="stt-test", file=file, response_format="text", language="en", temperature=0.2)])
    assert actual == expected


@pytest.fixture
def audio_stream(speech):
    file = object()
    return (file, speech)


@pytest.mark.asyncio
async def test_transcribe_accepts_nonseekable_input(audio_stream):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN the input is accepted and no language is specified."""
    file, speech = audio_stream
    result = await speech.transcribe(file)
    actual = (result, "language" not in speech.client.client.audio.transcriptions.create.call_args.kwargs)
    expected = ("transcribed", True)
    assert actual == expected


@pytest.mark.asyncio
async def test_synthesize_uses_defaults(speech):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN the request uses the default voice and format and returns audio."""
    result = await speech.synthesize("hello")
    actual = (result, speech.client.client.audio.speech.create.call_args_list)
    expected = (b"audio", [mock_call(model="tts-test", voice="nova", input="hello", response_format="mp3")])
    assert actual == expected


@pytest.fixture
def audio_response(speech):
    response = Mock()
    response.read.return_value = b"wav-audio"
    speech.client.client.audio.speech.create.return_value = response
    return (response, speech)


@pytest.mark.asyncio
async def test_synthesize_reads_response_and_accepts_overrides(audio_response):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN the request uses the specified options and returns the audio bytes."""
    response, speech = audio_response
    result = await speech.synthesize("hello", voice="alloy", response_format="wav", speed=1.2)
    actual = (result, speech.client.client.audio.speech.create.call_args_list, response.read.call_args_list)
    expected = (b"wav-audio", [mock_call(model="tts-test", voice="alloy", input="hello", response_format="wav", speed=1.2)], [mock_call()])
    assert actual == expected


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
def test_speech_presets(client, mode, stt, tts):
    """GIVEN a speech mode.
    WHEN the speech module is created.
    THEN the transcription and speech models match the selected mode."""
    instance = SpeechIO(client=client, mode=mode)
    actual = (instance.stt_model, instance.tts_model)
    expected = (stt, tts)
    assert actual == expected


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
async def test_web_search_returns_result_and_forwards_request(search_request):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN the request uses the specified tool and returns the search text."""
    tool, web_search = search_request
    result = await web_search.web_search(input="question", model="gpt-4o", tools=[tool])
    actual = (result, web_search.client.async_responses_create.await_args_list)
    expected = ("search result", [mock_call(input="question", model="gpt-4o", tools=[tool.model_dump()])])
    assert actual == expected


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
async def test_web_search_applies_tool_overrides(web_search):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN the request uses the specified tool values."""
    await web_search.web_search(input="question", tool_dict={"type": "custom_search", "search_context_size": "high"})
    tool = web_search.client.async_responses_create.await_args.kwargs["tools"][0]
    actual = (tool["type"], tool["search_context_size"])
    expected = ("custom_search", "high")
    assert actual == expected


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
async def test_web_search_omits_only_none_input(web_search, text):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN the request omits absent input and retains empty text."""
    await web_search.web_search(input=text)
    kwargs = web_search.client.async_responses_create.await_args.kwargs
    result = ("input" in kwargs, kwargs.get("input"))
    expected = (text is not None, text)
    assert result == expected


@pytest.mark.asyncio
async def test_web_search_empty_overrides_selects_default_tool(web_search):
    """GIVEN a client and specified request options.
    WHEN the request operation is called.
    THEN the request uses one default search tool."""
    result = await web_search.web_search(input="question", tool_dict={})
    tools = web_search.client.async_responses_create.await_args.kwargs["tools"]
    actual = (result, len(tools), tools[0]["type"])
    expected = ("search result", 1, "web_search_preview")
    assert actual == expected
