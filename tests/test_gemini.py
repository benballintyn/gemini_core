import os
from unittest.mock import MagicMock

import pytest
from google import genai
from google.genai import types

from gemini_core.data_models.models import GeminiConfig
from gemini_core.gemini import Gemini


@pytest.fixture
def mock_genai_client(mocker):
    return mocker.patch("gemini_core.gemini.genai.Client")


@pytest.fixture
def mock_env_vars(mocker):
    mocker.patch.dict(
        os.environ,
        {
            "GOOGLE_API_KEY": "test_key",  # pragma: allowlist secret
            "GEMINI_MODEL": "test-model",
        },
    )


def test_gemini_initialization_with_env(mock_genai_client, mock_env_vars):
    gemini = Gemini()
    assert gemini.config.api_key == "test_key"  # pragma: allowlist secret
    assert gemini.config.model_name == "test-model"
    mock_genai_client.assert_called_once_with(
        api_key="test_key"  # pragma: allowlist secret
    )


def test_gemini_initialization_explicit(mock_genai_client):
    gemini = Gemini(
        api_key="explicit_key",  # pragma: allowlist secret
        model_name="explicit-model",
    )
    assert gemini.config.api_key == "explicit_key"  # pragma: allowlist secret
    assert gemini.config.model_name == "explicit-model"
    mock_genai_client.assert_called_once_with(
        api_key="explicit_key"  # pragma: allowlist secret
    )


def test_gemini_initialization_no_key_raises_error(mocker):
    mocker.patch.dict(os.environ, {}, clear=True)
    with pytest.raises(
        ValueError, match="GOOGLE_API_KEY environment variable is not set"
    ):
        Gemini()


def test_generate_content(mock_genai_client, mock_env_vars):
    mock_client_instance = mock_genai_client.return_value
    mock_response = MagicMock(spec=types.GenerateContentResponse)
    mock_response.text = "Generated text"
    mock_client_instance.models.generate_content.return_value = mock_response

    gemini = Gemini()
    response = gemini.generate_content("Test prompt")

    assert response.text == "Generated text"
    mock_client_instance.models.generate_content.assert_called_once()
    call_args = mock_client_instance.models.generate_content.call_args
    assert call_args.kwargs["model"] == "test-model"
    assert call_args.kwargs["contents"] == "Test prompt"
    assert isinstance(call_args.kwargs["config"], types.GenerateContentConfig)


def test_generate_content_with_config_override(mock_genai_client, mock_env_vars):
    mock_client_instance = mock_genai_client.return_value

    gemini = Gemini()
    override_config = GeminiConfig(temperature=0.7)
    gemini.generate_content("Test prompt", generation_config=override_config)

    call_args = mock_client_instance.models.generate_content.call_args
    assert call_args.kwargs["config"].temperature == 0.7


def test_generate_content_structured_output(mock_genai_client, mock_env_vars):
    mock_client_instance = mock_genai_client.return_value

    gemini = Gemini()

    class TestSchema(GeminiConfig):  # Just using a dummy model for schema
        pass

    override_config = GeminiConfig(response_schema=TestSchema)
    gemini.generate_content("Test prompt", generation_config=override_config)

    call_args = mock_client_instance.models.generate_content.call_args
    assert call_args.kwargs["config"].response_schema == TestSchema
    assert call_args.kwargs["config"].response_mime_type == "application/json"


def test_generate_content_with_thinking_and_tools(mock_genai_client, mock_env_vars):
    mock_client_instance = mock_genai_client.return_value

    gemini = Gemini()

    def dummy_tool():
        pass

    override_config = GeminiConfig(thinking_level="high", tools=[dummy_tool])
    gemini.generate_content("Test prompt", generation_config=override_config)

    call_args = mock_client_instance.models.generate_content.call_args
    # SDK converts string to enum
    assert (
        call_args.kwargs["config"].thinking_config.thinking_level
        == types.ThinkingLevel.HIGH
    )
    assert call_args.kwargs["config"].thinking_config.include_thoughts is True
    assert call_args.kwargs["config"].tools == [dummy_tool]


def test_upload_file(mock_genai_client, mock_env_vars):
    mock_client_instance = mock_genai_client.return_value
    mock_file = MagicMock(spec=types.File)
    mock_client_instance.files.upload.return_value = mock_file

    gemini = Gemini()
    result = gemini.upload_file("path/to/file.jpg", mime_type="image/jpeg")

    assert result == mock_file
    mock_client_instance.files.upload.assert_called_once()
    call_args = mock_client_instance.files.upload.call_args
    assert call_args.kwargs["file"] == "path/to/file.jpg"
    assert call_args.kwargs["config"].mime_type == "image/jpeg"


def test_start_chat(mock_genai_client, mock_env_vars):
    mock_client_instance = mock_genai_client.return_value
    mock_chat = MagicMock(spec=genai.chats.Chat)
    mock_client_instance.chats.create.return_value = mock_chat

    gemini = Gemini()
    chat = gemini.start_chat(history=["Hello"])

    assert chat == mock_chat
    mock_client_instance.chats.create.assert_called_once()
    call_args = mock_client_instance.chats.create.call_args
    assert call_args.kwargs["model"] == "test-model"
    assert call_args.kwargs["history"] == ["Hello"]


def test_generate_content_stream(mock_genai_client, mock_env_vars):
    mock_client_instance = mock_genai_client.return_value
    mock_response_chunk = MagicMock(spec=types.GenerateContentResponse)
    mock_response_chunk.text = "Chunk"
    mock_client_instance.models.generate_content_stream.return_value = [
        mock_response_chunk
    ]

    gemini = Gemini()
    stream = gemini.generate_content_stream("Test prompt")
    chunks = list(stream)

    assert len(chunks) == 1
    assert chunks[0].text == "Chunk"
    mock_client_instance.models.generate_content_stream.assert_called_once()


@pytest.mark.asyncio
async def test_generate_content_async(mock_genai_client, mock_env_vars):
    mock_client_instance = mock_genai_client.return_value
    mock_response = MagicMock(spec=types.GenerateContentResponse)
    mock_response.text = "Async text"

    # Mock async call
    async def async_return():
        return mock_response

    mock_client_instance.aio.models.generate_content.side_effect = (
        lambda **kwargs: async_return()
    )

    gemini = Gemini()
    response = await gemini.generate_content_async("Test prompt")

    assert response.text == "Async text"
    mock_client_instance.aio.models.generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_generate_content_stream_async(mock_genai_client, mock_env_vars):
    mock_client_instance = mock_genai_client.return_value
    mock_response_chunk = MagicMock(spec=types.GenerateContentResponse)
    mock_response_chunk.text = "Async Chunk"

    # Mock async generator
    async def async_gen(**kwargs):
        yield mock_response_chunk

    mock_client_instance.aio.models.generate_content_stream.side_effect = async_gen

    gemini = Gemini()
    chunks = []
    async for chunk in gemini.generate_content_stream_async("Test prompt"):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].text == "Async Chunk"
    mock_client_instance.aio.models.generate_content_stream.assert_called_once()


@pytest.mark.asyncio
async def test_count_tokens_async(mock_genai_client, mock_env_vars):
    mock_client_instance = mock_genai_client.return_value
    mock_response = MagicMock(spec=types.CountTokensResponse)
    mock_response.total_tokens = 10

    async def async_return():
        return mock_response

    mock_client_instance.aio.models.count_tokens.side_effect = (
        lambda **kwargs: async_return()
    )

    gemini = Gemini()
    response = await gemini.count_tokens_async("Test prompt")

    assert response.total_tokens == 10
    mock_client_instance.aio.models.count_tokens.assert_called_once()


@pytest.mark.asyncio
async def test_upload_file_async(mock_genai_client, mock_env_vars):
    mock_client_instance = mock_genai_client.return_value
    mock_file = MagicMock(spec=types.File)

    async def async_return():
        return mock_file

    mock_client_instance.aio.files.upload.side_effect = lambda **kwargs: async_return()

    gemini = Gemini()
    result = await gemini.upload_file_async("path/to/file.jpg")

    assert result == mock_file
    mock_client_instance.aio.files.upload.assert_called_once()


@pytest.mark.asyncio
async def test_start_chat_async(mock_genai_client, mock_env_vars):
    mock_client_instance = mock_genai_client.return_value
    mock_chat = MagicMock(spec=genai.chats.AsyncChat)
    mock_client_instance.aio.chats.create.return_value = mock_chat

    gemini = Gemini()
    chat = await gemini.start_chat_async(history=["Hello"])

    assert chat == mock_chat
    mock_client_instance.aio.chats.create.assert_called_once()


def test_count_tokens(mock_genai_client, mock_env_vars):
    mock_client_instance = mock_genai_client.return_value
    mock_response = MagicMock(spec=types.CountTokensResponse)
    mock_response.total_tokens = 10
    mock_client_instance.models.count_tokens.return_value = mock_response

    gemini = Gemini()
    response = gemini.count_tokens("Test prompt")

    assert response.total_tokens == 10
    mock_client_instance.models.count_tokens.assert_called_once_with(
        model="test-model", contents="Test prompt"
    )


def test_generate_content_error(mock_genai_client, mock_env_vars):
    mock_client_instance = mock_genai_client.return_value
    mock_client_instance.models.generate_content.side_effect = Exception("API Error")

    gemini = Gemini()
    with pytest.raises(Exception, match="API Error"):
        gemini.generate_content("Test prompt")
