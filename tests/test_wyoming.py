"""Tests for Wyoming protocol integration."""
import asyncio
import pytest
from unittest.mock import MagicMock, patch
import json
import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioStart, AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Info, AsrInfo, AsrModel, AsrProgram

from app.wyoming import WhisperWyomingServer
from app.transcriber import WhisperTranscriber

@pytest.fixture
def transcriber():
    """Create a mock transcriber."""
    mock_transcriber = MagicMock(spec=WhisperTranscriber)
    mock_transcriber.get_model_info.return_value = {
        "name": "tiny",
        "device": "cpu",
        "loaded": True,
        "supported_models": ["tiny", "base"],
        "supported_languages": ["en", "es"]
    }
    return mock_transcriber

@pytest.fixture
async def wyoming_server(transcriber):
    """Create a Wyoming server instance."""
    server = WhisperWyomingServer(
        host="127.0.0.1",
        port=10300,
        transcriber=transcriber
    )
    yield server
    await server.stop()

async def test_info_event(wyoming_server):
    """Test handling of info event."""
    # Create info event
    event = Event(
        type="info",
        data=Info().event()
    )
    
    # Process event
    response = await wyoming_server.handle_event(event)
    
    # Verify response
    assert response.type == "asr-info"
    asr_info = AsrInfo.from_event(response)
    assert asr_info.name == "whisper"
    assert len(asr_info.models) > 0
    assert "tiny" in [m.name for m in asr_info.models]

async def test_transcribe_event(wyoming_server, transcriber):
    """Test handling of transcribe event."""
    # Mock transcriber response
    transcriber.transcribe.return_value = {
        "text": "test transcription",
        "language": "en",
        "confidence": 0.95
    }
    
    # Create audio data
    audio_data = np.zeros(16000, dtype=np.int16).tobytes()
    
    # Send events
    events = [
        Event(type="transcribe", data=Transcribe(language="en").event()),
        Event(type="audio-start", data=AudioStart().event()),
        Event(type="audio-chunk", data=AudioChunk(audio=audio_data).event()),
        Event(type="audio-stop", data=AudioStop().event())
    ]
    
    responses = []
    for event in events:
        response = await wyoming_server.handle_event(event)
        if response:
            responses.append(response)
    
    # Verify transcription response
    assert len(responses) == 1
    assert responses[0].type == "transcript"
    transcript = Transcript.from_event(responses[0])
    assert transcript.text == "test transcription"

async def test_error_handling(wyoming_server, transcriber):
    """Test error handling."""
    # Make transcriber raise an error
    transcriber.transcribe.side_effect = Exception("Test error")
    
    # Create audio data
    audio_data = np.zeros(16000, dtype=np.int16).tobytes()
    
    # Send events
    events = [
        Event(type="transcribe", data=Transcribe().event()),
        Event(type="audio-start", data=AudioStart().event()),
        Event(type="audio-chunk", data=AudioChunk(audio=audio_data).event()),
        Event(type="audio-stop", data=AudioStop().event())
    ]
    
    responses = []
    for event in events:
        response = await wyoming_server.handle_event(event)
        if response:
            responses.append(response)
    
    # Verify error response
    assert len(responses) == 1
    assert responses[0].type == "error"
    assert "Test error" in responses[0].data.decode()

async def test_audio_format(wyoming_server):
    """Test audio format handling."""
    # Create info event
    event = Event(
        type="info",
        data=Info().event()
    )
    
    # Get info response
    response = await wyoming_server.handle_event(event)
    asr_info = AsrInfo.from_event(response)
    
    # Verify audio format
    assert asr_info.audio_formats
    audio_format = asr_info.audio_formats[0]
    assert audio_format.rate == 16000
    assert audio_format.width == 2
    assert audio_format.channels == 1

async def test_model_info(wyoming_server, transcriber):
    """Test model information."""
    # Create info event
    event = Event(
        type="info",
        data=Info().event()
    )
    
    # Get info response
    response = await wyoming_server.handle_event(event)
    asr_info = AsrInfo.from_event(response)
    
    # Verify models
    assert len(asr_info.models) == 2
    model_names = [m.name for m in asr_info.models]
    assert "tiny" in model_names
    assert "base" in model_names
