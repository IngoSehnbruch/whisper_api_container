"""Wyoming protocol server implementation."""
import asyncio
import logging
from typing import Optional, AsyncGenerator
import wave

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.info import Info, AsrInfo
from wyoming.server import AsyncServer, AsyncTcpServer
from wyoming.event import Event

from transcriber import WhisperTranscriber
from audio import AudioProcessor
from utils import ResourceManager

_LOGGER = logging.getLogger(__name__)

class WhisperWyomingServer:
    """Wyoming protocol server for Whisper STT."""

    def __init__(
        self,
        host: str,
        port: int,
        transcriber: WhisperTranscriber,
        audio_processor: AudioProcessor,
        resource_manager: ResourceManager,
    ) -> None:
        """Initialize the server."""
        self.host = host
        self.port = port
        self.transcriber = transcriber
        self.audio_processor = audio_processor
        self.resource_manager = resource_manager
        self.server: Optional[AsyncServer] = None

    async def start(self) -> None:
        """Start the server."""
        self.server = AsyncTcpServer(self.host, self.port, handle_client=self.handle_client)
        await self.server.start()
        _LOGGER.info("Wyoming server started on %s:%s", self.host, self.port)

    async def stop(self) -> None:
        """Stop the server."""
        if self.server:
            await self.server.stop()
            self.server = None
            _LOGGER.info("Wyoming server stopped")

    async def handle_client(self, client: AsyncServer.Client) -> None:
        """Handle a client connection."""
        try:
            async for event in client:
                if isinstance(event, Info):
                    # Client requested server info
                    await client.write_event(
                        Info(
                            asr=AsrInfo(
                                models=self.transcriber.supported_models,
                                languages=self.transcriber.supported_languages,
                            )
                        )
                    )
                elif isinstance(event, Transcribe):
                    # Client wants to transcribe audio
                    try:
                        # Check resources
                        await self.resource_manager.check_resources()

                        # Set language and model if provided
                        if event.language:
                            self.transcriber.language = event.language
                        if event.model:
                            await self.transcriber.load_model(event.model)

                        # Process audio stream
                        audio_data = await self._process_audio_stream(client)
                        if audio_data:
                            # Transcribe audio
                            result = await self.transcriber.transcribe(audio_data)
                            await client.write_event(Transcript(text=result["text"]))
                    except Exception as e:
                        _LOGGER.error("Transcription error: %s", str(e))
                        await client.write_event(
                            Error(
                                code="transcription_error",
                                message=str(e)
                            )
                        )

        except Exception as e:
            _LOGGER.error("Client error: %s", str(e))
            try:
                await client.write_event(
                    Error(
                        code="client_error",
                        message=str(e)
                    )
                )
            except Exception:
                pass

    async def _process_audio_stream(self, client: AsyncServer.Client) -> Optional[bytes]:
        """Process audio stream from client."""
        audio_buffer = bytearray()
        audio_started = False

        async for event in client:
            if isinstance(event, AudioStart):
                audio_started = True
                continue
            elif isinstance(event, AudioStop):
                break
            elif isinstance(event, AudioChunk) and audio_started:
                audio_buffer.extend(event.audio)
            elif isinstance(event, Error):
                raise Exception(event.message)

        if not audio_buffer:
            return None

        # Process audio to match requirements
        return await self.audio_processor.process_audio_bytes(bytes(audio_buffer))
