"""Audio processing utilities."""
import ffmpeg
import os
import tempfile
import logging
import numpy as np
import wave
from fastapi import UploadFile, HTTPException
from typing import Optional, Union

logger = logging.getLogger("whisper-api")

class AudioProcessor:
    """Handles audio processing and format conversion."""
    
    def __init__(self):
        """Initialize the audio processor."""
        self.sample_rate = 16000  # Required by Wyoming/Whisper
        self.channels = 1  # Mono
        self.bit_depth = 16
        self._temp_files = []
    
    async def process_audio(self, file: UploadFile) -> bytes:
        """
        Process audio file to match Wyoming protocol requirements.
        
        Args:
            file: Audio file upload
            
        Returns:
            Processed audio data as bytes
        """
        try:
            # Create temporary file for input
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            self._temp_files.append(temp_input.name)
            
            # Save uploaded file
            content = await file.read()
            temp_input.write(content)
            temp_input.close()
            
            return await self._process_audio_file(temp_input.name)
                
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
            
        finally:
            # Clean up temporary files
            self.cleanup()

    async def process_audio_bytes(self, audio_data: bytes) -> bytes:
        """
        Process raw audio bytes to match Wyoming protocol requirements.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Processed audio data as bytes
        """
        try:
            # Create temporary file for input
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            self._temp_files.append(temp_input.name)
            
            # Write audio data with WAV header
            with wave.open(temp_input.name, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.bit_depth // 8)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data)
            
            return await self._process_audio_file(temp_input.name)
                
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
            
        finally:
            # Clean up temporary files
            self.cleanup()

    async def _process_audio_file(self, input_file: str) -> bytes:
        """
        Process audio file to match requirements.
        
        Args:
            input_file: Path to input audio file
            
        Returns:
            Processed audio data as bytes
        """
        # Create temporary file for output
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        self._temp_files.append(temp_output.name)
        temp_output.close()
        
        try:
            # Convert audio using ffmpeg
            (
                ffmpeg
                .input(input_file)
                .output(
                    temp_output.name,
                    acodec='pcm_s16le',
                    ac=self.channels,
                    ar=self.sample_rate,
                    f='wav'
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            # Read processed audio
            with open(temp_output.name, 'rb') as f:
                # Skip WAV header (44 bytes) for Wyoming protocol
                f.seek(44)
                processed_audio = f.read()
            
            return processed_audio
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise HTTPException(
                status_code=400,
                detail="Audio format not supported"
            )

    def cleanup(self) -> None:
        """Clean up temporary files."""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.error(f"Failed to remove temporary file {temp_file}: {str(e)}")
        self._temp_files.clear()
