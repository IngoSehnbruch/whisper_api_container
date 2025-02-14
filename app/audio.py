import ffmpeg
import os
import tempfile
import logging
import numpy as np
from fastapi import UploadFile, HTTPException
from typing import Optional

logger = logging.getLogger("whisper-api")

class AudioProcessor:
    """Handles audio processing and format conversion."""
    
    def __init__(self):
        """Initialize the audio processor."""
        self.sample_rate = 16000  # Required by Home Assistant
        self.channels = 1  # Mono
        self.bit_depth = 16
        self._temp_files = []
        
        # VAD parameters
        self.vad_frame_duration = 30  # ms
        self.vad_threshold = 0.3
        self.vad_energy_threshold = 0.001
    
    async def process_audio(self, file: UploadFile) -> bytes:
        """
        Process audio file to match Home Assistant requirements.
        
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
            
            # Create temporary file for output
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            self._temp_files.append(temp_output.name)
            temp_output.close()
            
            try:
                # Convert audio using ffmpeg
                (
                    ffmpeg
                    .input(temp_input.name)
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
                    processed_audio = f.read()
                
                return processed_audio
                
            except ffmpeg.Error as e:
                logger.error(f"FFmpeg error: {e.stderr.decode()}")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "stt-provider-unsupported-metadata",
                        "message": "Audio format not supported"
                    }
                )
                
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "stt-stream-failed",
                    "message": str(e)
                }
            )
            
        finally:
            # Clean up temporary files in background
            self.cleanup()
    
    def detect_speech(self, audio_chunk: bytes) -> bool:
        """
        Simple energy-based Voice Activity Detection.
        
        Args:
            audio_chunk: Raw audio data (16-bit PCM)
            
        Returns:
            True if speech is detected, False otherwise
        """
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # Calculate energy
            energy = np.mean(np.abs(audio_data))
            normalized_energy = energy / 32768.0  # Normalize by max int16 value
            
            return normalized_energy > self.vad_energy_threshold
            
        except Exception as e:
            logger.error(f"VAD processing failed: {str(e)}")
            return False
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.error(f"Failed to delete temporary file {temp_file}: {str(e)}")
        
        self._temp_files = []
