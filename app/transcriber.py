"""Whisper transcription service."""
import os
import torch
import logging
import numpy as np
from typing import Dict, Any, List, Optional
import whisper

logger = logging.getLogger("whisper-api")

class WhisperTranscriber:
    """Handles transcription using Whisper model."""
    
    def __init__(self):
        """Initialize the transcriber."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_model = None
        self.current_model_name = None
        self.language = None
        
        # List of supported models
        self.supported_models = [
            "tiny", "base", "small", "medium", "large-v3"
        ]
        
        # Load language codes
        self.supported_languages = list(whisper.tokenizer.LANGUAGES.keys())
        
        logger.info(f"Initialized Whisper transcriber (device: {self.device})")
    
    async def load_model(self, model_name: str) -> None:
        """
        Load a Whisper model.
        
        Args:
            model_name: Name of the model to load
        """
        if model_name not in self.supported_models:
            raise ValueError(f"Unsupported model: {model_name}")
            
        if self.current_model_name == model_name:
            return
            
        try:
            logger.info(f"Loading Whisper model: {model_name}")
            self.current_model = whisper.load_model(model_name, device=self.device)
            self.current_model_name = model_name
            logger.info(f"Model loaded successfully: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    async def transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Dict[str, Any]:
        """
        Transcribe audio data.
        
        Args:
            audio_data: Raw PCM audio data (16-bit, 16kHz, mono)
            language: Optional language code (ISO 639-1)
            task: Task to perform (transcribe or translate)
            
        Returns:
            Dictionary containing transcription results
        """
        if not self.current_model:
            raise RuntimeError("No model loaded")
            
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe audio
            result = self.current_model.transcribe(
                audio_np,
                language=language or self.language,
                task=task,
                fp16=torch.cuda.is_available()
            )
            
            # Format response
            response = {
                "text": result["text"],
                "language": result["language"],
                "segments": result["segments"]
            }
            
            # Add confidence if available
            if "confidence" in result:
                response["confidence"] = result["confidence"]
            
            return response
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "name": self.current_model_name,
            "device": self.device,
            "loaded": self.current_model is not None,
            "supported_models": self.supported_models,
            "supported_languages": self.supported_languages,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
