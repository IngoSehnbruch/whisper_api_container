import whisper
import torch
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("whisper-api")

class WhisperTranscriber:
    """Handles Whisper model management and transcription."""
    
    def __init__(self):
        """Initialize the transcriber."""
        self.model = None
        self.current_model_name = None
        self.device = self._get_device()
        self.supported_languages = [
            "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru",
            "zh", "ja", "ko", "ar", "hi", "tr"
        ]
        self._initialize_model()
    
    def _get_device(self) -> str:
        """
        Determine the device to use for inference.
        
        Returns:
            Device string ('cuda' or 'cpu')
        """
        use_cuda = os.getenv("USE_CUDA", "false").lower() == "true"
        
        if use_cuda and torch.cuda.is_available():
            logger.info("CUDA is available and enabled")
            return "cuda"
        elif use_cuda:
            logger.warning("CUDA was requested but is not available, falling back to CPU")
        
        return "cpu"
    
    def _initialize_model(self) -> None:
        """Initialize the Whisper model on startup."""
        model_name = os.getenv("WHISPER_MODEL", "tiny")
        try:
            self.model = whisper.load_model(model_name, device=self.device)
            self.current_model_name = model_name
            logger.info(f"Loaded Whisper model '{model_name}' on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    async def load_model(self, model_name: str) -> None:
        """
        Load a different Whisper model.
        
        Args:
            model_name: Name of the model to load
        """
        try:
            if model_name != self.current_model_name:
                # Clear CUDA cache if using GPU
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                self.model = whisper.load_model(model_name, device=self.device)
                self.current_model_name = model_name
                logger.info(f"Switched to model '{model_name}'")
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {str(e)}")
            raise
    
    async def transcribe(
        self,
        audio: bytes,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio using the current model.
        
        Args:
            audio: Audio data as bytes
            language: Optional language code
        
        Returns:
            Dict containing transcription results
        """
        try:
            # Prepare transcription options
            options = {}
            if language:
                if language not in self.supported_languages:
                    logger.warning(f"Unsupported language: {language}")
                else:
                    options["language"] = language
            
            # Perform transcription
            result = self.model.transcribe(audio, **options)
            
            # Format response
            response = {
                "text": result["text"],
                "language": result.get("language", "unknown"),
                "segments": [
                    {
                        "text": seg["text"],
                        "start": seg["start"],
                        "end": seg["end"]
                    }
                    for seg in result.get("segments", [])
                ],
                "duration": len(audio) / 16000  # Based on required sample rate
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise RuntimeError(f"Transcription failed: {str(e)}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Resource cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            # Don't raise here as this is cleanup
