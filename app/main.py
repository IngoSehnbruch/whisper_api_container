"""Whisper STT API server."""
from fastapi import FastAPI, UploadFile, HTTPException, Header, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import os
import time
import logging
from pythonjsonlogger import jsonlogger

from transcriber import WhisperTranscriber
from audio import AudioProcessor
from utils import ResourceManager
from wyoming_faster_whisper import WhisperWyomingServer

# Configure logging
logger = logging.getLogger("whisper-api")
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    '%(timestamp)s %(level)s %(name)s %(message)s',
    timestamp=True
)
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Initialize FastAPI with proxy support
trusted_proxies = os.getenv("TRUSTED_PROXIES", "").split(",")
app = FastAPI(
    title="Whisper STT API",
    description="Speech-to-Text API using OpenAI's Whisper model with Wyoming protocol support",
    version="2.0.0",
    root_path=os.getenv("ROOT_PATH", ""),
    servers=[{"url": "/", "description": "Current server"}]
)

if trusted_proxies and trusted_proxies[0]:
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_proxies)

# Initialize components
resource_manager = ResourceManager()
audio_processor = AudioProcessor()
transcriber = WhisperTranscriber()

# Initialize Wyoming server
wyoming_server = WhisperWyomingServer(
    host=os.getenv("WYOMING_HOST", "0.0.0.0"),
    port=int(os.getenv("WYOMING_PORT", "10300")),
    transcriber=transcriber,
    audio_processor=audio_processor,
    resource_manager=resource_manager,
)

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    try:
        # This will trigger model download if needed
        await transcriber.load_model(os.getenv("WHISPER_MODEL", "tiny"))
        # Start Wyoming server
        await wyoming_server.start()
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    try:
        await wyoming_server.stop()
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        memory_usage = resource_manager.get_memory_usage()
        return {
            "status": "healthy",
            "model_loaded": bool(transcriber.current_model),
            "model": transcriber.current_model_name,
            "memory_usage_mb": memory_usage,
            "languages": transcriber.supported_languages,
            "wyoming_server": {
                "host": wyoming_server.host,
                "port": wyoming_server.port,
                "running": wyoming_server.server is not None
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    x_language: Optional[str] = Header(None),
    x_model: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    Transcribe audio file using Whisper.
    
    Args:
        file: Audio file
        x_language: Optional language code (ISO 639-1)
        x_model: Optional model name
    """
    start_time = time.time()
    
    try:
        # Check resources
        await resource_manager.check_resources()
        
        # Process audio to match requirements
        audio_data = await audio_processor.process_audio(file)
        
        # Load different model if requested
        if x_model and x_model != transcriber.current_model_name:
            await transcriber.load_model(x_model)
        
        # Transcribe audio
        result = await transcriber.transcribe(
            audio_data,
            language=x_language
        )
        
        # Add timing info
        result["processing_time"] = time.time() - start_time
        
        return result
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )
