from fastapi import FastAPI, UploadFile, HTTPException, Header, BackgroundTasks, WebSocket
from fastapi.websockets import WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import os
import time
import logging
from pythonjsonlogger import jsonlogger

from transcriber import WhisperTranscriber
from audio import AudioProcessor
from utils import ResourceManager

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
    description="Speech-to-Text API using OpenAI's Whisper model",
    version="1.0.0",
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

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    try:
        # This will trigger model download if needed
        await transcriber.load_model(os.getenv("WHISPER_MODEL", "tiny"))
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

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
            "languages": transcriber.supported_languages
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
        
        # Process audio to match Home Assistant requirements
        audio_data = await audio_processor.process_audio(file)
        
        # Load different model if requested
        if x_model and x_model != transcriber.current_model_name:
            await transcriber.load_model(x_model)
        
        # Transcribe audio
        result = await transcriber.transcribe(
            audio_data,
            language=x_language
        )
        
        # Schedule cleanup
        background_tasks.add_task(audio_processor.cleanup)
        
        # Log success
        duration = time.time() - start_time
        logger.info(
            "Transcription complete",
            extra={
                "duration": duration,
                "model": transcriber.current_model_name,
                "file_size": len(audio_data)
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "stt-stream-failed",
                "message": str(e)
            }
        )

@app.post("/stream")
async def stream_audio(
    background_tasks: BackgroundTasks,
    x_language: Optional[str] = Header(None),
    x_model: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    Stream audio for real-time transcription.
    
    Args:
        x_language: Optional language code (ISO 639-1)
        x_model: Optional model name
    """
    try:
        # Check resources
        await resource_manager.check_resources()
        
        # Load different model if requested
        if x_model and x_model != transcriber.current_model_name:
            await transcriber.load_model(x_model)
        
        # Create WebSocket for streaming
        return {
            "status": "ready",
            "chunk_size": 30,  # milliseconds
            "sample_rate": 16000,
            "channels": 1,
            "format": "pcm_s16le"
        }
        
    except Exception as e:
        logger.error(f"Stream initialization failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "stt-stream-failed",
                "message": str(e)
            }
        )

@app.websocket("/stream/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    background_tasks: BackgroundTasks
):
    """
    WebSocket endpoint for streaming audio data.
    Follows Home Assistant's Assist pipeline event system.
    """
    try:
        await websocket.accept()
        logger.info(f"WebSocket connection accepted: {session_id}")
        
        # Send pipeline initialization event
        await websocket.send_json({
            "type": "run-start"
        })
        
        # Send STT start event
        await websocket.send_json({
            "type": "stt-start"
        })
        
        # Initialize audio buffer
        audio_buffer = bytearray()
        vad_active = False
        
        while True:
            # Receive audio chunk
            chunk = await websocket.receive_bytes()
            
            # Check if it's the end marker
            if len(chunk) == 1:
                logger.info("Received end marker")
                break
                
            # Add to buffer (skip handler ID byte)
            audio_buffer.extend(chunk[1:])
            
            # Process with VAD
            if not vad_active and audio_processor.detect_speech(chunk[1:]):
                vad_active = True
                await websocket.send_json({
                    "type": "stt-vad-start"
                })
            elif vad_active and not audio_processor.detect_speech(chunk[1:]):
                vad_active = False
                await websocket.send_json({
                    "type": "stt-vad-end"
                })
            
            # If we have enough audio and VAD detected speech
            if len(audio_buffer) >= 16000 * 2 and vad_active:  # 2 seconds of audio
                try:
                    # Process audio chunk
                    result = await transcriber.transcribe(bytes(audio_buffer))
                    
                    if result["text"].strip():
                        await websocket.send_json({
                            "type": "stt-end",
                            "stt_output": {
                                "text": result["text"].strip(),
                                "language": result.get("language", "unknown")
                            }
                        })
                    
                    # Clear buffer
                    audio_buffer.clear()
                    
                except Exception as e:
                    logger.error(f"Transcription error: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "code": "stt-stream-failed",
                        "message": str(e)
                    })
        
        # Process any remaining audio
        if audio_buffer:
            try:
                result = await transcriber.transcribe(bytes(audio_buffer))
                if result["text"].strip():
                    await websocket.send_json({
                        "type": "stt-end",
                        "stt_output": {
                            "text": result["text"].strip(),
                            "language": result.get("language", "unknown")
                        }
                    })
            except Exception as e:
                logger.error(f"Final transcription error: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "code": "stt-stream-failed",
                    "message": str(e)
                })
        
        # Send final end event if no text was recognized
        if not audio_buffer or not result["text"].strip():
            await websocket.send_json({
                "type": "error",
                "code": "stt-no-text-recognized",
                "message": "No speech detected"
            })
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "code": "stt-stream-failed",
                "message": str(e)
            })
        except:
            pass
    finally:
        background_tasks.add_task(audio_processor.cleanup)
        resource_manager.release_resources()

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "stt-stream-failed",
            "detail": str(exc)
        }
    )
