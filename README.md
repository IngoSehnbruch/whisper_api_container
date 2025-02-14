# Whisper STT API for Home Assistant

A FastAPI-based service that provides speech-to-text capabilities using OpenAI's Whisper model, designed to work with Home Assistant's Assist pipeline.

## Features

- Speech-to-text transcription using Whisper
- Optional GPU acceleration with CUDA
- Automatic model persistence
- Streaming support with Voice Activity Detection (VAD)
- Full Home Assistant Assist pipeline integration
- Resource management and monitoring
- Health checks and logging
- Docker deployment ready
- Reverse proxy support

## Requirements

- Docker and Docker Compose
- FFmpeg
- 4GB+ RAM (8GB recommended)
- NVIDIA GPU with CUDA support (optional)
- Network access to Home Assistant

## Configuration

### Environment Variables

- `WHISPER_MODEL`: Whisper model to use (default: "tiny")
- `USE_CUDA`: Enable GPU acceleration (default: "false")
- `MAX_MEMORY`: Maximum memory usage in MB (default: 8192)
- `MAX_CONCURRENT`: Maximum concurrent requests (default: 5)
- `LOG_LEVEL`: Logging level (default: "INFO")
- `TRUSTED_PROXIES`: Comma-separated list of trusted proxy hosts (optional)
- `ROOT_PATH`: Root path when behind a proxy (optional)

## GPU Support

To enable GPU acceleration:

1. Set `USE_CUDA=true` in environment variables
2. Uncomment the GPU section in `docker-compose.yml`
3. Ensure NVIDIA Container Toolkit is installed
4. Restart the container

The service will automatically fall back to CPU if CUDA is not available.

## Model Persistence

Models are automatically downloaded on first use and persisted in a Docker volume. 

## API Endpoints

### Health Check
```
GET /health
Response: {
    "status": "healthy",
    "model_loaded": true,
    "model": "tiny",
    "memory_usage_mb": float,
    "languages": ["en", "es", ...]
}
```

### File Transcription
```
POST /transcribe
Headers:
  - X-Language: ISO 639-1 language code (optional)
  - X-Model: Model name (optional)

Body: 
  - Multipart form with audio file

Response: {
    "text": "transcribed text",
    "language": "detected language",
    "segments": [
        {
            "text": "segment text",
            "start": float,
            "end": float
        }
    ],
    "duration": float
}
```

### Streaming
```
POST /stream
Response: {
    "status": "ready",
    "chunk_size": 30,  # milliseconds
    "sample_rate": 16000,
    "channels": 1,
    "format": "pcm_s16le"
}

WebSocket /stream/{session_id}
Events:
1. run-start: Pipeline initialization
   {
     "type": "run-start"
   }

2. stt-start: Begin speech recognition
   {
     "type": "stt-start"
   }

3. stt-vad-start: Voice activity detected
   {
     "type": "stt-vad-start"
   }

4. stt-vad-end: Voice activity ended
   {
     "type": "stt-vad-end"
   }

5. stt-end: Recognition completed
   {
     "type": "stt-end",
     "stt_output": {
       "text": "transcribed text",
       "language": "detected language"
     }
   }

6. error: Error occurred
   {
     "type": "error",
     "code": "error-code",
     "message": "error message"
   }

Binary Audio Format:
[stt_binary_handler_id (1 byte)][audio_chunk_data]
```

## Deployment

1. Clone the repository
2. Configure environment variables
3. Start the container:
   ```bash
   docker-compose up -d
   ```

### Behind a Reverse Proxy

1. Set `TRUSTED_PROXIES` to your proxy's hostname
2. Set `ROOT_PATH` if the API is not at the root path
3. The service will automatically handle proxy headers

## Audio Requirements

- Sample Rate: 16kHz
- Channels: Mono
- Bit Depth: 16-bit
- Format: Any format supported by FFmpeg (will be converted automatically)

## Error Codes

- `stt-provider-missing`: STT provider not available
- `stt-provider-unsupported-metadata`: Unsupported audio format
- `stt-stream-failed`: Processing error
- `stt-no-text-recognized`: No transcript produced

## Security

- Rate limiting through concurrent request limits
- Resource usage monitoring
- Temporary file cleanup
- Input validation
- Proxy validation
- Non-root container user

## License

MIT License
