# Whisper API Server

A high-performance API server for OpenAI's Whisper speech recognition model, designed for Home Assistant integration.

## Features

- Wyoming protocol support for seamless Home Assistant integration
- Real-time speech-to-text transcription
- Multi-language support
- Dynamic model switching
- GPU acceleration (when available)
- RESTful API endpoints
- Health monitoring
- Persistent model cache
- Secure by default

## API Endpoints

### REST API (Port 8000)

- `GET /health`: System health check and status
- `POST /transcribe`: Simple file-based transcription
  - Accepts audio file upload
  - Headers:
    - `x-language`: Optional language code
    - `x-model`: Optional model name

### Wyoming Protocol (Port 10300)

Primary interface for Home Assistant integration:
- `Info/AsrInfo`: Get server capabilities
- `Transcribe`: Start transcription session
- `AudioStart/AudioChunk/AudioStop`: Stream audio data
- `Transcript`: Get transcription result
- `Error`: Error information

## Quick Start

### Using Docker (Recommended)

1. Start the server:
```bash
docker-compose up -d
```

2. Check the status:
```bash
curl http://localhost:8000/health
```

### Manual Installation

1. Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# Windows
# Download and install ffmpeg from https://ffmpeg.org/download.html
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WHISPER_MODEL` | Model to use (tiny/base/small/medium/large-v3) | tiny |
| `MAX_MEMORY` | Maximum memory usage in MB | 8192 |
| `MAX_CONCURRENT` | Maximum concurrent transcriptions | 5 |
| `USE_CUDA` | Enable GPU acceleration | false |
| `LOG_LEVEL` | Logging level | INFO |
| `WYOMING_HOST` | Wyoming protocol host | 0.0.0.0 |
| `WYOMING_PORT` | Wyoming protocol port | 10300 |

### Available Models

| Model | Size | Memory | Speed | Accuracy |
|-------|------|---------|--------|-----------|
| tiny | 75MB | 1GB | Fastest | Good |
| base | 150MB | 1GB | Fast | Better |
| small | 500MB | 2GB | Medium | Great |
| medium | 1.5GB | 4GB | Slow | Excellent |
| large-v3 | 3GB | 8GB | Slowest | Best |

## Audio Format

- Sample rate: 16kHz
- Bit depth: 16-bit
- Channels: Mono
- Format: PCM WAV or raw audio

## Performance Tuning

### Memory Usage

- Set `MAX_MEMORY` based on available system RAM
- Consider using smaller models for limited resources
- Monitor memory usage with health endpoint

### GPU Acceleration

1. Install CUDA requirements:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. Enable GPU support:
```bash
export USE_CUDA=true
```

3. Uncomment GPU section in docker-compose.yml

## Development

### Running Tests

```bash
pytest
```

### Building Docker Image

```bash
docker build -t whisper-api .
```

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce `MAX_MEMORY`
   - Use smaller model
   - Check system resources

2. **GPU Issues**
   - Verify CUDA installation
   - Check GPU compatibility
   - Monitor GPU memory

3. **Connection Issues**
   - Check ports (8000, 10300)
   - Verify network settings
   - Check firewall rules

## Security

- Non-root user in Docker
- No exposed credentials
- Regular security updates
- Input validation
- Resource limits

## License

MIT License

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Wyoming Protocol](https://github.com/rhasspy/wyoming)
- [FastAPI](https://fastapi.tiangolo.com/)
