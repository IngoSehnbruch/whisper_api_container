import asyncio
import websockets
import requests
import json
import time
import wave
import numpy as np
from pathlib import Path

API_URL = "http://localhost:8000"
SAMPLE_RATE = 16000

def create_test_audio(filename: str, duration: float = 3.0):
    """Create a test WAV file with a simple tone."""
    # Generate a 440 Hz sine wave
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 32767
    audio = audio.astype(np.int16)
    
    # Save as WAV
    with wave.open(filename, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio.tobytes())

async def test_health():
    """Test the health check endpoint."""
    print("\nTesting health check...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200

async def test_transcribe():
    """Test the transcribe endpoint."""
    print("\nTesting transcribe endpoint...")
    
    # Create test audio file
    test_file = "test_audio.wav"
    create_test_audio(test_file)
    
    try:
        with open(test_file, 'rb') as f:
            files = {'audio': ('test.wav', f, 'audio/wav')}
            headers = {
                'X-Language': 'en',
            }
            response = requests.post(
                f"{API_URL}/transcribe",
                files=files,
                headers=headers
            )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 200
        
    finally:
        Path(test_file).unlink(missing_ok=True)

async def test_streaming():
    """Test the streaming endpoint."""
    print("\nTesting streaming endpoint...")
    
    # Initialize streaming
    response = requests.post(
        f"{API_URL}/stream",
        headers={'X-Language': 'en'}
    )
    print(f"Stream init status: {response.status_code}")
    print(f"Stream init response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    
    # Create test audio
    test_file = "test_stream.wav"
    create_test_audio(test_file, duration=1.0)
    
    try:
        # Read audio data
        with open(test_file, 'rb') as f:
            audio_data = f.read()
        
        # Stream the audio
        ws_url = f"ws://localhost:8000/stream/test-session"
        print(f"Connecting to WebSocket: {ws_url}")
        async with websockets.connect(
            ws_url,
            extra_headers={
                'X-Language': 'en'
            }
        ) as websocket:
            # Wait for initial events
            response = await websocket.recv()
            event = json.loads(response)
            assert event["type"] == "run-start", "Expected run-start event"
            print("Received run-start event")
            
            response = await websocket.recv()
            event = json.loads(response)
            assert event["type"] == "stt-start", "Expected stt-start event"
            print("Received stt-start event")
            
            # Send audio in chunks
            chunk_size = 3200  # 100ms chunks
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                # Add handler ID byte
                chunk = bytes([1]) + chunk
                await websocket.send(chunk)
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=1.0
                    )
                    event = json.loads(response)
                    print(f"Received event: {event['type']}")
                    
                    if event["type"] == "error":
                        print(f"Error: {event['code']} - {event['message']}")
                    elif event["type"] == "stt-end":
                        print(f"Transcription: {event['stt_output']['text']}")
                        
                except asyncio.TimeoutError:
                    pass
            
            # Send end marker
            await websocket.send(bytes([1]))
            
            # Get final response
            try:
                final_response = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=5.0
                )
                event = json.loads(final_response)
                print(f"Final event: {event['type']}")
                if event["type"] == "stt-end":
                    print(f"Final transcription: {event['stt_output']['text']}")
                elif event["type"] == "error":
                    print(f"Final error: {event['code']} - {event['message']}")
            except asyncio.TimeoutError:
                print("No final response received")
    
    finally:
        Path(test_file).unlink(missing_ok=True)

async def main():
    """Run all tests."""
    print("Starting API tests...")
    
    try:
        await test_health()
        await test_transcribe()
        await test_streaming()
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
