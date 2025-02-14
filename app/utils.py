import os
import psutil
import logging
from fastapi import HTTPException

logger = logging.getLogger("whisper-api")

class ResourceManager:
    """Manages system resources and limits."""
    
    def __init__(self):
        """Initialize the resource manager."""
        self.max_memory = int(os.getenv("MAX_MEMORY", "8192"))  # MB
        self.max_concurrent = int(os.getenv("MAX_CONCURRENT", "5"))
        self._current_requests = 0
    
    def get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Current memory usage in MB
        """
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024
        except Exception as e:
            logger.error(f"Failed to get memory usage: {str(e)}")
            return 0.0
    
    async def check_resources(self) -> None:
        """
        Check if enough resources are available.
        
        Raises:
            HTTPException if resources are exhausted
        """
        try:
            # Check memory usage
            memory_usage = self.get_memory_usage()
            if memory_usage > self.max_memory:
                logger.warning(f"Memory limit exceeded: {memory_usage}MB/{self.max_memory}MB")
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "stt-stream-failed",
                        "message": "Server is overloaded"
                    }
                )
            
            # Check concurrent requests
            if self._current_requests >= self.max_concurrent:
                logger.warning(f"Too many concurrent requests: {self._current_requests}")
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "stt-stream-failed",
                        "message": "Too many concurrent requests"
                    }
                )
            
            self._current_requests += 1
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Resource check failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "stt-stream-failed",
                    "message": "Resource check failed"
                }
            )
    
    def release_resources(self) -> None:
        """Release resources after request completion."""
        if self._current_requests > 0:
            self._current_requests -= 1
