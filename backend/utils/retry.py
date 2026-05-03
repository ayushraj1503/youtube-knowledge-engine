# backend/utils/retry.py
"""
Retry decorators for both async and sync functions.
Implements exponential backoff — essential for external API calls.
"""

import asyncio
import functools
import time
from typing import Callable, Optional, Tuple, Type

from backend.core.logging_config import get_logger

logger = get_logger(__name__)


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Async retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (including first)
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Which exception types trigger a retry
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            "retry_exhausted",
                            func=func.__name__,
                            attempts=max_attempts,
                            error=str(e),
                        )
                        raise

                    logger.warning(
                        "retry_attempt",
                        func=func.__name__,
                        attempt=attempt,
                        delay=current_delay,
                        error=str(e),
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

        return wrapper

    return decorator


def sync_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """Synchronous retry decorator with exponential backoff."""

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(
                            "retry_exhausted",
                            func=func.__name__,
                            attempts=max_attempts,
                            error=str(e),
                        )
                        raise

                    logger.warning(
                        "sync_retry_attempt",
                        func=func.__name__,
                        attempt=attempt,
                        delay=current_delay,
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper

    return decorator
