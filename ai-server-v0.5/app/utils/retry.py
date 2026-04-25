import asyncio
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def with_retry(max_retries=3, base_delay=1.0):
    """
    지수적 백오프(exponential backoff)를 지원하는 비동기 재시도 데코레이터
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Function {func.__name__} failed after {max_retries} attempts. Error: {e}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
        return wrapper
    return decorator
