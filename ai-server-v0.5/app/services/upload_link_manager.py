import asyncio
import logging
from typing import Dict
from app.schemas.backend_contracts import OnnxUploadLinkResponse

logger = logging.getLogger(__name__)

# job_id 별로 업로드 링크 응답을 기다리는 Future 저장소
_pending_links: Dict[str, asyncio.Future] = {}
_lock = asyncio.Lock()

async def get_link_future(job_id: str) -> asyncio.Future:
    """해당 job_id에 대한 Future 객체를 가져오거나 새로 생성합니다."""
    async with _lock:
        if job_id not in _pending_links:
            _pending_links[job_id] = asyncio.Future()
        return _pending_links[job_id]

async def set_link_result(job_id: str, link_info: OnnxUploadLinkResponse):
    """웹훅을 통해 받은 링크 정보를 Future에 채워 넣습니다."""
    async with _lock:
        if job_id not in _pending_links:
            _pending_links[job_id] = asyncio.Future()
        
        fut = _pending_links[job_id]
        if not fut.done():
            fut.set_result(link_info)
            logger.info(f"[LinkManager] Link set for job_id: {job_id}")

async def wait_for_link(job_id: str, timeout: int = 300) -> OnnxUploadLinkResponse:
    """링크가 도착할 때까지 대기합니다. (기본 5분)"""
    fut = await get_link_future(job_id)
    try:
        return await asyncio.wait_for(fut, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"[LinkManager] Timeout waiting for link: {job_id}")
        raise
    finally:
        async with _lock:
            if job_id in _pending_links:
                del _pending_links[job_id]
