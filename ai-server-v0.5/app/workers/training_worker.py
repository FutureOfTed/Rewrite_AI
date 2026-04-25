import logging
import asyncio
from app.schemas.backend_contracts import JobStartRequest
from app.services import training_service

logger = logging.getLogger(__name__)

async def run_training_job(job_id: str, request: JobStartRequest):
    """
    백그라운드에서 실행될 실제 학습 작업 래퍼.
    """
    logger.info(f"Background task started for job_id: {job_id}")
    try:
        # 비동기 환경에서 오케스트레이션 서비스 호출
        await training_service.run_training_pipeline(job_id, request)
    except Exception as e:
        logger.error(f"Unexpected error in training worker for job {job_id}: {e}")
        # training_service 내부에서 FAILED 처리하지 못한 최상위 예외 대응
