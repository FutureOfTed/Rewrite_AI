import logging
from typing import List, Optional
from app.schemas.backend_contracts import DatasetLinkInfo
from app.services import training_service

logger = logging.getLogger(__name__)

async def run_training_job(
    job_id: str,
    version_id: str,
    datasets: List[DatasetLinkInfo],
    is_finetune: bool = False,
    base_model_path: Optional[str] = None,
):
    """
    백그라운드에서 실행될 실제 학습 작업 래퍼.
    routes_mlops.py의 BackgroundTasks에서 호출됩니다.
    """
    logger.info(f"[{job_id}] Background training worker started (version={version_id})")
    try:
        await training_service.run_training_pipeline(
            job_id=job_id,
            version_id=version_id,
            datasets=datasets,
            is_finetune=is_finetune,
            base_model_path=base_model_path,
        )
    except Exception as e:
        logger.error(f"[{job_id}] Unexpected top-level error in training worker: {e}")
        # training_service 내부에서 FAILED 처리 못 한 최상위 예외 대응
