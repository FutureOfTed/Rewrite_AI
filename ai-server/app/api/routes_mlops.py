import logging
from fastapi import APIRouter, Header, HTTPException, BackgroundTasks
from app.schemas.backend_contracts import DatasetWebhookPayload
from app.services.training_service import run_training_pipeline
from app.config import settings

router = APIRouter(prefix="/mlops", tags=["MLOps"])
logger = logging.getLogger(__name__)

@router.post("/webhook/dataset-links")
async def receive_dataset_links(
    payload: DatasetWebhookPayload,
    background_tasks: BackgroundTasks,
    authorization: str = Header(None)
):
    """
    백엔드로부터 비동기적으로 데이터셋 링크를 수신하는 Webhook 엔드포인트.
    """
    # 1. 인증 토큰 검증
    expected_token = f"Bearer {settings.MLOPS_CALLBACK_TOKEN}" # 실제 운영시 별도 웹훅 토큰 사용 가능
    if not authorization or authorization != expected_token:
        logger.warning(f"Unauthorized webhook access attempt: {authorization}")
        raise HTTPException(status_code=401, detail="Invalid authorization token")

    logger.info(f"[{payload.job_id}] Webhook received with {len(payload.datasets)} datasets")

    # 2. 백그라운드에서 학습 파이프라인 시작
    background_tasks.add_task(
        run_training_pipeline,
        job_id=payload.job_id,
        version_id=payload.version_id,
        datasets=payload.datasets,
        is_finetune=payload.is_finetune,
        base_model_path=payload.base_model_path
    )

    return {"status": "accepted", "job_id": payload.job_id}
