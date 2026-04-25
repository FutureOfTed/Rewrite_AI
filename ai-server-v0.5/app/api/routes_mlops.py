import logging
from typing import Any

from fastapi import APIRouter, Header, HTTPException, BackgroundTasks, Request
from app.schemas.backend_contracts import DatasetWebhookPayload
from app.services import backend_client
from app.services.training_service import run_training_pipeline
from app.config import settings

router = APIRouter(tags=["MLOps"])
logger = logging.getLogger(__name__)

@router.post("/webhook/dataset-links")
@router.post("/api/v1/mlops/webhook/dataset-links")
async def receive_dataset_links(
    request: Request,
    background_tasks: BackgroundTasks,
    authorization: str = Header(None)
):
    """
    백엔드로부터 비동기적으로 데이터셋 링크를 수신하는 Webhook 엔드포인트.
    """
    # 1. 인증 토큰 검증
    webhook_token = settings.MLOPS_WEBHOOK_TOKEN or settings.MLOPS_CALLBACK_TOKEN
    expected_token = f"Bearer {webhook_token}"
    if not authorization or authorization != expected_token:
        logger.warning(f"Unauthorized webhook access attempt: {authorization}")
        raise HTTPException(status_code=401, detail="Invalid authorization token")

    payload = await _read_json_body(request)
    parsed = _parse_webhook_payload(payload)
    logger.info(f"[{parsed.job_id}] Webhook received with {len(parsed.links)} datasets")

    # 2. 백그라운드에서 학습 파이프라인 시작
    version_id = parsed.version_id or f"candidate_{parsed.job_id}"
    background_tasks.add_task(
        run_training_pipeline,
        job_id=parsed.job_id,
        version_id=version_id,
        datasets=parsed.links,
        is_finetune=parsed.is_finetune,
        base_model_path=parsed.base_model_path
    )

    return {"status": "accepted", "job_id": parsed.job_id}


@router.post("/jobs/{job_id}/train")
async def start_training(job_id: str, background_tasks: BackgroundTasks):
    """
    Pull 방식으로 백엔드에서 dataset links를 가져와 학습을 시작합니다.
    """
    datasets = await backend_client.get_dataset_links(job_id)
    if not datasets:
        raise HTTPException(status_code=400, detail="No dataset links returned for this job")

    version_id = f"candidate_{job_id}"
    background_tasks.add_task(
        run_training_pipeline,
        job_id=job_id,
        version_id=version_id,
        datasets=datasets,
        is_finetune=False,
        base_model_path=None,
    )
    return {"status": "accepted", "job_id": job_id, "dataset_count": len(datasets), "version_id": version_id}


def _parse_webhook_payload(payload: dict[str, Any]) -> DatasetWebhookPayload:
    """
    Accept both current backend shape and legacy teammate shape.
    """
    if "datasets" in payload and "links" not in payload:
        normalized_links = []
        for item in payload.get("datasets", []):
            s3_key = item.get("s3Key") or item.get("s3_key") or item.get("file_name") or "unknown.json"
            normalized_links.append(
                {
                    "s3Key": s3_key,
                    "downloadUrl": item.get("downloadUrl") or item.get("download_url"),
                    "expiresAt": item.get("expiresAt"),
                }
            )
        payload = {
            "jobId": payload.get("jobId") or payload.get("job_id"),
            "version_id": payload.get("version_id"),
            "is_finetune": payload.get("is_finetune", False),
            "base_model_path": payload.get("base_model_path"),
            "links": normalized_links,
            "page": payload.get("page", 0),
            "size": payload.get("size", len(normalized_links)),
            "totalElements": payload.get("totalElements", len(normalized_links)),
            "totalPages": payload.get("totalPages", 1),
        }
    elif "links" in payload:
        normalized_links = []
        for item in payload.get("links", []):
            s3_key = item.get("s3Key") or item.get("s3_key") or item.get("file_name") or "unknown.json"
            normalized_links.append(
                {
                    "s3Key": s3_key,
                    "downloadUrl": item.get("downloadUrl") or item.get("download_url"),
                    "expiresAt": item.get("expiresAt") or item.get("expires_at"),
                }
            )
        payload = {
            "jobId": payload.get("jobId") or payload.get("job_id"),
            "version_id": payload.get("version_id") or payload.get("versionId"),
            "is_finetune": payload.get("is_finetune", False),
            "base_model_path": payload.get("base_model_path"),
            "links": normalized_links,
            "page": payload.get("page", 0),
            "size": payload.get("size", len(normalized_links)),
            "totalElements": payload.get("totalElements") or payload.get("total_elements", len(normalized_links)),
            "totalPages": payload.get("totalPages") or payload.get("total_pages", 1),
        }
    try:
        return DatasetWebhookPayload(**payload)
    except Exception as e:
        logger.error("Invalid webhook payload: %s | payload=%s", e, payload)
        raise HTTPException(status_code=422, detail=f"Invalid webhook payload: {e}")


async def _read_json_body(request: Request) -> dict[str, Any]:
    headers = dict(request.headers)
    try:
        body_bytes = await request.body()
        if not body_bytes:
            logger.error(f"Webhook received EMPTY body. Headers: {headers}")
            raise HTTPException(status_code=400, detail="Empty request body")
        body = await request.json()
    except Exception as e:
        logger.error(f"Webhook JSON parse failed: {e} | Headers: {headers}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {str(e)}")
    if not isinstance(body, dict):
        logger.error(f"Webhook payload is not object: type={type(body)} body={body}")
        raise HTTPException(status_code=422, detail="Webhook payload must be a JSON object")
    return body