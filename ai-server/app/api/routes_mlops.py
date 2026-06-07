import logging
from typing import Any

from fastapi import APIRouter, Header, HTTPException, BackgroundTasks, Request
from app.schemas.backend_contracts import DatasetWebhookPayload, OnnxUploadLinkResponse, JobStartRequest
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

    # 2. version_id 결정: 백엔드가 보내준 값을 최우선으로 사용
    version_id = parsed.version_id or parsed.job_id
    logger.info(f"[{parsed.job_id}] Using version_id: {version_id}")

    # 3. 백그라운드에서 학습 파이프라인 시작

    background_tasks.add_task(
        run_training_pipeline,
        job_id=parsed.job_id,
        version_id=version_id,
        datasets=parsed.links,
        is_finetune=parsed.is_finetune,
        base_model_path=parsed.base_model_path
    )

    return {"status": "accepted", "job_id": parsed.job_id}

@router.post("/webhook/onnx-upload-link")
@router.post("/mlops/webhook/onnx-upload-link")
@router.post("/api/v1/mlops/webhook/onnx-upload-link")
async def receive_onnx_upload_link(request: Request, background_tasks: BackgroundTasks):
    """
    대시보드에서 ONNX 업로드 Presigned URL을 웹훅으로 보내주면:
    1. best_weights.pt → ONNX 변환
    2. S3 업로드
    3. onnx-complete 보고
    를 백그라운드에서 수행합니다.
    """
    payload = await request.json()
    logger.info(f"Received ONNX upload link webhook: {payload}")

    from datetime import datetime, timezone

    job_id     = payload.get("jobId")     or payload.get("job_id")
    upload_url = payload.get("uploadUrl") or payload.get("upload_url")
    s3_key     = payload.get("s3Key")     or payload.get("s3_key")
    version_id = payload.get("versionId") or payload.get("version_id") or "unknown"
    expires_at = payload.get("expiresAt") or payload.get("expires_at") or datetime.utcnow().isoformat() + "Z"

    if not job_id or not upload_url:
        raise HTTPException(status_code=422, detail="Missing jobId or uploadUrl in payload")

    # 502 Error: 접근 권한 만료 검증 (Deployment Control)
    try:
        exp_time = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        if datetime.now(timezone.utc) > exp_time:
            logger.warning(f"[{job_id}] ONNX Upload token expired (502). Expires at: {expires_at}")
            raise HTTPException(
                status_code=502, 
                detail="[에러 코드 502] 접근 권한 토큰이 만료되었습니다. 배포 작업을 보류합니다. 재로그인 해주세요."
            )
    except ValueError:
        pass  # 시간 포맷 오류 시 예외 처리 통과

    upload_info = OnnxUploadLinkResponse(
        jobId=job_id,
        versionId=version_id,
        uploadUrl=upload_url,
        s3Key=s3_key or f"mlops/models/candidate/{job_id}/model.onnx",
        expiresAt=expires_at
    )

    # 웹훅 수신 즉시 백그라운드에서 ONNX 변환 + 업로드 실행
    from app.services.training_service import run_onnx_export_and_upload
    background_tasks.add_task(
        run_onnx_export_and_upload,
        job_id=job_id,
        version_id=version_id,
        upload_info=upload_info,
    )

    return {"status": "accepted", "job_id": job_id}



@router.post("/jobs/{job_id}/train")
async def start_training(job_id: str, request: JobStartRequest, background_tasks: BackgroundTasks):
    """
    Pull 방식으로 백엔드에서 dataset links를 가져와 학습을 시작합니다.
    """
    datasets = await backend_client.get_dataset_links(job_id)
    if not datasets:
        raise HTTPException(status_code=400, detail="No dataset links returned for this job")

    # 백엔드가 지정해준 candidate_version_id가 있으면 최우선으로 사용, 없으면 fallback
    version_id = request.candidate_version_id or f"candidate_{job_id}"
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
            "version_id": payload.get("candidate_version_id") or payload.get("version_id"),
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
            "version_id": payload.get("candidate_version_id") or payload.get("version_id") or payload.get("versionId"),
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


@router.post("/rollback")
@router.post("/api/v1/mlops/rollback")
async def rollback_model_version(request: Request):
    """
    모델 버전 롤백 (Rollback)
    신규 배포된 모델에 문제가 발견되거나 성능이 기대 이하일 경우 이전 안정적인 버전으로 교체합니다.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    
    target_version = body.get("target_version", "stable")
    logger.info(f"모델 버전 롤백 승인 요청됨. Target: {target_version}")
    
    # AI 서버에서는 실제 서빙을 하지 않으므로 롤백 상태만 ACK 처리
    return {
        "status": "success", 
        "message": f"모델 버전 롤백 처리가 완료되었습니다. (Target: {target_version})",
        "action": "rollback"
    }


@router.get("/fail-safe-rules")
@router.get("/api/v1/mlops/fail-safe-rules")
async def get_fail_safe_rules():
    """
    강제 예외 규칙(Fail-Safe Ruleset) 정의
    배포된 모델이 클라이언트 기기에서 구동될 때 발생하는 문제에 대한 해결책(Inference Fallback)을 제공합니다.
    """
    return {
        "ruleset": "fallback_to_alpha_1",
        "conditions": {
            "max_inference_time_ms": 150,
            "error_codes_to_fallback": [403],
            "trigger_events": ["memory_error", "tensor_error", "infrastructure_failure", "timeout"]
        },
        "fallback_action": {
            "difficulty_alpha": 1.0,
            "message": "게임 플레이 연속성을 위해 난이도 보정 계수를 기본값으로 복구합니다."
        }
    }