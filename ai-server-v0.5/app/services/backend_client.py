import logging
import httpx
from typing import List, Dict, Any
from app.config import settings
from app.schemas.backend_contracts import DatasetLinkInfo, OnnxUploadLinkResponse

logger = logging.getLogger(__name__)

# ── 1. 인증 헤더 팩토리 ───────────────────────────────────────────────
# 각 요청마다 최신 설정을 반영하기 위해 함수 내부에서 생성하거나 
# 필요 시 호출하는 방식을 사용합니다.

def get_pull_headers():
    return {
        "Authorization": f"Bearer {settings.MLOPS_PULL_TOKEN}",
        "Content-Type": "application/json"
    }

def get_callback_headers():
    return {
        "Authorization": f"Bearer {settings.MLOPS_CALLBACK_TOKEN}",
        "Content-Type": "application/json"
    }

# ── 2. 백엔드 연동 메서드 ──────────────────────────────────────────────

async def get_dataset_links(job_id: str) -> List[DatasetLinkInfo]:
    """
    학습 데이터셋 Presigned URL 리스트 요청 (Pull 방식)
    응답 구조: { "links": [...], "totalElements": ... }
    """
    url = f"{settings.BACKEND_BASE_URL}/api/v1/mlops/pull/jobs/{job_id}/dataset-links"
    headers = get_pull_headers()
    all_links: List[DatasetLinkInfo] = []
    page = 0
    size = 200

    timeout = httpx.Timeout(settings.HTTP_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            payload = {"page": page, "size": size}
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            links_data = data.get("links", [])
            all_links.extend([DatasetLinkInfo(**item) for item in links_data])

            total_pages = int(data.get("totalPages", 0))
            current_page = int(data.get("page", page))
            if total_pages <= 0 or current_page + 1 >= total_pages:
                break
            page += 1

    logger.info("[%s] fetched dataset links: %s", job_id, len(all_links))
    return all_links

async def issue_onnx_upload_link(job_id: str, version_id: str) -> OnnxUploadLinkResponse:
    """모델 업로드용 Presigned URL 요청 (Pull 방식)"""
    url = f"{settings.BACKEND_BASE_URL}/api/v1/mlops/pull/jobs/{job_id}/onnx-upload-link"
    headers = get_pull_headers()
    payload = {"version_id": version_id}
    
    timeout = httpx.Timeout(settings.HTTP_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return OnnxUploadLinkResponse(**response.json())

async def report_progress(job_id: str, percent: int, status: str, message: str = ""):
    """학습 진행률 및 상태 보고 콜백 (PATCH)"""
    url = f"{settings.BACKEND_BASE_URL}/api/v1/mlops/callback/jobs/{job_id}/progress"
    headers = get_callback_headers()
    payload = {
        "status": status,
        "progress_percent": percent,
        "progress_message": message
    }
    timeout = httpx.Timeout(settings.HTTP_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.patch(url, headers=headers, json=payload)
        response.raise_for_status()

async def report_metrics(job_id: str, metrics: Dict[str, Any]):
    """평가 지표 보고 콜백 (POST)"""
    url = f"{settings.BACKEND_BASE_URL}/api/v1/mlops/callback/jobs/{job_id}/metrics"
    headers = get_callback_headers()
    timeout = httpx.Timeout(settings.HTTP_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=metrics)
        response.raise_for_status()

async def complete_onnx(job_id: str, payload: Dict[str, Any]):
    """모델 업로드 완료 보고 콜백 (POST)"""
    url = f"{settings.BACKEND_BASE_URL}/api/v1/mlops/callback/jobs/{job_id}/onnx-complete"
    headers = get_callback_headers()
    timeout = httpx.Timeout(settings.HTTP_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
