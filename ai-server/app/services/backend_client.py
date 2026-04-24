import logging
import httpx
from typing import List, Dict, Any
from app.config import settings
from app.schemas.backend_contracts import DatasetLinkInfo, OnnxUploadLinkResponse

logger = logging.getLogger(__name__)

async def get_dataset_links(job_id: str) -> List[DatasetLinkInfo]:
    """학습 데이터셋 Presigned URL 리스트 요청"""
    logger.info(f"[{job_id}] Requesting dataset links from backend")
    # 실제 구현: httpx.AsyncClient 등을 사용하여 백엔드 호출
    # return [DatasetLinkInfo(file_name="data.csv", download_url="http://presigned-url")]
    return []

async def issue_onnx_upload_link(job_id: str, version_id: str, file_name: str) -> OnnxUploadLinkResponse:
    """모델 업로드용 Presigned URL 요청"""
    logger.info(f"[{job_id}] Requesting ONNX upload link for version {version_id}")
    # return OnnxUploadLinkResponse(upload_url="http://presigned-put-url")
    return OnnxUploadLinkResponse(upload_url="")

async def report_progress(job_id: str, percent: int, status: str, message: str = ""):
    """학습 진행률 및 상태 보고 콜백"""
    logger.info(f"[{job_id}] Reporting progress: {percent}% ({status}) - {message}")
    # 백엔드 POST /callback/.../progress 호출

async def report_metrics(job_id: str, metrics: Dict[str, Any]):
    """평가 지표 보고 콜백"""
    logger.info(f"[{job_id}] Reporting metrics: {metrics}")
    # 백엔드 POST /callback/.../metrics 호출

async def complete_onnx(job_id: str, version_id: str):
    """모델 업로드 완료 보고 콜백"""
    logger.info(f"[{job_id}] Reporting ONNX completion for version {version_id}")
    # 백엔드 POST /callback/.../onnx-complete 호출
