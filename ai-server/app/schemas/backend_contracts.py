from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# 백엔드 연동용 요청/응답 스키마

class JobStartRequest(BaseModel):
    date_from: str
    date_to: str
    dataset_prefix: str
    # 필요한 파라미터 추가

class DatasetLinkInfo(BaseModel):
    file_name: str
    download_url: str  # Presigned GET URL

class OnnxUploadLinkResponse(BaseModel):
    upload_url: str    # Presigned PUT URL

class ProgressReportRequest(BaseModel):
    percent: int
    status: str  # e.g., "IN_PROGRESS", "FAILED"
    message: Optional[str] = None

class MetricsReportRequest(BaseModel):
    rmse: float
    f1_score: float
    # 추가 지표

class CompleteOnnxRequest(BaseModel):
    version_id: str
    # 추가 메타데이터

class DatasetWebhookPayload(BaseModel):
    job_id: str
    version_id: str
    datasets: List[DatasetLinkInfo]
    is_finetune: bool = False
    base_model_path: Optional[str] = None
