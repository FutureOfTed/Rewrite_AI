from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

# 백엔드 연동용 요청/응답 스키마

class JobStartRequest(BaseModel):
    job_id: Optional[str] = Field(alias="jobId", default=None)
    candidate_version_id: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    dataset_prefix: Optional[str] = None
    requested_by: Optional[str] = None
    requested_at: Optional[str] = None

    class Config:
        populate_by_name = True

class DatasetLinkInfo(BaseModel):
    s3_key: str = Field(alias="s3Key")
    download_url: str = Field(alias="downloadUrl")
    expires_at: Optional[datetime] = Field(alias="expiresAt", default=None)

    class Config:
        populate_by_name = True

class OnnxUploadLinkResponse(BaseModel):
    job_id: str = Field(alias="jobId")
    version_id: str = Field(alias="versionId")
    s3_key: str = Field(alias="s3Key")
    upload_url: str = Field(alias="uploadUrl")
    expires_at: datetime = Field(alias="expiresAt")

    class Config:
        populate_by_name = True

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
    job_id: str = Field(alias="jobId")
    page: int = 0
    size: int = 0
    total_elements: int = Field(alias="totalElements", default=0)
    total_pages: int = Field(alias="totalPages", default=0)
    links: list[DatasetLinkInfo]
    version_id: Optional[str] = None
    is_finetune: bool = False
    base_model_path: Optional[str] = None

    class Config:
        populate_by_name = True