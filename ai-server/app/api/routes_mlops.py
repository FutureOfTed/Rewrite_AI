from fastapi import APIRouter, BackgroundTasks, status
from app.schemas.backend_contracts import JobStartRequest
from app.workers.training_worker import run_training_job

router = APIRouter()

@router.post("/jobs/{job_id}/start", status_code=status.HTTP_202_ACCEPTED)
async def start_job(job_id: str, request: JobStartRequest, background_tasks: BackgroundTasks):
    # 비동기 task 큐에 등록 후 즉시 accepted 반환
    background_tasks.add_task(run_training_job, job_id, request)
    return {"message": "Job accepted"}

@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    # (선택) 작업 상태 조회 (디버깅용)
    return {"job_id": job_id, "status": "PENDING_OR_RUNNING"}
