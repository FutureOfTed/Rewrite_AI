import logging
from app.schemas.backend_contracts import JobStartRequest
from app.services import backend_client, presigned_transfer_service

logger = logging.getLogger(__name__)

async def run_training_pipeline(job_id: str, request: JobStartRequest):
    """
    학습 오케스트레이션: 데이터 다운로드 -> 학습 -> 평가 -> 모델 변환 -> 업로드
    """
    try:
        # 1. 진행 상태: 학습 시작 준비 (10%)
        await backend_client.report_progress(job_id, 10, "IN_PROGRESS", "Starting training pipeline")

        # 2. 백엔드로부터 데이터셋 다운로드 링크 요청
        dataset_links = await backend_client.get_dataset_links(job_id)
        
        # 3. Presigned GET URL로 데이터 다운로드
        for link_info in dataset_links:
            await presigned_transfer_service.download_file(
                url=link_info.download_url,
                dest_path=f"/tmp/{link_info.file_name}"
            )
        
        # 4. 진행 상태: 다운로드 완료, 학습 시작 (30%)
        await backend_client.report_progress(job_id, 30, "IN_PROGRESS", "Data downloaded, starting training")
        
        # 5. 파이프라인(전처리->학습->평가) 실행 (가정)
        # metrics, onnx_path = train_pipeline.execute(...)
        metrics = {"rmse": 0.5, "f1_score": 0.85}
        
        # 6. 진행 상태: 평가 완료 (60%)
        await backend_client.report_progress(job_id, 60, "IN_PROGRESS", "Training and evaluation completed")
        await backend_client.report_metrics(job_id, metrics)
        
        # 7. 백엔드로부터 모델 업로드용 Presigned PUT 링크 요청
        upload_link_info = await backend_client.issue_onnx_upload_link(job_id, "v1", "model.onnx")
        
        # 8. Presigned PUT URL로 ONNX 파일 업로드
        # await presigned_transfer_service.upload_file(
        #     url=upload_link_info.upload_url,
        #     file_path=onnx_path
        # )
        
        # 9. 완료 처리: onnx-complete 콜백 전송 (100%)
        await backend_client.complete_onnx(job_id, "v1")
        await backend_client.report_progress(job_id, 100, "COMPLETED", "Pipeline successfully finished")
        
    except Exception as e:
        logger.error(f"Training pipeline failed for job {job_id}: {e}")
        # 실패 시 FAILED 상태 콜백
        await backend_client.report_progress(job_id, 0, "FAILED", str(e))
