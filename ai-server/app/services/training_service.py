import logging
import os
import json
from typing import List
from app.schemas.backend_contracts import DatasetLinkInfo
from app.services import backend_client, presigned_transfer_service
from app.pipelines import train_pipeline

logger = logging.getLogger(__name__)

async def run_training_pipeline(
    job_id: str,
    version_id: str,
    datasets: List[DatasetLinkInfo],
    is_finetune: bool = False,
    base_model_path: str = None
):
    """
    학습 오케스트레이션 워커: 데이터 수집 -> 학습(또는 파인튜닝) -> 평가 -> 업로드
    """
    temp_dir = f"temp_data/{job_id}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # 1. 진행 상태: 시작 보고 (10%)
        await backend_client.report_progress(job_id, 10, "RUNNING", "Initializing training pipeline")

        # 2. 데이터 다운로드 (S3 -> AI Server)
        raw_data_list = []
        for i, link in enumerate(datasets):
            dest_path = os.path.join(temp_dir, f"sample_{i}.json")
            await presigned_transfer_service.download_file(link.download_url, dest_path)
            
            # 다운로드한 JSON 로드
            with open(dest_path, "r", encoding="utf-8") as f:
                raw_data_list.append(json.load(f))
        
        await backend_client.report_progress(job_id, 30, "RUNNING", f"Downloaded {len(raw_data_list)} datasets")

        # 3. 학습 실행 (Fine-tuning 지원)
        logger.info(f"[{job_id}] Starting {'fine-tuning' if is_finetune else 'training'}...")
        results = train_pipeline.execute(
            raw_data_list=raw_data_list,
            output_dir=f"app/models/artifacts/{job_id}",
            is_finetune=is_finetune,
            base_model_path=base_model_path
        )

        # 4. 평가 지표 보고 (60%)
        metrics = {
            "rmse": round(float(results["rmse"]), 4),
            "f1_score": round(float(results["f1_score"]), 4)
        }
        await backend_client.report_metrics(job_id, metrics)
        await backend_client.report_progress(job_id, 80, "RUNNING", "Training completed, uploading model...")

        # 5. 모델 업로드 (S3 Presigned PUT)
        upload_info = await backend_client.issue_onnx_upload_link(job_id, version_id)
        await presigned_transfer_service.upload_file(upload_info.upload_url, results["onnx_path"])

        # 6. 최종 완료 보고 (100%)
        completion_payload = {
            "version_id": version_id,
            "s3_key": f"mlops/models/{version_id}/{job_id}/model.onnx", # 백엔드 규칙에 따라 조정
            "rmse": metrics["rmse"],
            "f1_score": metrics["f1_score"],
            "activate": True
        }
        await backend_client.complete_onnx(job_id, completion_payload)
        await backend_client.report_progress(job_id, 100, "COMPLETED", "Pipeline successfully finished")

    except Exception as e:
        logger.error(f"[{job_id}] Pipeline failed: {str(e)}")
        await backend_client.report_progress(job_id, 0, "FAILED", str(e))
    finally:
        # 임시 데이터 삭제
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
