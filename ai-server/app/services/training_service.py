import logging
import os
import json
import hashlib
from typing import List
from app.schemas.backend_contracts import DatasetLinkInfo, OnnxUploadLinkResponse
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
    [dispatch] 학습 오케스트레이션 워커: 데이터 수집 → 학습 → 학습 상태 보고

    ONNX 변환/업로드는 이 함수에서 수행하지 않습니다.
    onnx-upload-link 웹훅 수신 시 run_onnx_export_and_upload()에서 별도 수행합니다.
    """
    temp_dir = f"temp_data/{job_id}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # 1. 진행 상태: 시작 보고
        await backend_client.report_progress(job_id, 10, "RUNNING", "Initializing training pipeline")

        # 2. 데이터 다운로드 (S3 → AI Server)
        raw_data_list = []
        for i, link in enumerate(datasets):
            dest_path = os.path.join(temp_dir, f"sample_{i}.json")
            await presigned_transfer_service.download_file(link.download_url, dest_path)

            with open(dest_path, "r", encoding="utf-8") as f:
                raw_data_list.append(json.load(f))

        await backend_client.report_progress(job_id, 30, "RUNNING", f"Downloaded {len(raw_data_list)} datasets")

        # 3. 학습 실행 (Fine-tuning 지원)
        logger.info(f"[{job_id}] Starting {'fine-tuning' if is_finetune else 'training'}...")
        output_dir = f"app/models/artifacts/{job_id}"
        results = train_pipeline.execute(
            raw_data_list=raw_data_list,
            output_dir=output_dir,
            is_finetune=is_finetune,
            base_model_path=base_model_path
        )

        # 4. 학습 완료: RMSE/F1 보고 → 학습 상태 보고
        await backend_client.report_training_complete(
            job_id,
            rmse=results["rmse"],
            f1_score=results["f1_score"],
        )
        logger.info(f"[{job_id}] Training pipeline finished. Waiting for onnx-upload-link webhook.")

    except train_pipeline.TrainingFailSafeException as fe:
        logger.error(f"[{job_id}] Training Fail Safe Triggered (Code {fe.code}): {fe.message}")
        try:
            await backend_client.report_progress(job_id, 0, "FAILED", fe.message)
        except Exception as report_error:
            logger.error(f"[{job_id}] Failed to report failure to backend: {report_error}")
    except Exception as e:
        logger.error(f"[{job_id}] Training pipeline failed: {str(e)}")
        try:
            await backend_client.report_progress(job_id, 0, "FAILED", f"Internal error: {str(e)}")
        except Exception as report_error:
            logger.error(f"[{job_id}] Failed to report failure to backend: {report_error}")
    finally:
        import shutil
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as clean_error:
                logger.warning(f"[{job_id}] Failed to clean temp dir: {clean_error}")


async def run_onnx_export_and_upload(
    job_id: str,
    version_id: str,
    upload_info: OnnxUploadLinkResponse,
):
    """
    [onnx-upload-link 웹훅] ONNX 변환 → S3 업로드 → onnx-complete 보고

    best_weights.pt 경로는 job_id로부터 복원합니다.
    (학습 완료 시 app/models/artifacts/{job_id}/best_weights.pt 에 저장됨)
    """
    output_dir = f"app/models/artifacts/{job_id}"
    best_weights_path = os.path.join(output_dir, "best_weights.pt")
    onnx_path = os.path.join(output_dir, "DDA_GRU_MultiTask.onnx")

    if not os.path.exists(best_weights_path):
        logger.error(f"[{job_id}] best_weights.pt not found: {best_weights_path}")
        try:
            await backend_client.report_progress(
                job_id, 0, "FAILED",
                f"best_weights.pt not found. Training may not have completed for job {job_id}."
            )
        except Exception:
            pass
        return

    try:
        await backend_client.report_progress(job_id, 10, "RUNNING", "Starting ONNX export")

        # 1. best_weights.pt → ONNX 변환
        import torch
        from app.models.predictor import DDA_GRU_MultiTask
        from app.pipelines.train_pipeline import (
            INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, export_to_onnx
        )

        device = torch.device("cpu")  # ONNX 변환은 CPU에서 수행
        model = DDA_GRU_MultiTask(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        )
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
        export_to_onnx(model, onnx_path)
        logger.info(f"[{job_id}] ONNX export complete: {onnx_path}")

        await backend_client.report_progress(job_id, 50, "RUNNING", "Uploading ONNX to S3")

        # 2. S3 Presigned PUT 업로드
        try:
            await presigned_transfer_service.upload_file(upload_info.upload_url, onnx_path)
        except Exception as upload_err:
            logger.warning(f"[{job_id}] ONNX upload failed, retrying: {upload_err}")
            await presigned_transfer_service.upload_file(upload_info.upload_url, onnx_path)

        # 3. onnx-complete 보고
        onnx_size = os.path.getsize(onnx_path)
        with open(onnx_path, "rb") as f:
            onnx_sha256 = hashlib.sha256(f.read()).hexdigest()

        completion_payload = {
            "version_id": version_id,
            "s3_key":     upload_info.s3_key,
            "sha256":     onnx_sha256,
            "size_bytes": onnx_size,
            "activate":   False,
        }
        await backend_client.complete_onnx(job_id, completion_payload)
        logger.info(f"[{job_id}] onnx-complete reported successfully.")

    except Exception as e:
        logger.error(f"[{job_id}] ONNX export/upload failed: {str(e)}")
        try:
            await backend_client.report_progress(job_id, 0, "FAILED", f"ONNX export error: {str(e)}")
        except Exception:
            pass
