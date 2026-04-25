"""
학습 파이프라인 (train_pipeline.py)

전체 흐름:
  데이터 로드 → 전처리 → 텐서화 → 데이터셋 분할
      → GRU 모델 학습 (Early Stopping)
      → 검증 평가 (RMSE / F1-Score)
      → 최적 가중치 저장 (.pt)
      → ONNX 변환 (.onnx)
"""

import logging
import os
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from app.models.predictor import DDA_GRU_MultiTask
from app.pipelines.preprocess import (
    clean_and_feature_engineering,
    compute_skill_label,
    compute_churn_label,
)
from app.pipelines.tensorize import (
    create_sliding_windows,
    normalize_tensor,
    build_dataset,
    split_dataset,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 하이퍼파라미터 상수
# ─────────────────────────────────────────────────────────────────────────────
INPUT_SIZE   = 5      # APM, 회피율, HP보존율, 명중률, 아이템효율
HIDDEN_SIZE  = 64     # 과적합·추론 지연 방지를 고려한 균형 차원
NUM_LAYERS   = 2      # Stacked GRU (클라이언트 150ms 기준 제한)
DROPOUT      = 0.2    # 레이어 간 드롭아웃

BATCH_SIZE   = 128    # GPU 메모리 효율 + 기울기 안정성 균형
LEARNING_RATE = 0.001  # Adam 옵티마이저 학습률
MAX_EPOCHS   = 100    # 최대 에포크 (Early Stopping으로 실제 학습 횟수는 가변)
EARLY_STOP_PATIENCE = 10   # 검증 손실이 N 에포크 동안 개선 없으면 조기 종료

# 다중 작업 손실 가중치 (BCE for C + MSE for S)
LOSS_WEIGHT_C = 0.5   # 이탈 위험도 손실 가중치
LOSS_WEIGHT_S = 0.5   # 숙련도 손실 가중치

# ONNX 관련
ONNX_OPSET_VERSION = 12   # Unity Sentis 엔진과 가장 안정적으로 호환되는 버전
WINDOW_SIZE  = 30     # 슬라이딩 윈도우 크기 (초)


# 학습 서버는 GPU 1번을 전용으로 사용
TARGET_GPU_INDEX = 1

def _get_device() -> torch.device:
    """GPU 1번을 고정으로 사용합니다. GPU가 없거나 인덱스가 범위를 벗어나면 예외를 발생시킵니다."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. GPU 환경을 확인하세요.")
    if torch.cuda.device_count() <= TARGET_GPU_INDEX:
        raise RuntimeError(
            f"GPU {TARGET_GPU_INDEX}번이 존재하지 않습니다. "
            f"현재 감지된 GPU 수: {torch.cuda.device_count()}"
        )
    device = torch.device(f"cuda:{TARGET_GPU_INDEX}")
    logger.info(f"학습 디바이스: {device} ({torch.cuda.get_device_name(TARGET_GPU_INDEX)})")
    return device


def _build_dataloaders(X, y_s, y_c):
    """
    numpy 배열을 받아 Train / Val / Test DataLoader를 생성합니다.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    (
        X_train, X_val, X_test,
        y_s_train, y_s_val, y_s_test,
        y_c_train, y_c_val, y_c_test,
    ) = split_dataset(X, y_s, y_c)

    def _to_loader(X_, y_s_, y_c_, shuffle):
        dataset = TensorDataset(
            torch.from_numpy(X_).float(),
            torch.from_numpy(y_s_).float().unsqueeze(1),  # [N, 1]
            torch.from_numpy(y_c_).float().unsqueeze(1),  # [N, 1]
        )
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = _to_loader(X_train, y_s_train, y_c_train, shuffle=True)
    val_loader   = _to_loader(X_val,   y_s_val,   y_c_val,   shuffle=False)
    test_loader  = _to_loader(X_test,  y_s_test,  y_c_test,  shuffle=False)

    logger.info(
        f"데이터셋 분할 완료 → "
        f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )
    return train_loader, val_loader, test_loader


def _train_one_epoch(model, loader, optimizer, loss_fn_bce, loss_fn_mse, device):
    """한 에포크 학습을 수행하고 평균 손실을 반환합니다."""
    model.train()
    total_loss = 0.0

    for X_batch, y_s_batch, y_c_batch in loader:
        X_batch   = X_batch.to(device)
        y_s_batch = y_s_batch.to(device)
        y_c_batch = y_c_batch.to(device)

        optimizer.zero_grad()

        s_pred, c_pred = model(X_batch)

        # 다중 작업 손실 가중합:
        #   - S(숙련도, 회귀) → MSE Loss
        #   - C(이탈 위험도, 분류) → BCE Loss
        loss_s = loss_fn_mse(s_pred, y_s_batch)
        loss_c = loss_fn_bce(c_pred, y_c_batch)
        loss   = LOSS_WEIGHT_S * loss_s + LOSS_WEIGHT_C * loss_c

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def _evaluate(model, loader, loss_fn_bce, loss_fn_mse, device):
    """
    검증/테스트 데이터로 손실, RMSE, F1-Score를 계산하고 반환합니다.

    Returns
    -------
    dict: { "loss", "rmse", "f1_score" }
    """
    model.eval()
    total_loss = 0.0
    all_s_pred, all_s_true = [], []
    all_c_pred, all_c_true = [], []

    for X_batch, y_s_batch, y_c_batch in loader:
        X_batch   = X_batch.to(device)
        y_s_batch = y_s_batch.to(device)
        y_c_batch = y_c_batch.to(device)

        s_pred, c_pred = model(X_batch)

        loss_s = loss_fn_mse(s_pred, y_s_batch)
        loss_c = loss_fn_bce(c_pred, y_c_batch)
        loss   = LOSS_WEIGHT_S * loss_s + LOSS_WEIGHT_C * loss_c
        total_loss += loss.item()

        all_s_pred.append(s_pred.cpu())
        all_s_true.append(y_s_batch.cpu())
        all_c_pred.append(c_pred.cpu())
        all_c_true.append(y_c_batch.cpu())

    # RMSE (숙련도 회귀 오차)
    s_pred_cat = torch.cat(all_s_pred).squeeze(1).numpy()
    s_true_cat = torch.cat(all_s_true).squeeze(1).numpy()
    rmse = math.sqrt(float(np.mean((s_pred_cat - s_true_cat) ** 2)))

    # F1-Score (이탈 위험도 이진 분류, 임계값 0.5)
    c_pred_cat  = (torch.cat(all_c_pred).squeeze(1).numpy() >= 0.5).astype(int)
    c_true_cat  = torch.cat(all_c_true).squeeze(1).numpy().astype(int)
    f1 = f1_score(c_true_cat, c_pred_cat, zero_division=0)

    avg_loss = total_loss / max(len(loader), 1)
    return {"loss": avg_loss, "rmse": rmse, "f1_score": f1}


def export_to_onnx(model: DDA_GRU_MultiTask, save_path: str):
    """
    학습 완료된 PyTorch 모델을 Unity Sentis 호환 ONNX 포맷으로 변환합니다.

    Parameters
    ----------
    model     : 학습 완료된 DDA_GRU_MultiTask 인스턴스
    save_path : ONNX 파일 저장 경로 (.onnx)
    """
    model.eval()

    # 클라이언트 환경 모사: Batch=1 (유저 1명, 실시간 추론)
    # 형태: [Batch=1, Timesteps=30, Features=5]
    dummy_input = torch.randn(1, WINDOW_SIZE, INPUT_SIZE)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=ONNX_OPSET_VERSION,  # Unity Sentis 호환 최적 버전
        input_names=["game_metrics_30s"],  # Unity C# Tensor 주입 키
        output_names=["S_score", "C_risk"],# Unity C# 추론 결과 호출 키
        dynamic_axes={
            # 배치 크기를 동적으로 허용 (클라이언트 1명 ~ 서버 배치 추론 모두 대응)
            "game_metrics_30s": {0: "batch_size"},
            "S_score":          {0: "batch_size"},
            "C_risk":           {0: "batch_size"},
        },
    )
    logger.info(f"ONNX 변환 완료: {save_path}")


def execute(
    raw_data_list: list,
    output_dir: str = "app/models/artifacts",
    is_finetune: bool = False,
    base_model_path: str = None
) -> dict:
    """
    전체 학습 파이프라인 진입점. (파인튜닝 지원)

    Parameters
    ----------
    raw_data_list   : 백엔드에서 수신한 웨이브 JSON 원시 데이터 리스트
    output_dir      : 모델 파일 저장 디렉토리
    is_finetune     : 기존 모델 가중치를 기반으로 미세 조정을 할지 여부
    base_model_path : 파인튜닝 시 로드할 기초 모델(.pt) 경로
    """
    os.makedirs(output_dir, exist_ok=True)
    device = _get_device()
    logger.info(f"파이프라인 시작 (Mode: {'Fine-Tuning' if is_finetune else 'Initial Training'})")

    # ── 1. 전처리: 피처 계산 + C/S 레이블 추론 ───────────────────────────
    logger.info("전처리 및 레이블링 중...")
    sample_list = []

    for wave_idx, raw_data in enumerate(raw_data_list):
        try:
            feature_df, raw_df = clean_and_feature_engineering(raw_data)
            if feature_df.empty:
                continue

            windows = create_sliding_windows(feature_df, window_size=WINDOW_SIZE)
            if windows.size == 0:
                continue

            s_label = compute_skill_label(feature_df)
            c_label = compute_churn_label(
                feature_df=feature_df,
                raw_data=raw_data,
                wave_index=wave_idx,
                session_history=[],
            )

            sample_list.append({
                "tensor":  windows,
                "s_label": s_label,
                "c_label": c_label,
            })
        except Exception as e:
            logger.error(f"웨이브 {wave_idx} 처리 중 오류: {e}")
            continue

    if not sample_list:
        raise ValueError("전처리 후 유효한 학습 샘플이 없습니다.")

    # ── 2. 정규화 및 데이터셋 조립 ────────────────────────────────────────
    X, y_s, y_c = build_dataset(sample_list)
    X = normalize_tensor(X)

    # ── 3. DataLoader 생성 (Train 80% / Val 10% / Test 10%) ──────────────
    train_loader, val_loader, test_loader = _build_dataloaders(X, y_s, y_c)

    # ── 4. 모델 / 옵티마이저 / 손실함수 초기화 ──────────────────────────
    model = DDA_GRU_MultiTask(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    # 파인튜닝 설정: 기존 가중치 로드 및 GRU 레이어 동결
    actual_lr = LEARNING_RATE
    if is_finetune and base_model_path:
        if os.path.exists(base_model_path):
            logger.info(f"기초 모델 로드: {base_model_path}")
            model.load_state_dict(torch.load(base_model_path, map_location=device))

            # GRU 특징 추출 레이어 동결 (이미 학습된 시계열 지식 유지)
            for param in model.gru.parameters():
                param.requires_grad = False
            
            actual_lr = 0.0001  # 파인튜닝 시 낮은 학습률 적용
            logger.info("GRU 레이어 동결 및 파인튜닝 학습률(0.0001) 설정 완료")
        else:
            logger.warning(f"기초 모델을 찾을 수 없어 신규 학습으로 전환합니다: {base_model_path}")

    # 동결되지 않은(requires_grad=True) 파라미터만 옵티마이저에 등록
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=actual_lr
    )
    loss_fn_bce = nn.BCELoss()
    loss_fn_mse = nn.MSELoss()

    # ── 5. 학습 루프 + Early Stopping ─────────────────────────────────────
    logger.info("학습 루프 진입...")
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights_path = os.path.join(output_dir, "best_weights.pt")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, loss_fn_bce, loss_fn_mse, device)
        val_metrics = _evaluate(model, val_loader, loss_fn_bce, loss_fn_mse, device)
        val_loss = val_metrics["loss"]

        logger.info(
            f"[Epoch {epoch:>3}/{MAX_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"RMSE: {val_metrics['rmse']:.4f} | F1: {val_metrics['f1_score']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_weights_path)
            logger.info(f"최적 가중치 저장 (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"개선 없음 ({patience_counter}/{EARLY_STOP_PATIENCE})")

        if patience_counter >= EARLY_STOP_PATIENCE:
            logger.info(f"Early Stopping 발동! (Epoch {epoch}에서 학습 종료)")
            break

    # ── 6. 최종 평가 및 ONNX 변환 ────────────────────────────────────────
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    test_metrics = _evaluate(model, test_loader, loss_fn_bce, loss_fn_mse, device)
    
    onnx_path = os.path.join(output_dir, "DDA_GRU_MultiTask.onnx")
    export_to_onnx(model, onnx_path)

    metrics = {
        "rmse":      test_metrics["rmse"],
        "f1_score":  test_metrics["f1_score"],
        "pt_path":   best_weights_path,
        "onnx_path": onnx_path,
    }
    logger.info(f"파이프라인 완료: {metrics}")
    return metrics
