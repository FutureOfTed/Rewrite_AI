import torch
import torch.nn as nn


class DDA_GRU_MultiTask(nn.Module):
    """
    동적 난이도 조정(DDA)을 위한 GRU 기반 다중 작업 학습 모델.

    구조 개요
    ---------
    입력  : [Batch, Sequence(30초), Features(5개)]
                ↓
    GRU   : 시계열 패턴 학습 (Stacked, 2층)
                ↓
    h_last: 마지막 시점의 압축된 은닉 상태 벡터 [Batch, 64]
                ↓
       ┌────────┴────────┐
     head_S           head_C
    (회귀, MSE)    (분류, BCE)
       ↓                 ↓
    S 숙련도 점수    C 이탈 위험도
    (0.0 ~ 1.0)     (0.0 ~ 1.0)

    Parameters
    ----------
    input_size  : 입력 피처 수 (기본값 5: APM, 회피율, HP보존율, 명중률, 아이템효율)
    hidden_size : GRU 은닉 차원 (기본값 64)
    num_layers  : GRU 레이어 수 (기본값 2, 클라이언트 추론 150ms 기준)
    dropout     : 과적합 방지용 드롭아웃 비율 (기본값 0.2, num_layers>1 일 때 적용)
    """

    def __init__(
        self,
        input_size:  int   = 5,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        dropout:     float = 0.2,
    ):
        super(DDA_GRU_MultiTask, self).__init__()

        # ── 1. 시계열 특징 추출기 (GRU) ───────────────────────────────────
        # batch_first=True → 입력 텐서 형태 [Batch, Seq_len, Features] 고정
        # dropout은 num_layers > 1 일 때 레이어 간 적용 (과적합 방지)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── 2. 다중 작업 출력층 (Multi-Task Heads) ────────────────────────
        # 하나의 은닉 벡터 h_last를 입력받아 S와 C를 독립적으로 추론

        # S 헤드: 숙련도 점수 (회귀, Regression)
        # Sigmoid → 0.0 ~ 1.0 사이 연속 실수 출력
        self.head_S = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

        # C 헤드: 이탈 위험도 (이진 분류 확률, Classification)
        # Sigmoid → 0.0 ~ 1.0 이탈 확률 출력 (BCE Loss 와 함께 사용)
        self.head_C = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : [Batch, Sequence(30), Features(5)]

        Returns
        -------
        s_score : [Batch, 1] 숙련도 점수 (0.0 ~ 1.0)
        c_risk  : [Batch, 1] 이탈 위험도 (0.0 ~ 1.0)
        """
        # GRU 순전파: 30초 시퀀스 전체를 순차 처리
        gru_out, _ = self.gru(x)  # gru_out: [Batch, 30, hidden_size]

        # 마지막 시점(t=30)의 은닉 상태만 추출 → 맥락이 압축된 최종 벡터
        h_last = gru_out[:, -1, :]  # [Batch, 64]

        # 두 헤드를 통해 S와 C를 동시에 독립 산출
        s_score = self.head_S(h_last)  # [Batch, 1]
        c_risk  = self.head_C(h_last)  # [Batch, 1]

        return s_score, c_risk


class Predictor:
    """
    학습 완료된 DDA_GRU_MultiTask 모델을 래핑하여
    numpy 배열 입력 → S/C 추론 결과 딕셔너리를 반환하는 인터페이스.
    """

    def __init__(self, model: DDA_GRU_MultiTask, device: str = "cuda:1"):
        self.model  = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, tensor_input) -> dict:
        """
        Parameters
        ----------
        tensor_input : np.ndarray 또는 torch.Tensor, shape [Batch, 30, 5]

        Returns
        -------
        {
            "s_score": List[float],  # 숙련도 점수 (0.0 ~ 1.0)
            "c_risk":  List[float],  # 이탈 위험도 (0.0 ~ 1.0)
        }
        """
        import numpy as np

        if isinstance(tensor_input, np.ndarray):
            x = torch.from_numpy(tensor_input).float().to(self.device)
        else:
            x = tensor_input.float().to(self.device)

        s_score, c_risk = self.model(x)

        return {
            "s_score": s_score.squeeze(-1).cpu().tolist(),
            "c_risk":  c_risk.squeeze(-1).cpu().tolist(),
        }
