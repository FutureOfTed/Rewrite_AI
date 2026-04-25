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

        # ── 2. 학습 가능한 S(숙련도) 가중치 ─────────────────────────────────
        # 입력 텐서 피처 순서: [apm(0), inv_hit(1), hp_ret(2), accuracy(3), item_eff(4)]
        # S 가중치는 [accuracy(3), inv_hit(1), item_eff(4), hp_ret(2)] 4개 피처에 적용
        self.S_FEAT_IDX = [3, 1, 4, 2]

        # w1~w4를 nn.Parameter로 선언 → backpropagation으로 자동 최적화
        # 초기값은 균등(0.25)으로 설정, Softmax로 합계가 항상 1.0 유지
        self.skill_weights = nn.Parameter(
            torch.tensor([0.25, 0.25, 0.25, 0.25])
        )

        # ── 3. 다중 작업 출력층 (Multi-Task Heads) ────────────────────────

        # S 헤드: GRU 은닉 벡터(64) + 가중합 스칼라(1) → 총 65 입력
        # Sigmoid → 0.0 ~ 1.0 사이 연속 실수 출력
        self.head_S = nn.Sequential(
            nn.Linear(hidden_size + 1, 1),
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
            피처 순서: [apm, inverse_hit_rate, hp_retention_rate, accuracy, attack_item_efficiency]

        Returns
        -------
        s_score : [Batch, 1] 숙련도 점수 (0.0 ~ 1.0)
        c_risk  : [Batch, 1] 이탈 위험도 (0.0 ~ 1.0)
        """
        # GRU 순전파: 30초 시퀀스 전체를 순차 처리
        gru_out, _ = self.gru(x)  # gru_out: [Batch, 30, hidden_size]

        # 마지막 시점(t=30)의 은닉 상태만 추출 → 맥락이 압축된 최종 벡터
        h_last = gru_out[:, -1, :]  # [Batch, 64]

        # ── S(?숙련도) 산출: 학습 가능한 가중치 적용 ──────────────────────
        # 마지막 프레임(t=30)의 S 관련 4개 피처만 추출
        # S_FEAT_IDX: [accuracy(3), inv_hit_rate(1), item_eff(4), hp_retention(2)]
        s_features = x[:, -1, :][:, self.S_FEAT_IDX]  # [Batch, 4]

        # Softmax로 가중치를 정규화하여 w1+w2+w3+w4 = 1.0 커제적으로 유지
        w = torch.softmax(self.skill_weights, dim=0)  # [4]

        # 피처별 가중합: 단순 평균이 아닌 모델이 학습한 가중치로 계산
        s_weighted = (w * s_features).sum(dim=1, keepdim=True)  # [Batch, 1]

        # GRU 맥락(h_last) + 학습된 가중합을 함께 헤드에 입력
        # → GRU가 리즈을 완전히 파악하면 가중치에 편향을 줌, 파악 몦하면 업고 관리함
        s_input = torch.cat([h_last, s_weighted], dim=1)  # [Batch, 65]
        s_score = self.head_S(s_input)   # [Batch, 1]

        # ── C(이탈 위험도) 산출: GRU 은닉 벡터만 사용 ──────────────────────
        c_risk = self.head_C(h_last)     # [Batch, 1]

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
