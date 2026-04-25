import logging
import os
import torch
from app.models.predictor import DDA_GRU_MultiTask

logger = logging.getLogger(__name__)

# 학습 서버는 GPU 1번을 전용으로 사용
TARGET_GPU_INDEX = 1


class ModelLoader:
    """
    학습 완료된 DDA_GRU_MultiTask 모델(.pt)을 디스크에서 로드하는 클래스.
    로드된 모델 인스턴스는 서빙 및 추론에 재사용된다.
    """

    def __init__(
        self,
        input_size:  int = 5,
        hidden_size: int = 64,
        num_layers:  int = 2,
    ):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # GPU 1번 고정 (학습 서버 전용 디바이스)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다. GPU 환경을 확인하세요.")
        if torch.cuda.device_count() <= TARGET_GPU_INDEX:
            raise RuntimeError(
                f"GPU {TARGET_GPU_INDEX}번이 존재하지 않습니다. "
                f"현재 감지된 GPU 수: {torch.cuda.device_count()}"
            )
        self.device = torch.device(f"cuda:{TARGET_GPU_INDEX}")
        logger.info(f"ModelLoader 초기화: 디바이스=cuda:{TARGET_GPU_INDEX} ({torch.cuda.get_device_name(TARGET_GPU_INDEX)})")

        self.model: DDA_GRU_MultiTask | None = None

    def load_model(self, model_path: str) -> DDA_GRU_MultiTask:
        """
        저장된 PyTorch 가중치 파일(.pt)을 로드합니다.

        Parameters
        ----------
        model_path : .pt 파일 경로 (state_dict 형식으로 저장된 것을 가정)

        Returns
        -------
        DDA_GRU_MultiTask: 평가 모드(eval)로 전환된 모델 인스턴스
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        logger.info(f"모델 로드 중: {model_path}")

        # 아키텍처 인스턴스 생성 후 가중치 복원
        model = DDA_GRU_MultiTask(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()  # 추론 모드 전환 (Dropout 비활성화)

        self.model = model
        logger.info("모델 로드 완료.")
        return model

    def get_model(self) -> DDA_GRU_MultiTask:
        """현재 로드된 모델 반환. 로드되지 않았으면 예외 발생."""
        if self.model is None:
            raise RuntimeError("로드된 모델이 없습니다. load_model()을 먼저 호출하세요.")
        return self.model
