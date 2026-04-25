from pydantic import BaseModel
from typing import List, Dict, Any

class InferenceRequest(BaseModel):
    features: List[float]
    # 필요 시 추가적인 입력 필드 정의

class InferenceResponse(BaseModel):
    prediction: float
    probability: float
    # 예측 결과 구조 정의
