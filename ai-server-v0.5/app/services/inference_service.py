from app.schemas.inference import InferenceRequest, InferenceResponse

async def run_inference(request: InferenceRequest) -> InferenceResponse:
    # 모델 추론 비즈니스 로직
    # 전처리 호출 -> 모델 실행 -> 후처리
    
    # 예시 응답
    return InferenceResponse(
        prediction=1.0,
        probability=0.95
    )
