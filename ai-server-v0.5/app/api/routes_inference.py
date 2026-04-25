from fastapi import APIRouter
from app.schemas.inference import InferenceRequest, InferenceResponse
from app.services import inference_service

router = APIRouter()

@router.post("/", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    # 요청 검증 -> inference_service 호출 -> 응답 반환
    result = await inference_service.run_inference(request)
    return result
