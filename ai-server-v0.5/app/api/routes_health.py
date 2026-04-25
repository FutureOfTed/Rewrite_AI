from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def health_check():
    # 상태 점검 API: 프로세스 생존 여부
    return {"status": "ok"}

@router.get("/ready")
async def readiness_check():
    # 상태 점검 API: 모델 로드/의존성 준비 상태 여부
    return {"status": "ready"}
