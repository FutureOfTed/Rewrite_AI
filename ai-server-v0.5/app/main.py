import sys
import os

# CUDA 라이브러리 경로 추가 (가장 중요!)
sys.path.append("/usr/local/cuda/lib64")

# Torch CUDA 경로 추가 (필요한 경우)
sys.path.append("/usr/local/lib/python3.9/site-packages/torch/lib")

from fastapi import FastAPI
from app.api import routes_health, routes_inference, routes_mlops
import logging
from app.logging_conf import configure_logging

# 로깅 설정 초기화
configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Server",
    description="AI Model Inference & Training Pipeline Server",
    version="1.0.0"
)

# 라우터 등록
app.include_router(routes_health.router, prefix="/health", tags=["health"])
app.include_router(routes_inference.router, prefix="/inference", tags=["inference"])
app.include_router(routes_mlops.router, prefix="/mlops", tags=["mlops"])

@app.on_event("startup")
async def startup_event():
    logger.info("AI Server is starting up...")
    # 여기서 모델 로드 등 초기화 작업 수행

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("AI Server is shutting down...")
