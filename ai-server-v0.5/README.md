# AI Server

FastAPI 기반의 AI 모델 추론 및 학습 파이프라인 관리 서버입니다.

## 기능
- **추론 API**: 학습된 모델을 로드하여 예측 결과를 반환합니다.
- **MLOps 학습 파이프라인**: 백엔드 시스템과 연동하여 비동기 모델 재학습 및 평가 과정을 수행합니다.
  - S3 직접 접근 대신 백엔드에서 발급하는 **Presigned URL**을 통해 데이터를 송수신합니다.

## 실행 방법
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
