# 🎮 Rewrite-AI (AI-based Dynamic Difficulty Adjustment (DDA) Server)

본 프로젝트는 유저의 플레이 데이터를 실시간으로 분석하여 **숙련도(Skill Level)와 이탈 위험도(Churn Risk)**를 예측하고, 이를 바탕으로 게임의 난이도를 동적으로 조절(DDA)하는 AI 학습 및 추론 서버입니다.

## 🚀 Key Features
- **시계열 데이터 처리**: 30초 단위 슬라이딩 윈도우 기반의 유저 로그 분석.
- **GRU(Gated Recurrent Unit)**: LSTM 대비 25~30% 적은 연산량으로 모바일/Unity 환경에 최적화된 시계열 특징 추출.
- **다중 작업 학습 (Multi-task Learning)**: 하나의 모델로 숙련도(회귀)와 이탈 위험도(분류)를 동시에 추론.
- **Unity Sentis 호환**: PyTorch 모델을 ONNX 포맷으로 변환하여 유니티 클라이언트 로컬 추론 지원.
- **학습 서버 최적화**: 학과 자체 GPU(CUDA 1) 자원을 활용한 고성능 학습 파이프라인.

## 🛠 Tech Stack
- **Framework**: FastAPI (Python 3.10+)
- **Deep Learning**: PyTorch, ONNX
- **Data Analysis**: Pandas, NumPy, Scikit-learn
- **Infrastructure**: AWS (Log storage), Local GPU Server (Training)

## 📁 Directory Structure
- `app/api`: MLOps 및 추론 엔드포인트
- `app/pipelines`: 데이터 전처리, 텐서화 및 학습 파이프라인
- `app/models`: GRU 모델 정의 및 가중치 로더
- `app/services`: 백엔드 통신 및 모델 레지스트리 관리
- `scratch`: 로컬 테스트용 데이터 생성 및 검증 스크립트

## 🧪 How to Test
1. **의존성 설치**: `pip install -r requirements.txt`
2. **테스트 데이터 생성**: `python scratch/generate_test_samples.py`
3. **파이프라인 실행**: `python scratch/run_manual_test.py`
