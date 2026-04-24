import json
import os
import sys
import torch

# 프로젝트 루트를 경로에 추가
sys.path.append(os.getcwd())

from app.pipelines.train_pipeline import execute

def run_test():
    # 1. 데이터 로드
    data_path = "scratch/test_dataset.json"
    if not os.path.exists(data_path):
        print("데이터 파일이 없습니다. generate_test_samples.py를 먼저 실행하세요.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data_list = json.load(f)

    print(f"총 {len(raw_data_list)}개의 웨이브 데이터를 로드했습니다.")
    
    # 2. GPU 환경 체크
    if not torch.cuda.is_available():
        print("\n[경고] CUDA를 사용할 수 없는 환경입니다.")
        print("현재 코드는 cuda:1을 강제 사용하도록 되어 있어 에러가 발생할 수 있습니다.")
        print("테스트를 위해 잠시 app/pipelines/train_pipeline.py의 _get_device()를 수정하시겠습니까?\n")
    elif torch.cuda.device_count() < 2:
        print(f"\n[경고] 감지된 GPU가 {torch.cuda.device_count()}개뿐입니다.")
        print("코드에서 요구하는 cuda:1(두 번째 GPU)이 없어 에러가 발생할 수 있습니다.\n")

    # 3. 파이프라인 실행
    try:
        print("학습 파이프라인(전처리->학습->평가->ONNX) 실행 중...")
        results = execute(raw_data_list)
        
        print("\n" + "="*50)
        print("학습 결과 보고서")
        print("="*50)
        print(f"최종 RMSE (숙련도 오차): {results['rmse']:.4f}")
        print(f"최종 F1-Score (이탈 예측): {results['f1_score']:.4f}")
        print(f"가중치 파일 저장 경로: {results['pt_path']}")
        print(f"ONNX 변환 파일 저장 경로: {results['onnx_path']}")
        print("="*50)
        
    except Exception as e:
        print(f"\n[에러 발생] 학습 도중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    run_test()
