import os
import torch
from datetime import datetime
from pathlib import Path
from src.new_dataloader import data_load 

from src.model import get_model
from src.train import train_model
from src.evaluate import evaluate_model, predict_and_visualize

def main():

    BATCH_SIZE = 16
    EPOCHS = 150
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    OPTIMIZER = "AdamW"
    IMAGE_SIZE = 1280       # 훈련 및 평가용 이미지 크기
    CONF_THRESHOLD = 0.15
    MAX_DET = 4
    AUGMENT = True          # 추론(TTA) 증강 여부
    MODEL_SIZE = "m"        # YOLOv11 미디움 체급

    MOSAIC = 1.0
    DEGREES = 15.0

    # 데이터 보강용 타겟 수치 정의 (소수 클래스는 최소 15개 이상 확보하도록 증강)
    TARGET_COUNT = 15 

    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🌟 [현재 할당된 연산 장치]: YOLO device='{DEVICE}'")

    # 경로 설정 (.resolve()를 통해 절대 경로로 안전하게 확보)
    BASE_DIR = Path(__file__).resolve().parent
    
    # 원본 원천 데이터가 들어있는 경로
    RAW_DATA_DIR = BASE_DIR / "data" / "sprint_ai_project1_data"
    # 정제 및 증강되어 YOLO 학습셋이 최종 저장될 경로
    OUTPUT_DATA_DIR = BASE_DIR / "data"
    # 최종 결과물 data.yaml의 위치
    DATA_YAML = OUTPUT_DATA_DIR / "data.yaml"

    # 모델과 학습 결과를 저장할 디렉토리 생성(시간별 폴더 생성)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_DIR = BASE_DIR / "saved_models" / current_time
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("[RUN] 프로젝트 파이프라인 가동")

    # [STEP 1] 데이터 로더 구동
    print("\n[STEP 1] 신규 파이프라인 데이터 전처리 및 타겟 증강 시작")
    

    if not RAW_DATA_DIR.exists():
        print(f"❌ [오류] 데이터 경로를 찾을 수 없습니다: {RAW_DATA_DIR}")
        return

    class_names, class_map, updated_output_dir = data_load(
        base_path=str(RAW_DATA_DIR),
        output_path=str(OUTPUT_DATA_DIR),
        target_count=TARGET_COUNT
    )
    
    print(f" └─ 데이터 전처리 및 자동 복구 완료!")
    print(f" └─ 생성된 고유 클래스 수: {len(class_names)}개")
    print(f" └─ 생성된 가동 가이드라인 파일: {DATA_YAML.resolve()}")

    # [STEP 2] 모델 생성 (MA 파트)
    print("\n[STEP 2] YOLO 모델 생성 및 가중치 빌드")
    model = get_model(model_size=MODEL_SIZE)

    # [STEP 3] 모델 학습 및 성능 평가 (EL 파트)
    print("\n[STEP 3] 고정밀 모델 학습 및 내부 검증 프로세스")
    
    trained_model = train_model(
        model,
        data_yaml=DATA_YAML,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        save_dir=SAVE_DIR,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        optimizer_name=OPTIMIZER,
        experiment_name="train",
        seed=42,
        mosaic=MOSAIC,
        degrees=DEGREES,
        device=DEVICE
    )

    print(" └─ 최적화 학습 완료 및 손실 함수(Loss) 추이 저장 완료.")

    # [STEP 4] 테스트 데이터셋 평가 및 추론 시각화 (EL 파트)
    print("\n[STEP 4] 리더보드용 테스트 평가 및 예측 결과 추출")

    TEST_IMG_DIR = RAW_DATA_DIR / "test_images"

    # 검증셋(Val)을 통한 정밀 성능 평가
    evaluate_model(
        model = trained_model[0],
        data_yaml=DATA_YAML,
        save_dir=SAVE_DIR,
        imgsz=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        experiment_name="val",
        augment=AUGMENT,
        device=DEVICE
    )

    # 캐글 제출의 기반이 되는 최종 테스트 추론 및 크롭 이미지, 텍스트 결과 생성
    predict_results = predict_and_visualize(
        model = trained_model[0],
        source = TEST_IMG_DIR,
        save_dir = SAVE_DIR,
        imgsz = IMAGE_SIZE,
        conf = CONF_THRESHOLD,
        max_det = MAX_DET,
        experiment_name = "predict",
        augment=AUGMENT,
        save_crop=True,
        save_txt=True,
        device = DEVICE
    )

    print(" └─ 캐글 대응용 추론(Prediction) 텍스트 및 결과물 저장 완료.")
    print(f"📂 [최종 결과 저장 경로]: {SAVE_DIR.resolve()}")
    print("\n[DONE] 모든 고도화 파이프라인 프로세스가 성공적으로 종료되었습니다.")

if __name__ == "__main__":
    main()