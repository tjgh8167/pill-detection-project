import os
import torch
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

from src.data_loader import load_yolo_data_config, validate_yolo_data_config, verify_yolo_conversion
from src.model import get_model
from src.train import train_model
from src.evaluate import evaluate_model, predict_and_visualize

def main():

    # 하이퍼파라미터 및 설정값 정의
    BATCH_SIZE = 16

    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    OPTIMIZER = "AdamW"
    IMAGE_SIZE = 1280
    CONF_THRESHOLD = 0.15
    MAX_DET = 4
    AUGMENT = True


    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🌟 [현재 할당된 연산 장치]: YOLO device='{DEVICE}'")

    # 모델과 학습 결과를 저장할 디렉토리 생성(시간별 폴더 생성)
    BASE_DIR = Path(__file__).resolve().parent
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_DIR = BASE_DIR / "saved_models" / current_time
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("[RUN] 프로젝트 파이프라인 가동")

    
    
    # 1. YOLOv11 데이터 구성 파일 로딩 (DE 파트)
    print("\n[STEP 1] YOLO 데이터 구성 파일 로딩")

    DATA_YAML = BASE_DIR / "data/TRAIN_VAL_DATASET/data.yaml"

    data_config = load_yolo_data_config(DATA_YAML)
    validate_yolo_data_config(data_config)
    
    print(f" └─ YOLO 데이터 구성 확인이 완료되었습니다.: root={data_config['root']}")

    # 데이터 구성 후 YOLO 형식으로 변환이 제대로 되었는지 검증용 코드 (필요시 사용)
    # verify_yolo_conversion(data_config)

    # 2. 모델 생성 (MA 파트)
    print("[STEP 2] 모델 생성")

    model = get_model(model_size="m")

    # 3. 모델 학습 및 성능 평가 (EL 파트)
    # 에폭별 Loss곡선 그래프 이미지 / 모델 저장: saved_models 폴더
    print("[STEP 3] 모델 학습 및 성능 평가")

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
    mosaic=1.0,
    degrees=15.0,
    device=DEVICE
)

    print(" └─ 학습 및 Loss 그래프, 모델 저장이 완료되었습니다.")

    # 4. 학습 후 모델을 테스트 데이터셋으로 평가 (EL 파트)
    # 시각화 결과 저장 (아무것도 안그려진 원본 이미지 / 예측 바운딩 박스 결과 비교 이미지)
    print("[STEP 4] 테스트 평가 및 시각화 저장")

    TEST_IMG_DIR = BASE_DIR / "data/sprint_ai_project1_data/test_images"

    evaluate_model(
    model = trained_model[0],
    data_yaml=DATA_YAML,
    save_dir=SAVE_DIR,
    imgsz=IMAGE_SIZE,
    batch_size=16,
    experiment_name="val",
    augment=AUGMENT,
    device = DEVICE
)

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

    print(" └─ 테스트 평가 및 시각화 저장이 완료되었습니다.")

    print("\n[DONE] 모든 프로세스가 성공적으로 종료되었습니다")
if __name__ == "__main__":
    main()