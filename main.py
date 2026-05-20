import os
from datetime import datetime

# from src.data_loader import get_loaders
# from src.model import get_model
# from src.train import train_model, evaluate_model


def main():

    # 하이퍼파라미터 및 설정값 정의
    DATA_PATH = "/Users/apple/Desktop/project_team4/data/sprint_ai_project1_data"
    BATCH_SIZE = 16

    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    OPTIMIZER = "Adam"
    LOSS_FUNCTION = "CrossEntropyLoss" 

    # 모델과 학습 결과를 저장할 디렉토리 생성(시간별 폴더 생성)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_DIR = f"/Users/apple/Desktop/project_team4/saved_models/{current_time}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("[RUN] 프로젝트 파이프라인 가동")

    # 1. 데이터셋과 데이터로더 생성 (DE 파트)
    print("\n[STEP 1] 데이터셋 로딩 및 데이터로더 생성")
    '''
    train_loader, val_loader, test_loader = get_loaders(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
    )
    '''
    print(f" └─ train_loader 샘플: {len(train_loader.dataset)}개, val_loader 샘플: {len(val_loader.dataset)}개, test_loader 샘플: {len(test_loader.dataset)}개를 생성 했습니다.")

    # 2. 모델 생성 (MA 파트)
    print("[STEP 2] 모델 생성")
    '''
    모델 함수 제작 후
    model = get_model()
    '''
    # 3. 모델 학습 및 성능 평가 (EL 파트)
    # 에폭별 Loss곡선 그래프 이미지 / 모델 저장: saved_models 폴더
    print("[STEP 3] 모델 학습 및 성능 평가")
    '''
    trained_model, loss_graph = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        optimizer_name=OPTIMIZER,
        loss_fn=LOSS_FUNCTION,
        save_dir=SAVE_DIR
    )
    '''
    print(" └─ 학습 및 Loss 그래프, 모델 저장이 완료되었습니다.")

    # 4. 학습 후 모델을 테스트 데이터셋으로 평가 (EL 파트)
    # 시각화 결과 저장 (아무것도 안그려진 원본 이미지 / 예측 바운딩 박스 결과 비교 이미지)
    print("[STEP 4] 테스트 평가 및 시각화 저장")
    '''
    test_results = evaluate_model(
        model=trained_model, 
        test_loader=test_loader)
    '''
    print(" └─ 테스트 평가 및 시각화 저장이 완료되었습니다.")

    print("\n[DONE] 모든 프로세스가 성공적으로 종료되었습니다")
if __name__ == "__main__":
    main()