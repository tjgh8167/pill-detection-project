import os
from datetime import datetime

# from src.data_loader import ... 
# from src.model import ...
# from src.train import ...


def main():

    DATA_PATH = "./data/sprint_ai_project1_data"
    BATCH_SIZE = 16

    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    OPTIMIZER = "Adam"
    LOSS_FUNCTION = "CrossEntropyLoss" 

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    SAVE_DIR = f"./saved_models/{current_time}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. 데이터셋과 데이터로더 생성
    '''
    train_loader, val_loader, test_loader = get_loaders(
        data_path=DATA_PATH, 
        batch_size=BATCH_SIZE)
    '''
    # 2. 모델 생성
    '''
    모델 함수 제작 후
    model = get_model()
    '''
    # 3. 모델 학습 및 성능 평가
    # 에폭별 Loss곡선 그래프 이미지 / 모델 저장: saved_models 폴더
    '''
    trained_model = train_model(
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
    print("학습이 완료되었습니다.")

    # 4. 학습 후 모델을 테스트 데이터셋으로 평가
    # 시각화 결과 저장 (원본 / 예측 결과 2개 비교 이미지)
    '''
    test_results = evaluate_model(
        model=trained_model, test_loader=test_loader)
    '''
    print("테스트 평가가 완료되었습니다.")


if __name__ == "__main__":
    main()