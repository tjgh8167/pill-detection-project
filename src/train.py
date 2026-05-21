import os


def train_model(
    model,
    data_yaml,
    epochs,
    batch_size,
    imgsz,
    save_dir,
    lr=1e-4,
    weight_decay=1e-4,
    optimizer_name="Adam",
    experiment_name="train"
):
    """
    YOLOv11 모델 학습 함수

    역할:
    - data.yaml을 기준으로 YOLOv11 모델 학습
    - best.pt, last.pt 자동 저장
    - results.png, results.csv 자동 저장
    - validation mAP 자동 계산

    인자:
    - model: get_model()에서 반환된 YOLO 모델
    - data_yaml: YOLO 데이터 설정 파일 경로
    - epochs: 학습 epoch 수
    - batch_size: batch size
    - imgsz: 입력 이미지 크기
    - save_dir: 학습 결과 저장 폴더
    - lr: learning rate
    - weight_decay: weight decay
    - optimizer_name: optimizer 이름
    - experiment_name: 저장될 실험 폴더 이름

    반환:
    - model: 학습된 YOLO 모델
    - results: Ultralytics 학습 결과 객체
    """

    os.makedirs(save_dir, exist_ok=True)

    print("========== Train Config ==========")
    print(f"Data YAML: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {imgsz}")
    print(f"Learning Rate: {lr}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Save Dir: {save_dir}")
    print("==================================")

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        lr0=lr,
        weight_decay=weight_decay,
        optimizer=optimizer_name,
        project=str(save_dir),
        name=experiment_name,
        exist_ok=True
    )

    print("학습 완료")
    print(f"학습 결과 저장 위치: {save_dir}/{experiment_name}")

    return model, results