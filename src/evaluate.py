import os


def evaluate_model(
    model,
    data_yaml,
    save_dir,
    imgsz=1280,
    batch_size=16,
    experiment_name="val",
    augment=None,
    device = "cpu",
    max_det = None
):
    """
    YOLOv11 모델 평가 함수

    역할:
    - validation set 기준으로 모델 성능 평가
    - precision, recall, mAP 계산
    - confusion matrix, PR curve 등 평가 결과 자동 저장
    - augment 옵션을 통해 TTA 적용 가능

    인자:
    - model: 학습된 YOLO 모델
    - data_yaml: YOLO 데이터 설정 파일 경로
    - save_dir: 평가 결과 저장 폴더
    - imgsz: 입력 이미지 크기
    - batch_size: batch size
    - experiment_name: 평가 결과 폴더 이름
    - augment: 검증 시 TTA 적용 여부


    반환:
    - metrics: Ultralytics validation 결과 객체
    """

    os.makedirs(save_dir, exist_ok=True)

    print("========== Evaluation Config ==========")
    print(f"Data YAML: {data_yaml}")
    print(f"Image Size: {imgsz}")
    print(f"Batch Size: {batch_size}")
    print(f"Save Dir: {save_dir}")
    print("=======================================")

    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch_size,
        project=str(save_dir),
        name=experiment_name,
        exist_ok=True,
        augment=augment,
        device = device,
        max_det = max_det
    )

    print("검증 평가 완료")
    print(f"평가 결과 저장 위치: {save_dir}/{experiment_name}")

    print("========== 핵심 평가 지표 ==========")
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@[0.75:0.95]: {metrics.box.map:.4f}")
    print("====================================")

    return metrics



def predict_and_visualize(
    model,
    source,
    save_dir,
    imgsz=640,
    conf=0.25,
    experiment_name="predict",
    augment=None,
    save_crop=True,
    save_txt=True,
    device = "cpu",
    max_det = None
):
    """
    YOLOv11 예측 및 시각화 함수

    역할:
    - 이미지 폴더 또는 이미지 파일에 대해 예측 수행
    - 예측 bbox가 그려진 이미지 자동 저장

    인자:
    - model: 학습된 YOLO 모델
    - source: 예측할 이미지 폴더 또는 이미지 파일 경로
    - save_dir: 예측 결과 저장 폴더
    - imgsz: 입력 이미지 크기
    - conf: confidence threshold
    - experiment_name: 예측 결과 폴더 이름
    - save_crop: 예측 객체 crop 이미지 저장 여부
    - save_txt: 예측 bbox txt 저장 여부

    반환:
    - predictions: Ultralytics 예측 결과 리스트
    """

    os.makedirs(save_dir, exist_ok=True)

    print("========== Predict Config ==========")
    print(f"Source: {source}")
    print(f"Image Size: {imgsz}")
    print(f"Confidence Threshold: {conf}")
    print(f"Augment: {augment}")
    print(f"Save Crop: {save_crop}")
    print(f"Save TXT: {save_txt}")
    print(f"Save Dir: {save_dir}")
    print("====================================")

    predictions = model.predict(
        source=str(source),
        imgsz=imgsz,
        conf=conf,
        save=True,
        project=str(save_dir),
        name=experiment_name,
        exist_ok=True,
        augment=augment,
        save_crop=save_crop,
        save_txt=save_txt,
        max_det=max_det,
    )

    print("예측 및 시각화 완료")
    print(f"예측 결과 저장 위치: {save_dir}/{experiment_name}")

    return predictions