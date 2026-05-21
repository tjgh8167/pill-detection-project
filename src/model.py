# 1. 설치 확인
# pip install ultralytics

# 2. 모델 로드
from ultralytics import YOLO

def get_model(model_size="m"):
    """
    YOLOv11 모델 반환
    - num_classes: 4 (알약 종류)
    - pretrained: yolo11m.pt
    - model_size: n, s, m, l, x 선택 가능 (기본값: m)
    - pretrained: yolo11{model_size}.pt
    """
    model = YOLO(f"yolo11{model_size}.pt")
    return model