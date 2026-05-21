# 1. 설치 확인
# pip install ultralytics

# 2. 모델 로드
from ultralytics import YOLO

def get_model():
    """
    YOLOv11 Medium 모델 반환
    - num_classes: 4 (알약 종류)
    - pretrained: yolo11m.pt
    """
    model = YOLO("yolo11m.pt")
    return model
