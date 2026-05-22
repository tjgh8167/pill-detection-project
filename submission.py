import csv
import os, sys
from pathlib import Path
from ultralytics import YOLO

# 1. 경로 설정
BASE_MODEL_DIR = "./saved_models"                                         
TEST_IMG_DIR = "./data/sprint_ai_project1_data/test_images"               
SUBMISSION_CSV_PATH = "./final_submission.csv"                            

try:
    if not os.path.exists(BASE_MODEL_DIR):
        raise FileNotFoundError(f"'{BASE_MODEL_DIR}' 폴더가 존재하지 않습니다. 먼저 모델 학습을 진행해 주세요.")
        
    subdirs = [os.path.join(BASE_MODEL_DIR, d) for d in os.listdir(BASE_MODEL_DIR) if os.path.isdir(os.path.join(BASE_MODEL_DIR, d))]
    
    if not subdirs:
        raise ValueError(f"'{BASE_MODEL_DIR}' 폴더 내부에 타임스탬프 모델 폴더가 비어 있습니다.")
    
    latest_dir = max(subdirs, key=os.path.getmtime)
    MODEL_PATH = os.path.join(latest_dir, "train", "weights", "best.pt")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"최신 모델 폴더는 찾았으나, 가중치 파일({MODEL_PATH})이 존재하지 않습니다.")

    print(f"가장 최신의 폴더 {MODEL_PATH}에서 가중치를 로드합니다.")

except Exception as e:
    print(f"오류 발생: {e}")
    sys.exit(1)

# 2. YOLO 모델 로드 및 추론
model = YOLO(MODEL_PATH)
results = model.predict(source=TEST_IMG_DIR, conf=0.25, save=False)

# 3. CSV 파일 작성
with open(SUBMISSION_CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    
    # 대회 규격 헤더 주입
    writer.writerow(["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"])
    
    annotation_counter = 1
    
    for result in results:
        # 파일명이 '1.png', '3.png' 이므로 순수 정수형 숫자로 안전하게 파싱됩니다.
        image_id = int(Path(result.path).stem)
            
        boxes = result.boxes
        
        for box in boxes:
            # ⭐ [중요 변경] 데이터 로더를 고쳤기 때문에 모델이 출력하는 cls 번호를 그대로 씁니다 (+1 안함)
            category_id = int(box.cls[0].item())        
            
            # ⭐ [중요 변경] 반올림 문자열 대신 순수 float 숫자형으로 소수점 정밀도를 보존합니다
            score = float(box.conf[0].item()) 
            
            # YOLO xyxy -> [좌상단_x, 좌상단_y, 우하단_x, 우하단_y]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # 대회 규격인 Width, Height로 변환 (COCO format)
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            
            # CSV 가로행 기록
            writer.writerow([
                annotation_counter, 
                image_id, 
                category_id, 
                x1,        # bbox_x
                y1,        # bbox_y
                bbox_w,    # bbox_w
                bbox_h,    # bbox_h
                score      # 숫자형 그대로 기록
            ])
            
            annotation_counter += 1

print("최종 제출용 CSV 추출 완료")
print(f"경로: {Path(SUBMISSION_CSV_PATH).resolve()}")