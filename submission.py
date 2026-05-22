import csv
from pathlib import Path
from ultralytics import YOLO

# 1. 경로 설정
MODEL_PATH = "./saved_models/20260522_090049/train/weights/best.pt"       # 10에폭 최적 가중치
TEST_IMG_DIR = "./data/TRAIN_VAL_DATASET/val/images"                      # 최종 평가용 이미지 폴더
SUBMISSION_CSV_PATH = "./final_submission.csv"                            # 저장할 파일명

# 2. YOLO 모델 로드 및 추론 (이미지 저장은 불필요하므로 save=False)
model = YOLO(MODEL_PATH)
results = model.predict(source=TEST_IMG_DIR, conf=0.25, save=False)

# 3. CSV 파일 작성
with open(SUBMISSION_CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    
    # 대회 규격 헤더 주입
    writer.writerow(["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"])
    
    annotation_counter = 1
    
    for result in results:
        # 파일명에서 숫자 ID만 추출 (예: '0001.jpg' -> 1)
        try:
            image_id = int(''.join(filter(str.isdigit, Path(result.path).stem)))
        except ValueError:
            image_id = Path(result.path).stem  # 숫자가 없으면 파일명 그대로 사용
            
        boxes = result.boxes
        
        for box in boxes:
            category_id = int(box.cls[0].item())        # 클래스 번호 (0~55)
            score = round(float(box.conf[0].item()), 2) # 신뢰도 점수
            
            # YOLO xyxy -> [좌상단_x, 좌상단_y, 우하단_x, 우하단_y]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # 대회 규격인 Width, Height로 변환
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            
            # CSV 가로행 기록
            writer.writerow([
                annotation_counter, 
                image_id, 
                category_id, 
                x1, 
                y1, 
                bbox_w, 
                bbox_h, 
                f"{score:.2f}"
            ])
            
            annotation_counter += 1

print("최종 제출용 CSV 추출 완료")
print(f"경로: {Path(SUBMISSION_CSV_PATH).resolve()}")