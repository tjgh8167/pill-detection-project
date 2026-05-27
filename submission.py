import csv
import os, sys
import json
from pathlib import Path
from ultralytics import YOLO

# 1. 경로 설정, 파라미터 설정
BASE_MODEL_DIR = "/content/project_team4/saved_models"                                         
TEST_IMG_DIR = "/content/project_team4/data/sprint_ai_project1_data/test_images"                                         
ORIGINAL_LABELS = "/content/project_team4/data/sprint_ai_project1_data/train_annotations"

IMAGE_SIZE = 1280
CONF_THRESHOLD = 0.15
MAX_DET = 4


lbl_src_dir = Path(ORIGINAL_LABELS)
json_files = list(lbl_src_dir.rglob("*.json")) + list(lbl_src_dir.rglob("*.JSON"))
real_categories = set()
for j_file in json_files:
    try:
        with open(j_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "categories" in data:
                for cat in data["categories"]:
                    real_categories.add(int(cat["id"]))
    except Exception:
        continue
sorted_original_ids = sorted(list(real_categories))

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

    SUBMISSION_CSV_PATH = os.path.join(latest_dir, "final_submission.csv")

except Exception as e:
    print(f"오류 발생: {e}")
    sys.exit(1)

# 2. YOLO 모델 로드 및 추론
model = YOLO(MODEL_PATH)
results = model.predict(
source=TEST_IMG_DIR,
imgsz=IMAGE_SIZE,
conf=CONF_THRESHOLD,
max_det=MAX_DET,
augment=None,
save=False
)

# 3. CSV 파일 작성
with open(SUBMISSION_CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    
    # 대회 규격 헤더 주입
    writer.writerow(["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"])
    
    annotation_counter = 1
    
    for result in results:
        image_id = int(Path(result.path).stem)
            
        boxes = result.boxes
        
        for box in boxes:
            yolo_cls_id = int(box.cls[0].item())        
            
            # ⭐ [수정] 모델이 예측한 0, 1, 2... 를 대회의 진짜 ID(1, 24, 11...)로 복원합니다.
            category_id = sorted_original_ids[yolo_cls_id] 
            
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