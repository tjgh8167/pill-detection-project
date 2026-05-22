import csv
import os, sys
from pathlib import Path
from ultralytics import YOLO

# 1. 경로 설정
BASE_MODEL_DIR = "./saved_models"                                         # 최적 가중치
TEST_IMG_DIR = "./data/TRAIN_VAL_DATASET/val/images"                      # 최종 평가용 이미지 폴더
SUBMISSION_CSV_PATH = "./final_submission.csv"                            # 저장할 파일명

try:
    # 1-1. 폴더 존재 여부 먼저 확인
    if not os.path.exists(BASE_MODEL_DIR):
        raise FileNotFoundError(f"'{BASE_MODEL_DIR}' 폴더가 존재하지 않습니다. 먼저 모델 학습을 진행해 주세요.")
        
    # 1-2. 하위 디렉토리 리스트 추출
    subdirs = [os.path.join(BASE_MODEL_DIR, d) for d in os.listdir(BASE_MODEL_DIR) if os.path.isdir(os.path.join(BASE_MODEL_DIR, d))]
    
    if not subdirs:
        raise ValueError(f"'{BASE_MODEL_DIR}' 폴더 내부에 타임스탬프 모델 폴더가 비어 있습니다.")
    
    # 1-3. 가장 최근에 생성(수정)된 폴더 선택 및 경로 조립
    latest_dir = max(subdirs, key=os.path.getmtime)
    MODEL_PATH = os.path.join(latest_dir, "train", "weights", "best.pt")
    
    # 1-4. 최종 가중치 파일(.pt)이 실제로 존재치 않는 경우 예외 처리
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"최신 모델 폴더는 찾았으나, 가중치 파일({MODEL_PATH})이 존재하지 않습니다. 학습이 정상 종료되었는지 확인하세요.")

    print(f"가장 최신의 폴더 {MODEL_PATH}에서 가중치를 로드합니다.")

except Exception as e:
    print(f"오류 발생: {e}")
    sys.exit(1)

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