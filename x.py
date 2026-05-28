import os
from pathlib import Path

# 검증할 YOLO 데이터셋 경로 지정
DATASET_ROOT = "data/TRAIN_VAL_DATASET"

def check_yolo_dataset(split_name):
    img_dir = Path(DATASET_ROOT) / split_name / "images"
    lbl_dir = Path(DATASET_ROOT) / split_name / "labels"
    
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"[{split_name}] 경로를 찾을 수 없습니다.")
        return

    img_files = set(f.stem for f in img_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png'])
    lbl_files = set(f.stem for f in lbl_dir.glob("*.txt"))
    
    print(f"📊 [{split_name.upper()} DATASET CHECK]")
    print(f" └─ 이미지 개수: {len(img_files)}장 / 라벨 개수: {len(lbl_files)}개")
    
    # 1. 미매칭 파일 확인
    img_only = img_files - lbl_files
    lbl_only = lbl_files - img_files
    if img_only: print(f" 라벨이 없는 이미지 (배경으로 학습됨): {list(img_only)[:5]}... 등 총 {len(img_only)}장")
    if lbl_only: print(f" 이미지가 없는 붕 뜬 라벨 파일 (에러 유발 가능): {list(lbl_only)[:5]}... 등 총 {len(lbl_only)}개")

    # 2. 라벨 내용물 및 클래스 ID 분포 전수조사
    all_classes = set()
    corrupt_counter = 0
    
    for txt_file in lbl_dir.glob("*.txt"):
        with open(txt_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    corrupt_counter += 1
                    continue
                
                cls_id = int(parts[0])
                all_classes.add(cls_id)
                
                # 좌표 범위 체크 (0.0 ~ 1.0 사이인지)
                coords = list(map(float, parts[1:]))
                if any(c < 0.0 or c > 1.0 for c in coords):
                    print(f" 좌표 정규화 오류 발견: {txt_file.name} -> {line.strip()}")

    print(f" └─ 탐지된 유효 클래스 인덱스 (0부터 연속적인지 확인): {sorted(list(all_classes))}")
    if corrupt_counter:
        print(f" 포맷이 깨진 불량 라벨 줄 수: {corrupt_counter}개")
    print("-" * 50)

# Train과 Val 폴더 각각 검증
check_yolo_dataset("train")
check_yolo_dataset("val")