import os
import json
import random
import shutil
import re
import cv2
import yaml
import albumentations as A
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

def data_load(base_path='/content/project_team4/data/sprint_ai_project1_data',
              output_path='/content/project_team4/data',
              target_count=None):

    BASE_DIR = Path(base_path)
    RAW_IMAGE_DIR = BASE_DIR / 'train_images'
    RAW_LABEL_DIR = BASE_DIR / 'train_annotations'
    OUTPUT_DIR = Path(output_path)
    
    # 1. 클래스 정보 추출

    json_files = list(RAW_LABEL_DIR.rglob('*.json')) + list(RAW_LABEL_DIR.rglob('*.JSON'))
    unique_categories = {}

    print(f"--- 1. 클래스 정보 추출 중 ({len(json_files)}개 파일 발견) ---")
    for j_file in json_files:
        try:
            with open(j_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for cat in data.get('categories', []):
                    if cat['id'] not in unique_categories:
                        unique_categories[cat['id']] = cat['name']
        except Exception:
            continue

    sorted_ids = sorted(unique_categories.keys())
    class_map = {orig_id: idx for idx, orig_id in enumerate(sorted_ids)}
    class_names = [unique_categories[orig_id] for orig_id in sorted_ids]

    print(f"발견된 고유 클래스 수: {len(class_names)}")

    # 2. 데이터 무결성 검사

    img_files = list(RAW_IMAGE_DIR.glob('*.jpg')) + list(RAW_IMAGE_DIR.glob('*.png'))
    missing_stats = Counter()
    inconsistent_files = []

    print(f"\n--- 2. 데이터 무결성 검사 시작 ({len(img_files)}개 이미지) ---")
    for img_path in img_files:
        stem = img_path.stem
        expected_ids = [int(id_str) for id_str in re.findall(r'K-(\d{6})', stem)]
        if not expected_ids: continue

        matched_jsons = list(RAW_LABEL_DIR.rglob(f"{stem}*.json"))
        actual_ids = set()
        for j_file in matched_jsons:
            try:
                with open(j_file, 'r', encoding='utf-8') as f:
                    for ann in json.load(f).get('annotations', []):
                        actual_ids.add(ann['category_id'])
            except: continue

        missing_in_this_file = [eid for eid in expected_ids if eid not in actual_ids]
        if missing_in_this_file:
            for mid in missing_in_this_file:
                pill_name = class_names[class_map[mid]] if mid in class_map else f"Unknown({mid})"
                missing_stats[pill_name] += 1
            inconsistent_files.append(stem)

    final_clean_images = [p for p in img_files if p.stem not in inconsistent_files]
    print(f"불일치 발견: {len(inconsistent_files)}장 / 최종 정제 완료: {len(final_clean_images)}장 확보")

    # 3. 데이터셋 분할 및 YOLO 변환
    print(f"\n--- 3. 데이터셋 분할 및 YOLO 변환 ---")
    train_list, val_list = train_test_split(final_clean_images, test_size=0.2, random_state=42)
    print(f"분할 완료: 학습용 {len(train_list)}장, 검증용 {len(val_list)}장")

    for split in ['train', 'val']:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

    stem_to_jsons = defaultdict(list)
    for jf in json_files: stem_to_jsons[jf.stem].append(jf)

    def convert_to_yolo(img_list, split_name):
        count = 0
        for img_path in img_list:
            stem, img_name = img_path.stem, img_path.name
            matched = [f for s, files in stem_to_jsons.items() if s.startswith(stem) for f in files]

            yolo_labels = []
            for json_path in matched:
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if not data.get('images'): continue
                        img_w, img_h = data['images'][0]['width'], data['images'][0]['height']

                        for ann in data.get('annotations', []):
                            orig_id = ann['category_id']
                            if orig_id not in class_map: continue
                            class_idx = class_map[orig_id]
                            x, y, w, h = ann['bbox']
                            yolo_labels.append(f"{class_idx} {(x+w/2)/img_w:.6f} {(y+h/2)/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}")
                except: continue

            if yolo_labels:
                shutil.copy(img_path, OUTPUT_DIR / split_name / 'images' / img_name)
                with open(OUTPUT_DIR / split_name / 'labels' / f"{stem}.txt", 'w', encoding='utf-8') as f:
                    f.write("\n".join(list(set(yolo_labels))))
                count += 1
        print(f"{split_name} 처리 완료: {count}장 저장됨")

    convert_to_yolo(train_list, 'train')
    convert_to_yolo(val_list, 'val')

    data_yaml = {'train': str(OUTPUT_DIR / 'train' / 'images'), 'val': str(OUTPUT_DIR / 'val' / 'images'), 'nc': len(class_names), 'names': class_names}
    with open(OUTPUT_DIR / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, allow_unicode=True)

    # 4. 데이터 증강 (Train)

    print(f"\n--- 4. 데이터 증강 모드 ---")
    aug_pipeline = A.Compose([
        A.HorizontalFlip(0.5), 
        A.VerticalFlip(0.5), 
        A.RandomRotate90(0.5),
        A.RandomBrightnessContrast(0.2), 
        A.GaussianBlur(0.1),
        A.HueSaturationValue(10, 15, 10, 0.1)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    train_img_dir, train_lbl_dir = OUTPUT_DIR / 'train' / 'images', OUTPUT_DIR / 'train' / 'labels'
    class_to_files = {i: [] for i in range(len(class_names))}
    train_instances = []

    for lbl_file in train_lbl_dir.glob('*.txt'):
        if lbl_file.stem.startswith('aug_'): continue
        with open(lbl_file, 'r', encoding='utf-8') as f:
            classes_in_file = set()
            for line in f:
                if line.strip():
                    try:
                        cls_idx = int(float(line.split()[0]))
                        train_instances.append(cls_idx)
                        classes_in_file.add(cls_idx)
                    except: continue
            for c in classes_in_file: class_to_files[c].append(lbl_file.stem)

    counts = Counter(train_instances)
    minority_classes = [i for i, count in counts.items() if count < target_count]

    for cls_idx in minority_classes:
        current_count, needed = counts[cls_idx], target_count - counts[cls_idx]
        source_stems = class_to_files[cls_idx]
        if not source_stems: continue

        for i in range(needed):
            src_stem = random.choice(source_stems)
            img_paths = list(train_img_dir.glob(f'{src_stem}.*'))
            if not img_paths: continue

            image = cv2.imread(str(img_paths[0]))
            if image is None: continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            bboxes, class_labels = [], []
            with open(train_lbl_dir / f'{src_stem}.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5: continue
                    try:
                        class_labels.append(int(float(parts[0])))
                        bboxes.append([float(x) for x in parts[1:]])
                    except: continue
            try:
                aug = aug_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                new_stem = f'aug_{cls_idx}_{src_stem}_{i}'
                cv2.imwrite(str(train_img_dir / f'{new_stem}.jpg'), cv2.cvtColor(aug['image'], cv2.COLOR_RGB2BGR))
                with open(train_lbl_dir / f'{new_stem}.txt', 'w', encoding='utf-8') as f:
                    for label, box in zip(aug['class_labels'], aug['bboxes']):
                        f.write(f"{label} {' '.join(map(lambda x: f'{x:.6f}', box))}\n")
            except: continue

    print("모든 전처리 및 증강 파이프라인이 완료되었습니다!")

    return class_names, class_map, OUTPUT_DIR

# --- 실행 ---
if __name__ == "__main__":
    class_names, class_map, OUTPUT_DIR = data_load()