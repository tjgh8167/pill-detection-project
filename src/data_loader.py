import os
import json
import shutil
import random
from pathlib import Path
from PIL import Image as PILImage, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_IMAGES = BASE_DIR / "../data/sprint_ai_project1_data/train_images"
ORIGINAL_LABELS = BASE_DIR / "../data/sprint_ai_project1_data/train_annotations"
OUTPUT_PROJECT_ROOT = BASE_DIR / "../data/TRAIN_VAL_DATASET"

# YOLOv11 학습에 필요한 train/val 이미지 라벨 폴더 구조 생성
def prepare_yolo_directories(output_root: Path):
    train_img_out = output_root / "train/images"
    train_lbl_out = output_root / "train/labels"
    val_img_out = output_root / "val/images"
    val_lbl_out = output_root / "val/labels"

    for path in [train_img_out, train_lbl_out, val_img_out, val_lbl_out]:
        path.mkdir(parents=True, exist_ok=True)
        
    return train_img_out, train_lbl_out, val_img_out, val_lbl_out

# COCO 포맷의 JSON들을 파싱하여 YOLO 데이터셋 생성
def build_yolo_dataset(image_dir: Path, label_dir: Path, output_root: Path):
    img_src_dir = image_dir
    lbl_src_dir = label_dir
    root_out_dir = output_root

    json_files = list(lbl_src_dir.rglob("*.json")) + list(lbl_src_dir.rglob("*.JSON"))
    
    # ---------------------------------------------------------------- #
    # ⭐ [수정 완료] 진짜 존재하는 카테고리 정보만 딕셔너리로 완벽하게 수집
    # ---------------------------------------------------------------- #
    real_categories = {}  # {대회_original_id: "알약이름"}
    for j_file in json_files:
        try:
            with open(j_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "categories" in data:
                    for cat in data["categories"]:
                        real_categories[int(cat["id"])] = cat["name"]
        except Exception:
            continue

    if not real_categories:
        print("[오류] JSON 파일에서 알약 카테고리 이름을 찾지 못했습니다.")
        return

    # 대회 오리지널 ID를 오름차순으로 정렬하여 유령 unknown_X 생성을 원천 차단합니다.
    sorted_original_ids = sorted(list(real_categories.keys()))
    
    # data.yaml에 깔끔하게 들어갈 실제 알약 이름 목록 (예: 56개, 80개 등 실제 개수만큼만 생성)
    class_names = [real_categories[orig_id] for orig_id in sorted_original_ids]
    
    # 거대한 원본 ID를 YOLO 학습용 0번 기반 인덱스로 압축 매핑하는 테이블
    # 예: {41769: 0, 41800: 1, ...}
    original_to_yolo_id = {orig_id: idx for idx, orig_id in enumerate(sorted_original_ids)}

    print(f"\n[안내] 발견된 실제 클래스 개수: {len(class_names)}개")
    print(f"[안내] 매핑된 원본 ID 예시 (앞 5개): {sorted_original_ids[:5]}\n")

    # 2. 폴더 구조 생성
    if root_out_dir.exists():
        shutil.rmtree(root_out_dir)
    prepare_yolo_directories(root_out_dir)

    # 3. 이미지 파일 스캔
    all_images = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
        all_images.extend(list(img_src_dir.rglob(ext)))
    
    # 시드 고정
    random.seed(42)
    random.shuffle(all_images)
    
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    def convert_and_copy(image_list, phase):
        print(f"[{phase.upper()} 세트] {len(image_list)}장 변환 진행")
        
        from collections import defaultdict
        json_groups = defaultdict(list)
        
        for j in json_files:
            normalized_name = j.name.lower().replace("-", "").replace("_", "").replace(" ", "")
            json_groups[normalized_name].append(j)

        for img_path in image_list:
            norm_img_stem = img_path.stem.lower().replace("-", "").replace("_", "").replace(" ", "")
            
            matched_jsons = []
            for norm_json_key, j_list in json_groups.items():
                if norm_img_stem in norm_json_key or norm_json_key in norm_img_stem:
                    matched_jsons.extend(j_list)
            
            if not matched_jsons:
                continue

            new_img_path = root_out_dir / phase / "images" / img_path.name
            new_lbl_path = root_out_dir / phase / "labels" / f"{img_path.stem}.txt"

            # 이미지 복사
            shutil.copy(img_path, new_img_path)

            yolo_lines = []

            for json_path in matched_jsons:
                with open(json_path, "r", encoding="utf-8") as f:
                    ann_data = json.load(f)

                images_list = ann_data.get("images", [])
                if isinstance(images_list, dict):
                    images_list = [images_list]
                
                img_w, img_h = 976, 1280
                if images_list:
                    img_w = images_list[0].get("width") or 976
                    img_h = images_list[0].get("height") or 1280

                annotations = ann_data.get("annotations", [])
                for ann in annotations:
                    cat_id = int(ann.get("category_id"))
                    
                    # ⭐ [수정 완료] 유동적으로 늘어나던 이전 로직 제거
                    # 압축 매핑 테이블을 사용하여 0, 1, 2... 순서의 안전한 class_id를 부여합니다.
                    if cat_id in original_to_yolo_id:
                        class_id = original_to_yolo_id[cat_id]
                    else:
                        continue

                    bbox = ann.get("bbox") 
                    if bbox and len(bbox) == 4:
                        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                        
                        if None in [x, y, w, h]: 
                            continue
                        
                        # YOLO 표준 정규화 좌표 계산
                        x_center = (x + w / 2) / img_w
                        y_center = (y + h / 2) / img_h
                        norm_w = w / img_w
                        norm_h = h / img_h

                        if x_center > 1.0 or y_center > 1.0 or norm_w > 1.0 or norm_h > 1.0:
                            continue

                        # 클래스 ID와 정규화된 좌표 저장
                        line_str = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
                        if line_str not in yolo_lines:  # 중복 저장 방지
                            yolo_lines.append(line_str)

            # 최종 병합된 좌표가 있을 때만 파일 쓰기
            if yolo_lines:
                with open(new_lbl_path, "w", encoding="utf-8") as lf:
                    lf.write("\n".join(yolo_lines))

    convert_and_copy(train_images, "train")
    convert_and_copy(val_images, "val")

    # 4. 최종 data.yaml 작성
    yaml_path = root_out_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(f"path: {root_out_dir.resolve()}\n")
        yf.write("train: train/images\n")
        yf.write("val: val/images\n\n")
        yf.write(f"nc: {len(class_names)}\n")
        yf.write(f"names: {class_names}\n")
        
    print(f"[완료] YOLO 포맷 데이터셋 .yaml 파일 생성 / 저장경로 -> {yaml_path}")

# data.yaml 파일이 없거나 데이터셋이 미완성이면 자동으로 빌드한 뒤 생성
def load_yolo_data_config(yaml_path: str) -> dict:
    yaml_file_path = Path(yaml_path)
    
    if not yaml_file_path.exists() or not (yaml_file_path.parent / "train/images").exists():
        build_yolo_dataset(
            image_dir=ORIGINAL_IMAGES,
            label_dir=ORIGINAL_LABELS,
            output_root=OUTPUT_PROJECT_ROOT
        )

    config = {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                if val.startswith("[") and val.endswith("]"):
                    val = [item.strip().strip("'").strip('"') for item in val[1:-1].split(",")]
                elif val.isdigit():
                    val = int(val)
                config[key] = val
                
    return {
        "root": config.get("path"),
        "train_dir": config.get("train"),
        "val_dir": config.get("val"),
        "nc": config.get("nc"),
        "names": config.get("names", [])
    }

# YAML 파일의 내용을 바탕으로 실제 경로와 클래스 수 등을 검증
def validate_yolo_data_config(config: dict):
    if not config["root"]:
        raise ValueError("YAML 파일에 'path' 설정이 누락되었습니다.")
        
    root_path = Path(config["root"])
    train_path = root_path / config["train_dir"]
    val_path = root_path / config["val_dir"]
    
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("YAML에 적힌 물리적인 train/val 이미지 디렉토리를 찾을 수 없습니다.")
        
    if config["nc"] != len(config["names"]):
        raise ValueError("YAML 내 nc(클래스 수)와 names 배열의 크기가 일치하지 않습니다.")

# 학습셋 중 이미지와 어노테이션 매칭이 잘 됐는지 시각적 검증 함수
def verify_yolo_conversion(yaml_config: dict):
    root_path = Path(yaml_config["root"])
    train_img_dir = root_path / yaml_config["train_dir"]
    train_lbl_dir = root_path / yaml_config["train_dir"].replace("images", "labels")
    class_names = yaml_config["names"]

    img_files = []
    for ext in ["*.png", "*.jpg", "*.PNG", "*.JPG"]:
        img_files.extend(list(train_img_dir.glob(ext)))
        
    if not img_files:
        print("검증할 이미지를 찾을 수 없습니다.")
        return

    sample_img_path = random.choice(img_files)
    sample_lbl_path = train_lbl_dir / f"{sample_img_path.stem}.txt"

    if not sample_lbl_path.exists():
        print(f"{sample_img_path.name}에 매칭되는 라벨 파일이 없습니다.")
        return

    print(f"\n[검증 진행] 샘플 이미지 대상: {sample_img_path.name}")

    img = PILImage.open(sample_img_path)
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    try:
        font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
        font = ImageFont.truetype(font_path, 24)
    except IOError:
        font = None

    with open(sample_lbl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
            
        class_id = int(parts[0])
        x_center, y_center, norm_w, norm_h = map(float, parts[1:])

        w = norm_w * img_w
        h = norm_h * img_h
        x1 = int((x_center * img_w) - (w / 2))
        y1 = int((y_center * img_h) - (h / 2))
        x2 = int(x1 + w)
        y2 = int(y1 + h)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        label_text = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        draw.text((x1 + 5, y1 - 32), label_text, fill="red", font=font)

    output_path = root_path / "verification_sample.png"
    img.save(output_path)
    print(f" └─ 검증 시각화 완료: {output_path.resolve()}")


if __name__ == "__main__":
    build_yolo_dataset(
        image_dir=ORIGINAL_IMAGES,
        label_dir=ORIGINAL_LABELS,
        output_root=OUTPUT_PROJECT_ROOT
    )