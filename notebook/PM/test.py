import os              # 파일 및 디렉토리 경로를 다루기 위한 표준 라이브러리
import json            # JSON 파일을 읽고 쓰기 위한 표준 라이브러리
from pathlib import Path
import torch           # 딥러닝 라이브러리 PyTorch (텐서 연산 등)
from PIL import Image  # 이미지를 다루기 위한 Pillow 라이브러리
from torch.utils.data import Dataset  # PyTorch의 Dataset 클래스를 상속받기 위한 모듈
from torchvision import transforms
from torch.utils.data import DataLoader

# COCO 데이터셋의 JSON 파일을 직접 파싱하기 위한 사용자 정의 클래스
class CustomCOCO:
    def __init__(self, annotation_file):
        """
        CustomCOCO 클래스 초기화 함수.
        :param annotation_file: COCO 데이터셋의 어노테이션(JSON) 파일 경로
        """
        # JSON 파일을 열어서 데이터를 읽습니다.
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # 이미지 정보를 'id'를 키로 하는 딕셔너리로 저장합니다.
        self.images = {img["id"]: img for img in self.data.get("images", [])}

        # 어노테이션(annotation) 정보를 이미지 id별로 그룹화합니다.
        self.annotations = {}
        for ann in self.data.get("annotations", []):
            img_id = ann["image_id"]
            self.annotations.setdefault(img_id, []).append(ann)

        # 카테고리 정보를 'id'를 키로 저장합니다.
        self.cats = {cat["id"]: cat for cat in self.data.get("categories", [])}

    def loadImgs(self, ids):
        return [self.images[i] for i in ids if i in self.images]

    def getAnnIds(self, imgIds):
        ann_ids = []
        for img_id in imgIds:
            ann_ids.extend([ann["id"] for ann in self.annotations.get(img_id, [])])
        return ann_ids

    def loadAnns(self, annIds):
        return [ann for ann in self.data.get("annotations", []) if ann["id"] in annIds]

# PyTorch Dataset 클래스를 상속받아 사용자 데이터셋을 처리하기 위한 클래스
class COCODataset(Dataset):
    def __init__(self, root, train, transform=None):
        """
        COCODataset 클래스 초기화 함수.
        :param root: 데이터셋의 최상위 디렉토리 경로
        :param train: 학습용 데이터(True)와 테스트용 데이터(False)를 구분
        :param transform: 이미지에 적용할 전처리 함수
        """
        super().__init__()
        self.root = Path(root)
        self.image_dir = self.root / ("train_images" if train else "test_images")
        self.annotations_root = self.root / "train_annotations" if train else None
        self.transform = transform

        self.categories = {0: "background"}
        self.annotation_index = self._build_annotation_index()
        self.data = self._load_data()

    def _build_annotation_index(self):
        index = {}
        if self.annotations_root is None or not self.annotations_root.exists():
            return index
        for ann_file in self.annotations_root.rglob("*.json"):
            index[ann_file.stem] = ann_file
        return index

    def _load_data(self):
        data = []
        if not self.image_dir.exists():
            raise FileNotFoundError(f"이미지 폴더를 찾을 수 없습니다: {self.image_dir}")

        image_files = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        for index, image_path in enumerate(image_files):
            ann_path = self.annotation_index.get(image_path.stem)
            if ann_path is None:
                image = Image.open(image_path).convert("RGB")
                target = {
                    "image_id": torch.LongTensor([index]),
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64)
                }
                data.append((image, target))
                continue

            coco = CustomCOCO(str(ann_path))
            if not coco.images:
                continue

            img_info = next(iter(coco.images.values()))
            image = Image.open(image_path).convert("RGB")

            boxes = []
            labels = []
            for ann in coco.annotations.get(img_info["id"], []):
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])

            for cat_id, cat in coco.cats.items():
                self.categories[cat_id] = cat["name"]

            target = {
                "image_id": torch.LongTensor([img_info["id"]]),
                "boxes": torch.FloatTensor(boxes),
                "labels": torch.LongTensor(labels)
            }
            data.append((image, target))

        return data

    def __getitem__(self, index):
        image, target = self.data[index]
        if self.transform:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.data)

transform = transforms.Compose(
    [
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(dtype=torch.float)
    ]
)

train_dataset = COCODataset("./data/sprint_ai_project1_data", train=True, transform=transform)
test_dataset = COCODataset("./data/sprint_ai_project1_data", train=False, transform=transform)

print(len(train_dataset))
print(len(test_dataset))
print(train_dataset[0])