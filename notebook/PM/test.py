import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms

class CustomCOCO:
    def __init__(self, annotation_file: str):
        with open(annotation_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.images = {img["file_name"]: img for img in self.data.get("images", [])}
        self.annotations = {}
        for ann in self.data.get("annotations", []):
            self.annotations.setdefault(ann["image_id"], []).append(ann)
        self.cats = {cat["id"]: cat for cat in self.data.get("categories", [])}

    def get_image_info(self, file_name: str) -> Optional[Dict]:
        return self.images.get(file_name)

    def get_annotations(self, image_id: int) -> List[Dict]:
        return self.annotations.get(image_id, [])


class COCODataset(Dataset):
    def __init__(self, root: str, train: bool, transform=None, image_paths: Optional[List[Path]] = None):
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.image_dir = self.root / ("train_images" if train else "test_images")
        self.annotations_root = self.root / "train_annotations" if train else None
        self.categories = {0: "background"}

        if image_paths is None:
            self.image_paths = sorted(
                [p for p in self.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            )
        else:
            self.image_paths = list(image_paths)

        self.annotation_index = self._build_annotation_index() if train else {}
        # 💡 메모리 절약을 위해 이미지 전체를 미리 로드하지 않고 경로 리스트만 유지합니다.

    def _build_annotation_index(self) -> Dict[str, Dict]:
        index = {}
        if self.annotations_root is None or not self.annotations_root.exists():
            return index

        for ann_file in sorted(self.annotations_root.rglob("*.json")):
            coco = CustomCOCO(str(ann_file))
            for file_name, img_info in coco.images.items():
                annotations = coco.get_annotations(img_info["id"])
                index[file_name] = {
                    "image_info": img_info,
                    "annotations": annotations,
                    "categories": coco.cats,
                }
                for cat_id, cat in coco.cats.items():
                    self.categories[cat_id] = cat["name"]
        return index

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 실제 배치가 요청될 때 이미지를 한 장씩 인메모리로 읽어옵니다 (OOM 방지)
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        
        target = {
            "image_id": torch.LongTensor([index]),
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

        if self.train:
            ann_record = self.annotation_index.get(image_path.name)
            if ann_record is not None:
                boxes = []
                labels = []
                for ann in ann_record["annotations"]:
                    x, y, w, h = ann["bbox"]
                    boxes.append([x, y, x + w, y + h]) # PyTorch Faster R-CNN 포맷
                    labels.append(ann["category_id"])

                if boxes:
                    target = {
                        "image_id": torch.LongTensor([ann_record["image_info"]["id"]]),
                        "boxes": torch.FloatTensor(boxes),
                        "labels": torch.LongTensor(labels),
                    }

        if self.transform is not None:
            image = self.transform(image)
            
        return image, target

    def __len__(self) -> int:
        return len(self.image_paths)


class TransformedSubset(Dataset):
    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        # 원본 데이터셋의 __getitem__을 호출하여 이미지를 가져온 뒤 transform만 재적용
        image, target = self.subset[idx]
        # 데이터셋 자체에 이미 텐서 변환이 들어가 있다면 중복 적용 안 되도록 유의
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.subset)


def detection_collate_fn(batch):
    return tuple(zip(*batch))


def get_loaders(
    data_path: str,
    batch_size: int = 16,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    # 데이터 증강 (Augmentation) 정의
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(dtype=torch.float),
    ])

    val_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(dtype=torch.float),
    ])
    test_transform = val_transform

    # Train / Val 분리
    full_train_dataset = COCODataset(data_path, train=True, transform=None)
    total = len(full_train_dataset)
    val_len = int(total * val_split)
    train_len = total - val_len
    
    train_subset, val_subset = random_split(
        full_train_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(seed),
    )

    train_dataset = TransformedSubset(train_subset, train_transform)
    val_dataset = TransformedSubset(val_subset, val_transform)
    test_dataset = COCODataset(data_path, train=False, transform=test_transform)

    # 💡 [수정] 전역 변수 BATCH_SIZE 대신 인자로 받은 소문자 batch_size를 연결했습니다.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=detection_collate_fn, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=detection_collate_fn, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=detection_collate_fn, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader

DATA_PATH = "./data/sprint_ai_project1_data"
BATCH_SIZE = 16

train_loader, val_loader, test_loader = get_loaders(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE)

print(f" └─ train_loader: {len(train_loader.dataset)} samples, val_loader: {len(val_loader.dataset)} samples, test_loader: {len(test_loader.dataset)} samples")