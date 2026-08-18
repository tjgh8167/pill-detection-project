# 경구약제 이미지 객체 탐지 프로젝트

> 후보 객체 탐지 모델의 장단점을 비교하고 팀 토의를 거쳐 YOLOv11 Medium을 최종 모델로 선택한 뒤, 데이터 품질 개선을 통해 Kaggle Public Score 0.97162를 달성한 팀 프로젝트입니다.

## 프로젝트 요약

| 항목 | 내용 |
| --- | --- |
| 프로젝트 기간 | 2026.05.18 - 2026.06.05 |
| 목표 | 이미지 속 최대 4개 알약의 클래스와 위치를 탐지 |
| 문제 유형 | Object Detection, Multi-class Detection |
| 최종 선택 모델 | YOLOv11 Medium |
| 평가 지표 | mAP@[0.75:0.95], Kaggle Public Score |
| 최종 결과 | Public Score 0.97162, 4개 팀 중 2위 |
| 주요 개선 방향 | 모델 후보 비교 후 YOLOv11을 선택하고, 이후 데이터 품질 검증 중심으로 성능 개선 |

## 내가 맡은 역할

**정서호 / PM & Data Engineering**

- 팀장으로 일정 관리, 데일리 스크럼, 협업 방식, GitHub 작업 흐름을 구성했습니다.
- `src/dataloader.py`, `main.py`, `submission.py`를 중심으로 데이터 로딩, 전처리, 학습 실행, 제출 파일 생성 흐름을 관리했습니다.
- 이미지와 JSON 어노테이션의 매칭 여부를 점검해 누락 어노테이션 이미지를 식별하고 제거했습니다.
- 클래스 불균형을 확인하고, 소수 클래스 대상 Albumentations 기반 증강 파이프라인을 구성했습니다.
- 최종 발표 자료를 공동 제작했으며, 발표 후 질의응답은 전체 답변을 담당했습니다.

## 문제 정의

대회 데이터는 여러 알약이 한 이미지에 함께 등장하며, 각 알약의 `category_id`와 bounding box를 정확히 예측해야 했습니다. 단순히 모델을 학습시키는 것보다, 이미지와 어노테이션 파일이 정확히 연결되어 있는지, 클래스 분포가 지나치게 불균형하지 않은지, 제출 형식이 대회 규격과 맞는지가 성능에 큰 영향을 주는 과제였습니다.

## 접근 방식

### 1. 모델 후보 비교 및 선정

- 객체 탐지 과제에 적용할 수 있는 후보 모델들의 장단점을 비교했습니다.
- 팀 내 토의를 통해 bbox 정밀도, 구현 난이도, 학습/추론 효율, 대회 제출 파이프라인 적용 가능성을 비교했고 YOLOv11 Medium을 최종 선택했습니다.
- 선택한 모델을 기준으로 객체 탐지 파이프라인을 구성했습니다.
- `main.py`에서 데이터 전처리, 모델 생성, 학습, 평가, 예측 저장 과정을 한 번에 실행할 수 있도록 연결했습니다.

### 2. 데이터 품질 검증

- 학습 이미지와 JSON 어노테이션의 파일명, 알약 ID, 촬영 각도를 비교했습니다.
- 어노테이션이 누락된 이미지를 탐지해 최종적으로 8장을 제거했습니다.
- 이 과정에서 학습 데이터는 232장에서 224장으로 정제되었습니다.

### 3. YOLO 포맷 변환 및 증강

- 원본 JSON bbox를 YOLO 학습 형식으로 변환했습니다.
- train/validation split을 구성하고 실행 시 `data.yaml`을 자동 생성했습니다.
- 소수 클래스 인스턴스가 부족한 경우 Albumentations로 증강했습니다.

### 4. 학습 및 제출 자동화

- `main.py`에서 학습과 검증 결과가 시간별 폴더에 저장되도록 구성했습니다.
- `submission.py`에서 YOLO 예측 결과를 Kaggle 제출 CSV 형식으로 변환했습니다.
- YOLO 내부 클래스 ID를 대회 원본 category ID로 복원하는 로직을 포함했습니다.

## 주요 결과

| 구분 | 결과 |
| --- | --- |
| 초기 점수 | 0.95점대에서 정체 |
| 최종 Public Score | 0.97162 |
| 최종 순위 | 4개 팀 중 2위 |
| mAP@50 | 0.995 |
| mAP@[0.75:0.95] | 0.99471 |
| 제거한 누락 어노테이션 이미지 | 8장 |

## 기술 스택

- Python
- PyTorch
- Ultralytics YOLOv11
- OpenCV
- Albumentations
- scikit-learn
- pandas / numpy
- Kaggle submission workflow

## 프로젝트 구조

```text
pill-detection-project/
├── src/
│   ├── dataloader.py      # 데이터 무결성 검사, YOLO 변환, 증강
│   ├── model.py           # YOLOv11 모델 로드
│   ├── train.py           # 학습 실행
│   └── evaluate.py        # 검증 및 예측 결과 저장
├── notebook/
│   ├── PM/
│   ├── MA/
│   └── EL/
├── docs/
│   ├── contribution.md
│   ├── model_selection.md
│   ├── data_quality.md
│   └── experiment_summary.md
├── download_data.py
├── main.py
├── submission.py
├── requirements.txt
└── README.md
```

## 실행 방법

데이터와 모델 가중치는 용량 및 대회 데이터 정책상 저장소에 포함하지 않았습니다. 실행 시에는 원본 데이터를 아래 구조로 배치해야 합니다.

```text
data/
└── sprint_ai_project1_data/
    ├── train_images/
    ├── train_annotations/
    └── test_images/
```

```bash
pip install -r requirements.txt
python main.py
python submission.py
```

## 회고

이 프로젝트를 통해 객체 탐지 성능은 모델 선택이나 하이퍼파라미터만으로 결정되지 않는다는 점을 확인했습니다. 성능이 0.95점대에서 정체되었을 때 이미지와 어노테이션의 연결 상태를 다시 검증했고, 누락 데이터를 제거한 뒤 학습 파이프라인을 안정화했습니다.

결과적으로 데이터 품질 검증, 학습 실행, 제출 파일 생성까지 이어지는 전체 흐름을 점검하면서 모델 결과가 실제 평가 지표로 이어지기 위해 필요한 과정을 경험했습니다.

## 관련 문서

- [기여 정리](docs/contribution.md)
- [모델 선정 과정](docs/model_selection.md)
- [데이터 품질 개선 과정](docs/data_quality.md)
- [실험 및 결과 요약](docs/experiment_summary.md)
- [협업일지](https://docs.google.com/spreadsheets/d/17k2NNIQL951lEIIPhnsUgdwL5cIskf00ocd0drbfYgM/edit?gid=203327972#gid=203327972)
