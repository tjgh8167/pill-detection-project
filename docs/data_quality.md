# Data Quality

## 문제 상황

초기 실험에서는 모델과 하이퍼파라미터를 조정해도 Kaggle Public Score가 0.95점대에서 정체되었습니다. 그래서 모델 튜닝만 계속하기보다, 학습 데이터 자체에 문제가 없는지 확인하는 방향으로 접근을 바꿨습니다.

## 확인한 항목

- 학습 이미지와 JSON annotation 파일이 서로 매칭되는지 확인
- 이미지 파일명에 포함된 알약 ID와 annotation 파일의 알약 ID가 일치하는지 확인
- 촬영 각도 정보가 같은 annotation이 존재하는지 확인
- bbox 정보를 YOLO 형식으로 변환할 때 누락되는 label이 없는지 확인
- 클래스별 instance 수가 지나치게 불균형하지 않은지 확인

## 처리 결과

| 항목 | 결과 |
| --- | --- |
| 원본 학습 이미지 | 232장 |
| 제거한 누락 annotation 이미지 | 8장 |
| 최종 정제 이미지 | 224장 |
| 클래스 수 | 56개 |

## 구현 방식

`src/dataloader.py`에서 이미지 파일명에 포함된 6자리 알약 ID와 annotation JSON 파일명을 비교했습니다. 각 이미지에 필요한 알약 조합과 촬영 각도에 맞는 annotation이 없으면 누락 데이터로 판단했습니다.

정제 후에는 train/validation split을 구성하고, 원본 bbox를 YOLO format으로 변환했습니다.

## 데이터 증강

소수 클래스의 instance 부족을 완화하기 위해 Albumentations 기반 증강을 적용했습니다.

적용한 주요 증강:

- HorizontalFlip
- VerticalFlip
- RandomRotate90
- RandomBrightnessContrast
- GaussianBlur
- HueSaturationValue

## 배운 점

객체 탐지 프로젝트에서는 bbox label 품질이 성능에 직접적인 영향을 줍니다. 모델 성능이 정체될 때는 모델 구조 변경보다 데이터 누락, label 오류, class imbalance를 먼저 확인하는 것이 더 효과적일 수 있다는 점을 체감했습니다.
