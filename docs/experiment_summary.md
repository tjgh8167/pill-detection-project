# Experiment Summary

## 최종 설정

| 항목 | 값 |
| --- | --- |
| Final Model | YOLOv11 Medium |
| Epochs | 80 |
| Batch Size | 16 |
| Image Size | 1280 |
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 1e-4 |
| Mosaic | 1.0 |
| Degrees | 15.0 |
| Confidence Threshold | 0.25 |
| Max Detection | 4 |

## 성능 변화

| 구분 | 결과 |
| --- | --- |
| 초기 Public Score | 0.95점대 |
| 최종 Public Score | 0.97162 |
| 최종 순위 | 4개 팀 중 2위 |
| mAP@50 | 0.995 |
| mAP@[0.75:0.95] | 0.99471 |

## 개선 방향

초기에는 후보 모델의 장단점을 비교하고 팀 토의를 통해 YOLOv11 Medium을 최종 모델로 선정했습니다. 이후에는 선택한 모델을 기준으로 하이퍼파라미터와 데이터 처리 흐름을 조정했지만, 점수 개선이 제한적이었기 때문에 데이터 품질 검증으로 방향을 바꾸었습니다.

주요 개선 흐름:

1. 후보 모델 비교 및 YOLOv11 Medium 선정
2. 이미지와 annotation 매칭 여부 확인
3. 누락 annotation 이미지 8장 제거
4. YOLO format 변환 파이프라인 정리
5. 소수 클래스 대상 데이터 증강
6. 예측 결과를 Kaggle 제출 형식으로 변환하는 submission pipeline 정리

## 결과 요약

점수가 정체된 상황에서 모델 튜닝만 반복하지 않고, 데이터 품질과 제출 파이프라인을 점검해 최종 Public Score 0.97162와 4개 팀 중 2위를 달성했습니다.
