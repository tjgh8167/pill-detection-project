# 프로젝트미션_경구약제 이미지 객체 검출 프로젝트
- **환경세팅**: python = 3.14.5
- **데이터 다운(Local)**: `python3 download_data.py`

## 1. 프로젝트 개요
- **진행 기간**: 2026년 5월 18일 ~ 2026년 6월 5일
- **목표**: 사진 속에 있는 최대 4개의 알약의 이름(클래스)과 위치(바운딩 박스)를 검출하는 것입니다.
- **성능 평가**: mAP@[0.75:0.95] 지표를 사용하여 모델의 성능을 측정

## 2. 팀원 소개 및 역할

| 이름 | 역할 | 주요 업무 |
| :--- | :--- | :--- |
| **정서호** | **Project Manager** | 프로젝트 일정 관리, 데일리 스크럼 주도, GitHub/Notion 환경 세팅 및 파이프라인 총괄, 발표 자료 제작 |
| **김윤현** | **Data Engineer** | 알약 이미지 데이터셋 검수, 데이터 전처리/변환, 데이터 셋/로더 파이프라인 구축 |
| **박종선** | **Model Architect** | 알약 검출을 위한 객체 검출 모델 선택, 베이스라인 코드 구현 및 개선 |
| **권소현** | **Experimentation Lead** | 하이퍼파라미터 튜닝, 모델 실험, 이미지 시각화, 오차 분석(Error Analysis) 및 실험 결과(mAP) 기록 및 최적화 |


## 3. 프로젝트 구조
```text
project_team4-/
├── .github/
│   └── pull_request_template.md    
│
├── data/                           # 데이터셋 저장 폴더 (Github에서는 제외)
|    └── sprint_ai_project1_data
|     ├── test_images(.png)
|     ├── train_annotation (.json)
|     └── train_images (.png)
|
├── notebook/                       # 개인 타이핑 노트북
|
├── saved_models/                   # 학습 후 모델 저장 폴더 (Github에서는 제외)
|
├── src/                            # 기능별 파이썬 소스 코드
│   ├── __init__.py
│   ├── data_loader.py              # 데이터 전처리 및 로더 (Data Engineer)
│   ├── model.py                    # 모델 아키텍처 정의 (Model Architect)
│   └── train.py                    # 모델 학습 및 평가 (Experimentation Lead)
│
├── .gitignore                      #data/
├── download_data.py                # 데이터 다운로드 스크립트
├── main.py                         # 전체 파이프라인 실행 스크립트
└── README.md                       # 프로젝트 설명서
```

## 4. 팀원 협업일지

| 이름 | 협업일지 링크 |
| :--- | :--- |
| **권소현** | [Daily 협업일지 - 권소현](https://www.notion.so/AI-Daily-364f5217300d80e7955be3666c26d401) |
| **김윤현** | ... |
| **박종선** | ... |
| **정서호** | [Daily 협업일지 - 정서호](https://www.notion.so/4-_-365d795f8e0780a694e8eb5dfb2606b1?source=copy_link)|
