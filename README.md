# 프로젝트미션_경구약제 이미지 객체 검출 프로젝트
- **환경세팅**: python = 3.14.5
- **데이터 다운(Local)**: `python3 download_data.py`
- **실행 방법** `python3 main.py` `python3 submission.py` 후 saved_model/에 저장된 가장 최신의 파일 제출

## 1. 프로젝트 개요
- **진행 기간**: 2026년 5월 18일 ~ 2026년 6월 5일
- **목표**: 사진 속에 있는 최대 4개의 알약의 이름(클래스)과 위치(바운딩 박스)를 검출하는 것입니다.
- **성능 평가**: mAP@[0.75:0.95] 지표를 사용하여 모델의 성능을 측정

## 2. 팀원 소개 및 역할

| 이름 | 역할 | 주요 업무 |
| :--- | :--- | :--- |
| **정서호** | **PM/DE** | 프로젝트 일정 관리, 데일리 스크럼 주도, GitHub/Notion 환경 세팅 및 파이프라인 총괄, 발표 자료 제작, 데이터 셋/로더 파이프라인 구축, 데이터 전처리 변환 |
| **박종선** | **MA** | 알약 검출을 위한 객체 검출 모델 선택, 베이스라인 코드 구현 및 개선 |
| **권소현** | **EL** | 하이퍼파라미터 튜닝, 모델 실험 및 기록, 이미지 시각화, 오차 분석(Error Analysis) 및 실험 결과(mAP) 기록 및 파라미터 최적화 |


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
│   ├── train.py                    # 모델 학습 (Experimentation Lead)
│   └── evaluate.py                 # 모델 평가 및 저장 (Experimentation Lead)
│
├── .gitignore                      #data/
├── download_data.py                # 데이터 다운로드 스크립트
├── main.py                         # 전체 파이프라인 실행 스크립트
├── submission.py                   # kaggle 제출 csv파일로 변환
└── README.md                       # 프로젝트 설명서
```

## 4. 팀 문서

| 목록| 협업일지 링크 |
| :--- | :--- |
| **협업일지** |[링크](https://docs.google.com/spreadsheets/d/17k2NNIQL951lEIIPhnsUgdwL5cIskf00ocd0drbfYgM/edit?gid=203327972#gid=203327972) |
| **보고서** |[링크](https://drive.google.com/drive/folders/1fV5U3a8G8sc8LJ7likgKH3rwgxokPT_q?usp=sharing) |
