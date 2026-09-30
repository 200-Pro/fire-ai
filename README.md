# 산불 CCTV AI 프로젝트

YOLO26n 전이학습으로 영상에서 `fire`와 `smoke`를 탐지하고, 반복 검출을 시간 필터로 묶어 사건 JSON을 생성하는 수업 프로젝트입니다.

## 현재 범위

- 입력: 저장 이미지 또는 MP4
- 모델: Ultralytics YOLO26n
- 출력: 불꽃·연기 바운딩 박스와 `fire_event.json`
- 학습: Google Colab GPU
- 로컬 PC: 데이터 검사, 코드 작성, 이미지·MP4 추론

## 현재 제외 범위

- 실시간 전국 CCTV 연결
- 확산예측 CA
- PostgreSQL/PostGIS
- 웹 지도와 배포
- 정확한 발화 지점 산출

위 항목은 탐지 파이프라인이 동작한 뒤 순서대로 추가합니다.

## 저장 위치

- 공유 코드: 이 저장소
- 로컬 가상환경: `C:\fire-ai-local\.venv`
- 로컬 데이터: `C:\fire-ai-local\data`
- 로컬 모델: `C:\fire-ai-local\weights`
- 로컬 결과: `C:\fire-ai-local\runs`
- MLflow 기록: `C:\fire-ai-local\mlruns`

## 빠른 확인

```powershell
& "C:\fire-ai-local\.venv\Scripts\python.exe" "scripts\verify_environment.py"
& "C:\fire-ai-local\.venv\Scripts\yolo.exe" checks
```

팀원 최초 설치는 [TEAM_SETUP.md](docs/TEAM_SETUP.md)를 따릅니다.

## 현재 진행상황

- D-Fire 이미지·라벨 21,527쌍 검증 완료
- 공식 클래스 `0=smoke`, `1=fire` 확인
- Colab GPU에서 YOLO26n 3epoch 시험 학습 완료
- 시험 결과 `best.pt`와 지표를 Google Drive에 보관
- 다음 단계는 공식 fold 기반 데이터 준비 재현과 80epoch 본 학습

자세한 상태는 [PROJECT_STATUS.md](docs/PROJECT_STATUS.md)를 확인합니다.

## Colab 데이터 준비

`notebooks/dfire_colab_pipeline.ipynb`를 Colab에서 열어 순서대로 실행합니다. 핵심 데이터 준비 명령은 다음과 같습니다.

```bash
python scripts/prepare_dfire.py \
  --dataset-zip "D-Fire.zip" \
  --split-zip "d-fire 텍스트 분할.zip" \
  --output "/content/fire-ai-data" \
  --fold 1
```

원본 ZIP과 학습 결과는 Google Drive에 보관하고, 압축 해제된 학습 데이터는 Colab의 `/content`에 둡니다.
