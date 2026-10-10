# 산불 CCTV AI 프로젝트

## 2026-10-10 최신 상태

YOLO26n 3epoch 시험 후 YOLO11n 본 학습 **70epoch**를 완료했습니다. 이후의 개선은 새 재학습이 아니라 미탐 분석·후처리 정책입니다. 현재 시작점은 [학습·서비스 인수인계](docs/TECHNICAL_HANDOFF.md)이며 [전체 개발 연혁](https://github.com/200-Pro/fire-detection/tree/main/docs/history)에서 초기 계획과 현재 구현을 구분해 확인할 수 있습니다.

이 저장소는 학습 코드입니다. FastAPI·React·CCTV·지도·확산 서비스는 [200-Pro/fire-detection](https://github.com/200-Pro/fire-detection)에 분리돼 있으므로 아래의 웹·확산 제외 범위는 학습 저장소 기준입니다. 모델·원본 데이터·실험 사진·API 키는 공개하지 않습니다. 이번 학습 코드 회귀 검사는 unittest 35개가 통과했습니다.

YOLO11n 전이학습으로 영상에서 `fire`와 `smoke`를 탐지하는 수업 프로젝트입니다. 이후 시간 필터·사건 생성·확산 예측·React 지도 서비스를 연결합니다. 현재 구현과 전체 기획 범위를 구분해 진행합니다.

**현재 시작점:** [두 사람 학습 준비 안내](docs/TRAINING_READY.md)와 `notebooks/dfire_team_training.ipynb`.
기본 학습 스위치는 OFF입니다. [모델 선정 기록](docs/MODEL_DECISION.md)을 참고하세요.

## 현재 범위

- 입력: 저장 이미지 또는 MP4
- 모델: Ultralytics YOLO11n
- 현재 추론 코드 출력: 프레임별 불꽃·연기 박스와 `detections.jsonl` (사건 판정은 후속 구현)
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
- YOLO26n 3epoch는 과거 시험 이력이며 새 본 학습은 YOLO11n 사용
- 데이터 전체 재검사에서 발견된 퇴화/경계 이탈 박스를 학습용 사본에서 정리하는 v2 코드 추가
- 다음 단계는 새 팀 노트북으로 사전 준비와 최대 80epoch 본 학습

자세한 상태는 [PROJECT_STATUS.md](docs/PROJECT_STATUS.md)를 확인합니다.

## Colab 데이터 준비

`notebooks/dfire_team_training.ipynb`를 각자 사본 저장하여 순서대로 실행합니다. 핵심 데이터 준비 명령은 다음과 같습니다.

```bash
python scripts/prepare_dfire.py \
  --dataset-zip "D-Fire.zip" \
  --split-zip "d-fire 텍스트 분할.zip" \
  --output "/content/fire-ai-data-v2" \
  --fold 1 --verify-images
```

원본 ZIP과 학습 결과는 Google Drive에 보관하고, 압축 해제된 학습 데이터는 Colab의 `/content`에 둡니다.
