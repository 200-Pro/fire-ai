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

