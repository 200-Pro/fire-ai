# 다음 수동 작업

> 2026-09-30 갱신: 새 본 학습은 YOLO11n과 `dfire_team_training.ipynb`를 사용합니다. 최신 단계는 [TRAINING_READY.md](TRAINING_READY.md)입니다. 아래 기존 YOLO26 고정 경로와 종료 후 복사 지침은 과거 작업 기록으로만 참고하세요.

현재 날짜는 2026년 9월 30일이며 계획상 5주차다. 자동화 파일과 3epoch 시험 결과는 준비됐고, 아래 작업은 사용자가 Colab 또는 데스크탑에서 직접 실행하고 결과를 확인해야 한다.

## 지금 Colab에서 할 일

1. `notebooks/dfire_colab_pipeline.ipynb`를 Google Colab으로 연다.
2. 런타임 유형을 GPU로 설정한다.
3. 위에서부터 데이터 준비와 라벨 시각화 셀까지 실행한다.
4. 출력된 split 수와 이미지 박스를 눈으로 확인한다.
5. 문제가 없으면 `RUN_FULL_TRAIN=True`로 바꿔 80epoch 본 학습을 시작한다.
6. 학습 종료 후 Google Drive의 `training_results/yolo26n_fire_v001`에 결과가 복사됐는지 확인한다.
7. validation 결과와 대표 FP/FN을 정리한 뒤에만 `RUN_TEST_EVAL=True`를 실행한다.

## 데스크탑에서 할 일

1. Google Drive 동기화 완료를 확인한다.
2. 프로젝트 폴더에서 다음 점검 명령을 실행한다.

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\audit_system.ps1"
```

3. RTX 4060 PC에서 `nvidia-smi`가 실행되는지 확인한다.
4. `torch.cuda.is_available()`가 `True`인지 확인한다.
5. `False`이면 CPU용 PyTorch를 제거하고 PyTorch 공식 선택기의 Windows, Pip, CUDA 명령으로 다시 설치한다.
6. 확정된 `best.pt`를 `C:\fire-ai-local\weights`에 복사하고 이미지 및 MP4 추론을 실행한다.

## 직접 작성해야 하는 기록

- 주차별 실제 투입시간
- 활동 사진과 화면 캡처
- 팀원별 역할 분담
- 지도교수 또는 멘토 면담 내용
- 대표 FP와 FN의 원인 및 개선 결정
- 실패한 실험과 재실행 이유

## 본 학습 후 할 일

- 모델카드 작성
- 학습에 사용하지 않은 MP4로 추론
- 시간 필터 구현
- 사건 recall, FP/hour, 탐지 지연, FPS 계산
- ONNX 내보내기 및 PyTorch 결과 비교

## 아직 하지 않을 일

- 실시간 전국 CCTV 연결
- API 키를 코드나 GitHub에 저장
- 확산예측과 웹 지도 구현
- test 결과를 보고 학습 설정을 반복 변경
- 실제 불을 피우는 실험
